import torch
import torch.nn.functional as F
import yaml
from pathlib import Path
import logging
import numpy as np
from sklearn.cluster import KMeans
from src.features.audio_features import AudioFeatureExtractor
import torchaudio
import requests
import json
import os
import shutil
from collections import OrderedDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置环境变量，强制使用复制而不是符号链接
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_ENABLE_SYMLINKS'] = '0'

class ECAPA_TDNN(torch.nn.Module):
    """ECAPA-TDNN 模型结构"""
    def __init__(self, input_size=80, channels=[512, 512, 512, 512, 1536], kernel_sizes=[5,3,3,3,1]):
        super().__init__()
        
        self.conv1 = torch.nn.Conv1d(input_size, channels[0], kernel_sizes[0], padding=kernel_sizes[0]//2)
        self.relu = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(channels[0])
        
        self.conv2 = torch.nn.Conv1d(channels[0], channels[1], kernel_sizes[1], padding=kernel_sizes[1]//2)
        self.bn2 = torch.nn.BatchNorm1d(channels[1])
        
        self.conv3 = torch.nn.Conv1d(channels[1], channels[2], kernel_sizes[2], padding=kernel_sizes[2]//2)
        self.bn3 = torch.nn.BatchNorm1d(channels[2])
        
        self.conv4 = torch.nn.Conv1d(channels[2], channels[3], kernel_sizes[3], padding=kernel_sizes[3]//2)
        self.bn4 = torch.nn.BatchNorm1d(channels[3])
        
        self.conv5 = torch.nn.Conv1d(channels[3], channels[4], kernel_sizes[4], padding=kernel_sizes[4]//2)
        self.bn5 = torch.nn.BatchNorm1d(channels[4])
        
        self.attention = torch.nn.Sequential(
            torch.nn.Conv1d(channels[4], 128, 1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Conv1d(128, channels[4], 1),
            torch.nn.Softmax(dim=2),
        )
        
        self.fc = torch.nn.Linear(channels[4], 192)
        
    def forward(self, x):
        """
        输入: x shape [batch, mel_bins, time]
        输出: embedding shape [batch, embedding_dim]
        """
        # 打印输入维度以便调试
        logger.info(f"Model input shape: {x.shape}")
        
        # 确保输入形状正确
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [mel_bins, time] -> [1, mel_bins, time]
        
        # 通过卷积层
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        
        x = self.conv4(x)
        x = self.relu(x)
        x = self.bn4(x)
        
        x = self.conv5(x)
        x = self.relu(x)
        x = self.bn5(x)
        
        # 应用注意力机制
        w = self.attention(x)
        x = torch.sum(x * w, dim=2)
        
        # 最终的线性层
        x = self.fc(x)
        return x

class SpeakerRecognizer:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化特征提取器
        self.feature_extractor = AudioFeatureExtractor(config_path)
        
        # 加载预训练模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # 设置模型保存目录
        self.model_dir = Path("models/pretrained/ecapa_tdnn")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 下载并加载模型
        try:
            self._prepare_model()
            self.model = self._load_model()
            self.model.eval()
            logger.info("Successfully loaded the model")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
        self.threshold = self.config['recognition']['threshold']

    def _prepare_model(self):
        """准备模型文件"""
        model_url = 'https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/embedding_model.ckpt'
        model_path = self.model_dir / 'embedding_model.ckpt'
        
        if not model_path.exists():
            logger.info("Downloading model...")
            response = requests.get(model_url)
            with open(model_path, 'wb') as f:
                f.write(response.content)

    def _load_model(self):
        """加载ECAPA-TDNN模型"""
        # 创建模型实例
        model = ECAPA_TDNN()
        
        # 加载预训练权重
        checkpoint = torch.load(self.model_dir / 'embedding_model.ckpt', map_location=self.device)
        
        # 处理权重键名
        state_dict = OrderedDict()
        for k, v in checkpoint.items():
            if k.startswith('encoder.'):
                k = k.replace('encoder.', '')
                if k.startswith('blocks.'):
                    continue
                state_dict[k] = v
        
        # 加载权重
        model.load_state_dict(state_dict, strict=False)
        return model.to(self.device)

    def extract_embedding(self, audio_path):
        """提取说话人嵌入向量"""
        try:
            # 加载并预处理音频
            waveform, sample_rate = torchaudio.load(audio_path)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # 确保音频是单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 提取mel频谱特征
            features = self.feature_extractor.extract_mel_spectrogram(waveform)
            
            # 调整特征维度 [batch, mel_bins, time] -> [batch, mel_bins, time]
            # features 已经是正确的维度 [1, 80, time]，不需要转置
            features = features.to(self.device)
            
            # 打印特征维度以便调试
            logger.info(f"Feature shape: {features.shape}")
            
            # 提取嵌入向量
            with torch.no_grad():
                embedding = self.model(features)
                # L2标准化
                embedding = F.normalize(embedding, p=2, dim=1)
            
            return embedding[0].cpu().numpy()
        
        except Exception as e:
            logger.error(f"Error extracting embedding: {str(e)}")
            logger.error(f"Error details: {str(e.__class__.__name__)}")
            return None

    def compute_similarity(self, embedding1, embedding2):
        """计算两个嵌入向量的余弦相似度"""
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def verify_speaker(self, reference_embedding, test_embedding):
        """说话人验证"""
        similarity = self.compute_similarity(reference_embedding, test_embedding)
        is_same_speaker = similarity >= self.threshold
        return {
            'is_same_speaker': is_same_speaker,
            'similarity': float(similarity),
            'threshold': self.threshold
        }

    def cluster_speakers(self, embeddings, n_clusters=None):
        """说话人聚类"""
        if not self.config['recognition']['use_clustering']:
            return None
        
        if n_clusters is None:
            n_clusters = min(len(embeddings), self.config['recognition']['max_speakers'])
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        return {
            'labels': labels,
            'centroids': kmeans.cluster_centers_
        }

    def process_audio(self, audio_path, reference_embeddings=None):
        """处理音频文件"""
        embedding = self.extract_embedding(audio_path)
        if embedding is None:
            return None
        
        result = {
            'embedding': embedding,
            'audio_path': str(audio_path)
        }
        
        # 如果提供了参考嵌入向量，进行说话人验证
        if reference_embeddings is not None:
            verifications = []
            for ref_name, ref_embedding in reference_embeddings.items():
                verification = self.verify_speaker(ref_embedding, embedding)
                verifications.append({
                    'reference_name': ref_name,
                    **verification
                })
            result['verifications'] = verifications
        
        return result

def main():
    """测试说话人识别"""
    config_path = "configs/config.yaml"
    recognizer = SpeakerRecognizer(config_path)
    
    # 测试音频处理
    test_audio = "data/raw/test.wav"
    if Path(test_audio).exists():
        result = recognizer.process_audio(test_audio)
        if result:
            logger.info(f"Successfully processed {test_audio}")
            logger.info(f"Embedding shape: {result['embedding'].shape}")
        else:
            logger.error(f"Failed to process {test_audio}")
    else:
        logger.warning(f"Test audio file {test_audio} does not exist")

if __name__ == "__main__":
    main() 