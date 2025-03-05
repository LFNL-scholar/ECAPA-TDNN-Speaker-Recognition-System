import torch
import torchaudio
import numpy as np
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioFeatureExtractor:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.sample_rate = self.config['data']['sample_rate']
        self.window_size = self.config['feature_extraction']['window_size']
        self.hop_size = self.config['feature_extraction']['hop_size']
        self.mel_bins = self.config['feature_extraction']['mel_bins']
        
        # 初始化mel频谱特征提取器
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=int(self.window_size * self.sample_rate / 1000),
            hop_length=int(self.hop_size * self.sample_rate / 1000),
            n_mels=self.mel_bins
        )
        
    def load_audio(self, audio_path):
        """加载音频文件并重采样"""
        try:
            waveform, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            return waveform
        except Exception as e:
            logger.error(f"Error loading audio file {audio_path}: {str(e)}")
            return None

    def extract_mel_spectrogram(self, waveform):
        """提取mel频谱特征"""
        try:
            with torch.no_grad():
                # 确保输入维度正确 [batch, time]
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                
                # 提取mel频谱 [batch, n_mels, time]
                mel_spec = self.mel_transform(waveform)
                
                # 对数变换
                mel_spec = torch.log1p(mel_spec)
                
                # 标准化
                if self.config['feature_extraction']['normalize']:
                    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
                
                # 打印特征维度以便调试
                logger.info(f"Mel spectrogram shape: {mel_spec.shape}")
                
                return mel_spec
                
        except Exception as e:
            logger.error(f"Error in mel spectrogram extraction: {str(e)}")
            logger.error(f"Waveform shape: {waveform.shape}")
            raise

    def extract_features(self, audio_path):
        """提取音频特征"""
        waveform = self.load_audio(audio_path)
        if waveform is None:
            return None
        
        features = {
            'mel_spectrogram': self.extract_mel_spectrogram(waveform),
            'waveform': waveform
        }
        return features

    def save_features(self, features, save_path):
        """保存特征"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(features, save_path)
        logger.info(f"Features saved to {save_path}")

    def process_file(self, audio_path, save_dir):
        """处理单个音频文件"""
        audio_path = Path(audio_path)
        save_path = Path(save_dir) / f"{audio_path.stem}_features.pt"
        
        features = self.extract_features(audio_path)
        if features is not None:
            self.save_features(features, save_path)
            return save_path
        return None

def main():
    """测试特征提取"""
    config_path = "configs/config.yaml"
    extractor = AudioFeatureExtractor(config_path)
    
    # 测试音频文件
    test_audio = "data/raw/test.wav"
    save_dir = "data/features"
    
    if Path(test_audio).exists():
        result = extractor.process_file(test_audio, save_dir)
        if result:
            logger.info(f"Successfully processed {test_audio}")
        else:
            logger.error(f"Failed to process {test_audio}")
    else:
        logger.warning(f"Test audio file {test_audio} does not exist")

if __name__ == "__main__":
    main() 