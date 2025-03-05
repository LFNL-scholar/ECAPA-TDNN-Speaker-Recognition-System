# ECAPA-TDNN Speaker Recognition System
# 基于ECAPA-TDNN的说话人识别系统

A CPU-compatible speaker recognition and annotation system based on SpeechBrain's ECAPA-TDNN architecture, providing robust speaker verification and identification capabilities.

基于 SpeechBrain 预训练的 ECAPA-TDNN 模型的说话人标注和识别系统。本项目支持在CPU环境下运行，使用 PyTorch 和 torchaudio 实现，主要功能包括：

## 功能特点

### 1. 说话人标注（Speaker Annotation）
- 支持音频文件的说话人信息标注
- 自动提取音频特征（Mel频谱图）
- 标注数据的JSON格式存储和管理
- 支持多说话人标注和管理

### 2. 说话人识别（Speaker Recognition）
- 基于ECAPA-TDNN的说话人特征提取
- 支持说话人验证（对比两段音频是否为同一说话人）
- 支持说话人聚类（对多段音频进行说话人分组）
- 提供相似度计算和阈值判断

## 项目结构
```
speaker_info_sys/
├── data/                    # 数据目录
│   ├── raw/                # 原始音频文件
│   ├── features/           # 提取的特征
│   └── annotations/        # 标注结果
├── models/                 # 模型目录
│   └── pretrained/        # 预训练模型
├── src/                    # 源代码
│   ├── annotation/        # 标注相关代码
│   ├── recognition/       # 识别相关代码
│   ├── features/          # 特征提取
│   └── utils/            # 工具函数
├── configs/               # 配置文件
└── scripts/              # 运行脚本
```

## 环境要求
- Python 3.7+
- CPU 或 CUDA 支持的 GPU
- 16GB+ RAM（推荐）
- 操作系统：Windows/Linux/MacOS

## 安装说明

1. 克隆项目：
```bash
git clone [项目地址]
cd speaker_info_sys
```

2. 创建虚拟环境（推荐）：
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 预训练模型：
- 系统首次运行时会自动下载ECAPA-TDNN预训练模型
- 模型将保存在 `models/pretrained/ecapa_tdnn` 目录下
- 如需手动下载，可访问：[ECAPA-TDNN Model](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)

## 使用说明

### 1. 说话人标注
```bash
python run.py --mode annotate --audio data/raw/your_audio.wav
```
标注过程将引导您：
1. 输入说话人信息（姓名、性别、年龄等）
2. 设置音频片段的时间范围
3. 输入说话内容
4. 自动保存标注结果

### 2. 说话人识别
```bash
# 单个音频处理
python run.py --mode recognize --audio data/raw/test.wav

# 对比两段音频
python run.py --mode recognize --audio data/raw/test1.wav --reference data/raw/test2.wav
```

### 配置说明
在 `configs/config.yaml` 中可以修改以下配置：
- 音频采样率
- 特征提取参数
- 识别阈值
- 聚类参数等

## 注意事项

1. 音频要求：
   - 采样率：16kHz（推荐）
   - 格式：WAV（推荐）
   - 时长：建议 1-30 秒
   - 质量：建议在安静环境下录制

2. 性能优化：
   - 已针对CPU环境优化
   - 支持批处理模式
   - 特征缓存机制

3. 数据管理：
   - 标注结果自动保存为JSON格式
   - 特征文件自动管理
   - 支持增量更新

## 常见问题

1. 模型下载失败：
   - 检查网络连接
   - 尝试手动下载并放置在指定目录

2. 音频处理错误：
   - 确认音频格式是否支持
   - 检查采样率是否正确

3. 内存不足：
   - 减小处理的音频长度
   - 关闭其他占用内存的程序

## 许可证
[MIT License]

## 贡献指南
欢迎提交Issue和Pull Request

## 联系方式
[lfnltech@163.com] 