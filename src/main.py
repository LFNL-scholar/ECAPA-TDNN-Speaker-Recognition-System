import argparse
import logging
from pathlib import Path
from src.annotation.annotator import SpeakerAnnotator
from src.recognition.speaker_recognizer import SpeakerRecognizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_argparse():
    parser = argparse.ArgumentParser(description='说话人标注和识别系统')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                      help='配置文件路径')
    parser.add_argument('--mode', type=str, required=True,
                      choices=['annotate', 'recognize'],
                      help='运行模式：annotate（标注）或 recognize（识别）')
    parser.add_argument('--audio', type=str, required=True,
                      help='音频文件路径')
    parser.add_argument('--reference', type=str,
                      help='参考音频文件路径（仅用于识别模式）')
    return parser

def annotate_audio(config_path, audio_path):
    """音频标注模式"""
    annotator = SpeakerAnnotator(config_path)
    
    # 添加新说话人
    print("\n=== 添加说话人信息 ===")
    name = input("请输入说话人姓名: ")
    gender = input("请输入性别 (M/F): ")
    age = input("请输入年龄: ")
    
    metadata = {
        "gender": gender,
        "age": int(age) if age.isdigit() else None
    }
    
    speaker_id = annotator.add_speaker(name, metadata)
    print(f"\n已添加说话人: {speaker_id}")
    
    # 创建标注
    print("\n=== 创建音频标注 ===")
    start_time = float(input("请输入起始时间（秒）: "))
    end_time = float(input("请输入结束时间（秒）: "))
    text = input("请输入说话内容: ")
    
    segments = [{
        "start_time": start_time,
        "end_time": end_time,
        "speaker_id": speaker_id,
        "text": text
    }]
    
    # 验证片段
    valid, message = annotator.validate_segment(segments[0])
    if not valid:
        print(f"\n错误: {message}")
        return
    
    # 保存标注
    annotation_path = annotator.create_annotation(audio_path, segments)
    print(f"\n标注已保存至: {annotation_path}")

def recognize_audio(config_path, audio_path, reference_path=None):
    """音频识别模式"""
    recognizer = SpeakerRecognizer(config_path)
    
    # 提取测试音频的特征
    print("\n=== 处理测试音频 ===")
    result = recognizer.process_audio(audio_path)
    
    if result is None:
        print("错误: 无法处理音频文件")
        return
    
    print(f"已成功提取特征向量，维度: {result['embedding'].shape}")
    
    # 如果提供了参考音频，进行说话人验证
    if reference_path:
        print("\n=== 进行说话人验证 ===")
        ref_result = recognizer.process_audio(reference_path)
        
        if ref_result is None:
            print("错误: 无法处理参考音频文件")
            return
        
        verification = recognizer.verify_speaker(
            ref_result['embedding'],
            result['embedding']
        )
        
        print("\n验证结果:")
        print(f"相似度: {verification['similarity']:.4f}")
        print(f"阈值: {verification['threshold']}")
        print(f"是否为同一说话人: {'是' if verification['is_same_speaker'] else '否'}")

def main():
    parser = setup_argparse()
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.audio).exists():
        print(f"错误: 音频文件不存在: {args.audio}")
        return
    
    if args.reference and not Path(args.reference).exists():
        print(f"错误: 参考音频文件不存在: {args.reference}")
        return
    
    if args.mode == 'annotate':
        annotate_audio(args.config, args.audio)
    else:  # recognize
        recognize_audio(args.config, args.audio, args.reference)

if __name__ == "__main__":
    main() 