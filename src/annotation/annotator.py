import json
import yaml
from pathlib import Path
import logging
import numpy as np
from datetime import datetime
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeakerAnnotator:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.min_duration = self.config['annotation']['min_segment_duration']
        self.max_duration = self.config['annotation']['max_segment_duration']
        self.overlap = self.config['annotation']['overlap']
        
        self.annotations_dir = Path(self.config['data']['annotations_dir'])
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载或创建说话人数据库
        self.speakers_db_path = self.annotations_dir / 'speakers_db.json'
        self.speakers_db = self._load_speakers_db()

    def _load_speakers_db(self):
        """加载或创建说话人数据库"""
        if self.speakers_db_path.exists():
            with open(self.speakers_db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            db = {
                'speakers': {},
                'next_speaker_id': 1
            }
            self._save_speakers_db(db)
            return db

    def _save_speakers_db(self, db):
        """保存说话人数据库"""
        with open(self.speakers_db_path, 'w', encoding='utf-8') as f:
            json.dump(db, f, ensure_ascii=False, indent=2)

    def add_speaker(self, name, metadata=None):
        """添加新说话人"""
        speaker_id = f"SPK_{self.speakers_db['next_speaker_id']:04d}"
        self.speakers_db['speakers'][speaker_id] = {
            'name': name,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat()
        }
        self.speakers_db['next_speaker_id'] += 1
        self._save_speakers_db(self.speakers_db)
        return speaker_id

    def get_speaker_info(self, speaker_id):
        """获取说话人信息"""
        return self.speakers_db['speakers'].get(speaker_id)

    def list_speakers(self):
        """列出所有说话人"""
        return self.speakers_db['speakers']

    def create_annotation(self, audio_path, segments):
        """创建音频标注"""
        audio_path = Path(audio_path)
        annotation = {
            'audio_file': str(audio_path),
            'duration': self._get_audio_duration(audio_path),
            'segments': segments,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        # 保存标注文件
        annotation_path = self.annotations_dir / f"{audio_path.stem}_annotation.json"
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Annotation saved to {annotation_path}")
        return annotation_path

    def _get_audio_duration(self, audio_path):
        """获取音频时长"""
        try:
            with sf.SoundFile(audio_path) as f:
                return len(f) / f.samplerate
        except Exception as e:
            logger.error(f"Error getting audio duration: {str(e)}")
            return None

    def validate_segment(self, segment):
        """验证片段有效性"""
        start_time = segment.get('start_time', 0)
        end_time = segment.get('end_time', 0)
        duration = end_time - start_time
        
        if duration < self.min_duration:
            return False, f"Segment duration {duration:.2f}s is less than minimum {self.min_duration}s"
        if duration > self.max_duration:
            return False, f"Segment duration {duration:.2f}s is more than maximum {self.max_duration}s"
        if not segment.get('speaker_id'):
            return False, "Speaker ID is required"
        
        return True, None

    def update_annotation(self, annotation_path, segments):
        """更新标注"""
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)
        
        annotation['segments'] = segments
        annotation['updated_at'] = datetime.now().isoformat()
        
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Updated annotation at {annotation_path}")
        return annotation

def main():
    """测试标注功能"""
    config_path = "configs/config.yaml"
    annotator = SpeakerAnnotator(config_path)
    
    # 添加测试说话人
    speaker_id = annotator.add_speaker("测试说话人", {"gender": "M", "age": 30})
    logger.info(f"Added speaker: {speaker_id}")
    
    # 创建测试标注
    test_segments = [
        {
            "start_time": 0.0,
            "end_time": 5.0,
            "speaker_id": speaker_id,
            "text": "测试音频片段1"
        }
    ]
    
    test_audio = "data/raw/test.wav"
    if Path(test_audio).exists():
        annotation_path = annotator.create_annotation(test_audio, test_segments)
        logger.info(f"Created test annotation at {annotation_path}")
    else:
        logger.warning(f"Test audio file {test_audio} does not exist")

if __name__ == "__main__":
    main() 