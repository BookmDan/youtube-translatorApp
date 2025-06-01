from .config import supabase
from .local_storage import (
    save_memory_card, 
    get_memory_cards_by_video_id, 
    get_all_memory_cards,
    save_frequent_phrases_from_transcript
)
from typing import List, Dict, Optional
import json

class MemoryCard:
    @staticmethod
    def create(video_id: str, video_title: str, phrase: str, translation: str, frequency: int) -> Dict:
        """Create a new memory card"""
        return save_memory_card(video_id, video_title, phrase, translation, frequency)

    @staticmethod
    def get_by_video_id(video_id: str) -> List[Dict]:
        """Get all memory cards for a video"""
        return get_memory_cards_by_video_id(video_id)

    @staticmethod
    def create_from_transcript(video_id: str, video_title: str, transcript_text: str, translations: Dict[str, str]) -> List[Dict]:
        """Create memory cards from transcript text"""
        return save_frequent_phrases_from_transcript(video_id, video_title, transcript_text, translations)

class Transcript:
    @staticmethod
    def create(video_id: str, video_title: str, korean_text: str, english_text: Optional[str] = None) -> Dict:
        """Create a new transcript"""
        data = {
            "video_id": video_id,
            "video_title": video_title,
            "korean_text": korean_text,
            "english_text": english_text
        }
        result = supabase.table("transcripts").insert(data).execute()
        return result.data[0] if result.data else None

    @staticmethod
    def get_by_video_id(video_id: str) -> Optional[Dict]:
        """Get transcript by video ID"""
        result = supabase.table("transcripts").select("*").eq("video_id", video_id).execute()
        return result.data[0] if result.data else None

    @staticmethod
    def update(video_id: str, video_title: str, korean_text: str, english_text: Optional[str] = None) -> Dict:
        """Update an existing transcript"""
        data = {
            "video_title": video_title,
            "korean_text": korean_text
        }
        if english_text is not None:
            data["english_text"] = english_text
        result = supabase.table("transcripts").update(data).eq("video_id", video_id).execute()
        return result.data[0] if result.data else None

    @staticmethod
    def update_english_text(video_id: str, english_text: str) -> Dict:
        """Update English translation of a transcript"""
        data = {"english_text": english_text}
        result = supabase.table("transcripts").update(data).eq("video_id", video_id).execute()
        return result.data[0] if result.data else None

class Translation:
    @staticmethod
    def create(video_id: str, source_language: str, target_language: str, translation_data: List[Dict]) -> Dict:
        """Create a new translation"""
        data = {
            "video_id": video_id,
            "source_language": source_language,
            "target_language": target_language,
            "translation_data": json.dumps(translation_data)
        }
        result = supabase.table("translations").insert(data).execute()
        return result.data[0] if result.data else None

    @staticmethod
    def get_by_video_id(video_id: str, target_language: str) -> Optional[Dict]:
        """Get translation by video ID and target language"""
        result = supabase.table("translations")\
            .select("*")\
            .eq("video_id", video_id)\
            .eq("target_language", target_language)\
            .execute()
        if result.data:
            data = result.data[0]
            data["translation_data"] = json.loads(data["translation_data"])
            return data
        return None 