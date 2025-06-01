import os
import json
from datetime import datetime
from typing import List, Dict, Optional
from collections import Counter
from soynlp.word import WordExtractor
from soynlp.tokenizer import RegexTokenizer

# Create memory cards directory if it doesn't exist
MEMORY_CARDS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "memoryCards")
os.makedirs(MEMORY_CARDS_DIR, exist_ok=True)

def extract_frequent_phrases(text: str, top_n: int = 20) -> List[Dict[str, str]]:
    """Extract top N most frequent phrases from Korean text"""
    try:
        # Initialize tokenizer
        tokenizer = RegexTokenizer()
        
        # Extract words
        words = tokenizer.tokenize(text)
        
        # Count frequencies
        word_counts = Counter(words)
        
        # Get top N words
        top_words = word_counts.most_common(top_n)
        
        # Convert to list of dictionaries
        result = []
        for word, frequency in top_words:
            if len(word.strip()) > 0:  # Skip empty words
                result.append({
                    "phrase": word,
                    "frequency": frequency
                })
        
        return result
    except Exception as e:
        print(f"Error extracting phrases: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def save_frequent_phrases_from_transcript(video_id: str, video_title: str, transcript_text: str, translations: Dict[str, str]) -> List[Dict]:
    """Extract and save top frequent phrases from a transcript"""
    # Extract frequent phrases
    frequent_phrases = extract_frequent_phrases(transcript_text)
    
    # Save each phrase as a memory card
    saved_cards = []
    for phrase_data in frequent_phrases:
        phrase = phrase_data["phrase"]
        frequency = phrase_data["frequency"]
        
        # Get translation for the phrase
        translation = translations.get(phrase, "")  # Empty string if no translation found
        
        # Save memory card
        memory_card = save_memory_card(
            video_id=video_id,
            video_title=video_title,
            phrase=phrase,
            translation=translation,
            frequency=frequency
        )
        saved_cards.append(memory_card)
    
    return saved_cards

def save_memory_card(video_id: str, video_title: str, phrase: str, translation: str, frequency: int) -> Dict:
    """Save a memory card to local storage"""
    # Create a unique filename using video_id and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"memory_card_{video_id}_{timestamp}.json"
    filepath = os.path.join(MEMORY_CARDS_DIR, filename)
    
    # Create memory card data
    memory_card = {
        "id": timestamp,  # Use timestamp as a simple ID
        "video_id": video_id,
        "video_title": video_title,
        "phrase": phrase,
        "translation": translation,
        "frequency": frequency,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    # Save to file
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(memory_card, f, ensure_ascii=False, indent=2)
    
    return memory_card

def get_memory_cards_by_video_id(video_id: str) -> List[Dict]:
    """Get all memory cards for a specific video"""
    memory_cards = []
    
    # Look for all memory card files for this video
    pattern = f"memory_card_{video_id}_*.json"
    for filename in os.listdir(MEMORY_CARDS_DIR):
        if filename.startswith(f"memory_card_{video_id}_"):
            filepath = os.path.join(MEMORY_CARDS_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    memory_card = json.load(f)
                    memory_cards.append(memory_card)
            except Exception as e:
                print(f"Error reading memory card file {filename}: {str(e)}")
    
    return memory_cards

def get_all_memory_cards() -> List[Dict]:
    """Get all memory cards"""
    memory_cards = []
    
    for filename in os.listdir(MEMORY_CARDS_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(MEMORY_CARDS_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    memory_card = json.load(f)
                    memory_cards.append(memory_card)
            except Exception as e:
                print(f"Error reading memory card file {filename}: {str(e)}")
    
    return memory_cards 