import os
import json
from datetime import datetime
from typing import List, Dict, Optional
from collections import Counter
from soynlp.word import WordExtractor
from soynlp.tokenizer import RegexTokenizer, LTokenizer
from soynlp.normalizer import *
from transformers import MarianMTModel, MarianTokenizer

# Create memory cards directory if it doesn't exist
MEMORY_CARDS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "memoryCards")
os.makedirs(MEMORY_CARDS_DIR, exist_ok=True)

# Initialize Korean-English translator
def get_translator():
    model_name = "Helsinki-NLP/opus-mt-ko-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

def translate_text(text: str, model, tokenizer) -> str:
    """Translate Korean text to English using the MarianMT model"""
    try:
        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        
        # Generate translation
        translated = model.generate(**inputs)
        
        # Decode the translation
        translation = tokenizer.decode(translated[0], skip_special_tokens=True)
        return translation
    except Exception as e:
        print(f"Error translating text: {str(e)}")
        return ""

def extract_frequent_phrases(text: str, top_n: int = 20) -> List[Dict[str, str]]:
    """Extract top N most frequent meaningful phrases from Korean text"""
    try:
        # Normalize text
        text = repeat_normalize(text, num_repeats=2)
        
        # Split text into sentences for WordExtractor
        sentences = [s.strip() for s in text.replace('\n', '.').replace('!', '.').replace('?', '.').split('.') if s.strip()]
        
        # Initialize word extractor
        word_extractor = WordExtractor()
        word_extractor.train(sentences)
        word_scores = word_extractor.extract()
        
        # Get words and their scores
        words = list(word_scores.keys())
        
        # Common Korean particles and single characters to filter out
        stop_words = {'아', '야', '나', '이', '그', '저', '는', '은', '이', '가', '을', '를', '에', '의', '도', '만', '에서', '에게', '으로', '하고', '이랑', '랑', '과', '와'}
        
        # Filter out stop words and single characters
        meaningful_words = [word for word in words if len(word) > 1 and word not in stop_words]
        
        # Count word frequencies in the text
        word_counts = Counter()
        for word in meaningful_words:
            # Count exact matches
            word_counts[word] = text.count(word)
            
            # Also count words that are part of longer phrases
            for sentence in sentences:
                if word in sentence:
                    word_counts[word] += 1
        
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

def save_all_phrases_to_json(video_id: str, video_title: str, transcript_text: str, translations: Dict[str, str] = None, top_n: int = 20) -> Dict:
    """Extract top N frequent phrases from a transcript and save them all into a single JSON file."""
    try:
        # Extract frequent phrases
        frequent_phrases = extract_frequent_phrases(transcript_text, top_n=top_n)
        
        # Initialize translator
        model, tokenizer = get_translator()
        
        # Create a unique filename using video_id and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"memory_cards_{video_id}_{timestamp}.json"
        filepath = os.path.join(MEMORY_CARDS_DIR, filename)
        
        # Prepare the data to save
        data = {
            "id": timestamp,
            "video_id": video_id,
            "video_title": video_title,
            "phrases": []
        }
        
        # Add all phrases with translations
        for phrase_data in frequent_phrases:
            phrase = phrase_data["phrase"]
            # Get translation from provided translations or use the translator
            translation = ""
            if translations and phrase in translations:
                translation = translations[phrase]
            else:
                translation = translate_text(phrase, model, tokenizer)
            
            data["phrases"].append({
                "phrase": phrase,
                "translation": translation,
                "frequency": phrase_data["frequency"]
            })
        
        # Sort phrases by frequency
        data["phrases"].sort(key=lambda x: x["frequency"], reverse=True)
        
        # Save to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return data
    except Exception as e:
        print(f"Error saving phrases: {str(e)}")
        import traceback
        traceback.print_exc()
        return None 