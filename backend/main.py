from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import os
import json
import traceback
import requests
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from datetime import timedelta, datetime
import torch
import re
from pytube import YouTube
from glob import glob
from collections import Counter
from database import init_db, MemoryCard, Transcript, Translation
from database.local_storage import save_all_phrases_to_json

app = FastAPI()

# Initialize database
init_db()

# Hugging Face Inference API Configuration
HF_API_BASE = "https://api-inference.huggingface.co/models"
HF_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")  # Get from environment variable
HF_ENABLED = HF_API_KEY is not None

print(f"🔧 Hugging Face API: {'Enabled' if HF_ENABLED else 'Disabled (no API key)'}")

# Update CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins during development
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get port from environment variable
port = int(os.environ.get("PORT", 8080))  # Changed default to 8080

os.makedirs("model_cache", exist_ok=True)
os.makedirs("transcripts", exist_ok=True)
os.makedirs("translations", exist_ok=True)

# Set environment variable for HuggingFace cache
os.environ["TRANSFORMERS_CACHE"] = "model_cache"

# Load the translator with better error handling
print("🚀 Loading translation models...")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Korean-English translator with retry mechanism
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Loading Korean-English translator (attempt {attempt + 1}/{max_retries})...")
            translator = pipeline(
                "translation",
                model="Helsinki-NLP/opus-mt-ko-en",
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ Korean-English translator loaded!")
            break
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                print("❌ Failed to load Korean-English translator after all attempts")
                translator = None

    # Load English-Tagalog translator with retry mechanism
    for attempt in range(max_retries):
        try:
            print(f"Loading English-Tagalog translator (attempt {attempt + 1}/{max_retries})...")
            en_tl_translator = pipeline(
                "translation",
                model="Helsinki-NLP/opus-mt-en-tl",
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ English-Tagalog translator loaded!")
            break
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                print("❌ Failed to load English-Tagalog translator after all attempts")
                en_tl_translator = None

    if translator is None and en_tl_translator is None:
        print("⚠️ No translation models loaded successfully")
    else:
        print("🎉 Translation models loaded successfully!")
except Exception as e:
    print(f"❌ Error in translation setup: {str(e)}")
    print("Will try to load models on first request instead...")
    translator = None
    en_tl_translator = None

LANG_MODEL_MAP = {
    "eng_Latn": "Helsinki-NLP/opus-mt-ko-en",
    "spa_Latn": "Helsinki-NLP/opus-mt-ko-es",
    "fra_Latn": "Helsinki-NLP/opus-mt-ko-fr",
    "zho_Hans": "Helsinki-NLP/opus-mt-en-zh",   # English to Chinese (Simplified)
    "arb_Arab": "Helsinki-NLP/opus-mt-en-ar",   # English to Arabic
    "tl_Latn": "Helsinki-NLP/opus-mt-en-tl",    # English to Tagalog
    "swe_Latn": "Helsinki-NLP/opus-mt-en-sv",   # English to Swedish
}

os.makedirs("output", exist_ok=True)
app.mount("/output", StaticFiles(directory="output"), name="output")

class VideoRequest(BaseModel):
    url: str
    start_time: Optional[int] = 0
    end_time: Optional[int] = None
    target_lang: Optional[str] = "eng_Latn"  # Default to English

class SummarizeRequest(BaseModel):
    transcript: str
    url: Optional[str] = None

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format"""
    return str(timedelta(seconds=int(seconds)))

def get_video_id(url: str) -> str:
    """Extract video ID from YouTube URL, handling various formats and parameters."""
    try:
        # Remove any URL parameters first
        base_url = url.split('?')[0]
        
        if "youtu.be" in base_url:
            return base_url.split("/")[-1]
        elif "youtube.com" in base_url:
            if "/v/" in base_url:
                return base_url.split("/v/")[1]
            elif "/watch" in base_url:
                # Handle watch URLs
                if "v=" in url:
                    return url.split("v=")[1].split("&")[0]
        
        # If we get here, try to find any 11-character YouTube ID in the URL
        import re
        video_id_match = re.search(r'[a-zA-Z0-9_-]{11}', url)
        if video_id_match:
            return video_id_match.group(0)
            
        raise ValueError("Could not extract video ID from URL")
    except Exception as e:
        print(f"Error extracting video ID: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid YouTube URL format")

def get_video_title(video_id: str) -> str:
    """Get the title of a YouTube video using YouTube Transcript API"""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Get video info from transcript list
        video_info = transcript_list.video_id
        if video_info:
            return f"video_{video_id}"
        return f"video_{video_id}"
    except Exception as e:
        print(f"Error getting video title: {str(e)}")
        return f"video_{video_id}"

def remove_repetition(text, max_repeat=1):
    # Replace more than max_repeat consecutive identical words with max_repeat
    return re.sub(r'(\b\w+\b)( \1\b){' + str(max_repeat) + r',}', lambda m: ' '.join([m.group(1)] * max_repeat), text)

def remove_phrase_repetition(text, max_repeat=1):
    # Remove repeated phrases of 2-6 words
    for n in range(6, 1, -1):
        pattern = r'((?:\\b\\w+\\b[ ,]*){' + str(n) + r'})(?: \\1){' + str(max_repeat) + r',}'
        text = re.sub(pattern, lambda m: m.group(1), text, flags=re.IGNORECASE)
    return text

def clean_text(text):
    return text.strip()

def translate_with_huggingface_api(text: str, model_name: str, max_retries: int = 3) -> str:
    """Translate text using Hugging Face Inference API"""
    if not HF_ENABLED:
        raise Exception("Hugging Face API not available")
    
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    api_url = f"{HF_API_BASE}/{model_name}"
    
    payload = {"inputs": text}
    
    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 503:
                # Model loading, wait and retry
                print(f"Model loading, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    raise Exception("Model still loading after retries")
            
            response.raise_for_status()
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                if 'translation_text' in result[0]:
                    return result[0]['translation_text']
                elif 'generated_text' in result[0]:
                    return result[0]['generated_text']
            
            raise Exception(f"Unexpected API response format: {result}")
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed (attempt {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                raise Exception(f"HF API failed after {max_retries} attempts: {str(e)}")
            
    return text  # Fallback

def summarize_transcript(transcript_text: str) -> str:
    """Simple local transcript summarization"""
    try:
        # Simple extractive summarization approach
        sentences = transcript_text.split('.')
        # Get first few sentences and some from middle and end
        summary_sentences = []
        
        if len(sentences) > 10:
            # Take first 3 sentences
            summary_sentences.extend(sentences[:3])
            # Take 2 from middle
            mid_point = len(sentences) // 2
            summary_sentences.extend(sentences[mid_point:mid_point+2])
            # Take last 2
            summary_sentences.extend(sentences[-3:-1])
        else:
            # For shorter transcripts, take first half
            summary_sentences = sentences[:len(sentences)//2+1]
        
        summary = '. '.join([s.strip() for s in summary_sentences if s.strip()])
        bullet_points = summary.replace('. ', '.\n• ').strip()
        
        formatted_summary = [
            "**Main Points from Video:**\n",
            f"• {bullet_points}\n",
            "**Key Takeaways:**\n",
            "This video covers important concepts with practical insights and explanations."
        ]
        
        return '\n'.join(formatted_summary)
        
    except Exception as e:
        print(f"Error in summarization: {str(e)}")
        # Fallback to simple truncation with structure
        words = transcript_text.split()
        if len(words) > 100:
            summary = ' '.join(words[:100]) + "..."
        else:
            summary = transcript_text
        
        return '\n'.join([
            "**Summary:**\n",
            summary,
            "\n**Note:** This is a basic summary of the transcript content."
        ])

def save_full_transcript(transcript_data: list, video_id: str) -> str:
    """Save the full original transcript in paragraph form"""
    # Create transcripts directory if it doesn't exist
    os.makedirs("transcripts", exist_ok=True)
    
    # Get video title
    video_title = get_video_title(video_id)
    # Sanitize video title for filename
    safe_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_title = safe_title.replace(' ', '_')
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"transcript_{timestamp}_{safe_title}_ko.txt"
    filepath = os.path.join("transcripts", filename)
    
    # Save transcript
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Video Title: {video_title}\n")
        f.write(f"Video ID: {video_id}\n")
        f.write(f"Language: Korean\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n" + "="*50 + "\n\n")
        
        # Save as paragraph
        full_transcript = " ".join(line['text'] for line in transcript_data)
        f.write(full_transcript)
    
    return filepath

def save_translation(translation_data: list, video_id: str, source_language: str, target_lang: str) -> str:
    """Save translation results to file"""
    try:
        print("Starting to save translation...")
        # Get video title
        video_title = get_video_title(video_id)
        print(f"Got video title: {video_title}")
        
        # Create translations directory if it doesn't exist
        translations_dir = os.path.join(os.getcwd(), "translations")
        print(f"Creating translations directory at: {translations_dir}")
        os.makedirs(translations_dir, exist_ok=True)
        
        # Sanitize video title for filename
        safe_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_title = safe_title.replace(' ', '_')
        print(f"Sanitized title: {safe_title}")
        
        # Create filename with timestamp and language info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"translation_{timestamp}_{safe_title}_{source_language[:2]}_to_{target_lang}.json"
        filepath = os.path.join(translations_dir, filename)
        print(f"Will save to file: {filepath}")
        
        # Prepare data to save
        save_data = {
            "video_title": video_title,
            "video_id": video_id,
            "source_language": source_language,
            "target_language": target_lang,
            "generated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "subtitle_count": len(translation_data),
            "translations": translation_data
        }
        
        # Save translation
        print("Writing translation to file...")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"Successfully saved translation to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Error saving translation: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Translation data length: {len(translation_data)}")
        return None

def load_existing_translation(video_id: str, target_lang: str) -> dict:
    """Check if a translation already exists for this video and target language"""
    try:
        translations_dir = os.path.join(os.getcwd(), "translations")
        if not os.path.exists(translations_dir):
            return None
        
        # Look for translation files for this video and target language
        pattern = os.path.join(translations_dir, f"*{video_id}*_to_{target_lang}.json")
        files = sorted(glob(pattern), reverse=True)  # Most recent first
        
        if files:
            print(f"Found existing translation file: {files[0]}")
            with open(files[0], "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        
        return None
        
    except Exception as e:
        print(f"Error loading existing translation: {str(e)}")
        return None

def extract_popular_phrases(text, min_length=2, top_n=20):
    """Extract popular phrases from Korean text using soynlp."""
    try:
        print(f"Extracting phrases from text of length: {len(text)}")
        
        from soynlp.word import WordExtractor
        from soynlp.tokenizer import LTokenizer
        
        # Initialize word extractor
        word_extractor = WordExtractor()
        # Train the word extractor
        word_extractor.train([text])
        # Get word scores
        word_scores = word_extractor.extract()
        
        # Initialize tokenizer with word scores
        tokenizer = LTokenizer(scores=word_scores)
        
        # Tokenize text
        tokens = tokenizer.tokenize(text)
        
        # Create phrases of 2-3 words
        phrases = []
        for i in range(len(tokens) - 1):
            if len(tokens[i]) >= min_length:
                phrases.append(tokens[i])
            if i < len(tokens) - 1:
                phrase = tokens[i] + " " + tokens[i + 1]
                if len(phrase) >= min_length:
                    phrases.append(phrase)
            if i < len(tokens) - 2:
                phrase = tokens[i] + " " + tokens[i + 1] + " " + tokens[i + 2]
                if len(phrase) >= min_length:
                    phrases.append(phrase)
        
        print(f"Created {len(phrases)} phrases")
        
        # Count frequencies
        phrase_counter = Counter(phrases)
        
        # Get top N phrases
        top_phrases = phrase_counter.most_common(top_n)
        print(f"Found {len(top_phrases)} top phrases")
        
        return top_phrases
    except Exception as e:
        print(f"Error in extract_popular_phrases: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())
        return []

def create_memory_cards(transcript_data, video_id, video_title):
    """Create memory cards from transcript data."""
    try:
        print("Starting memory card creation...")
        
        # Extract text from transcript data
        print("Extracting text from transcript...")
        if isinstance(transcript_data, list):
            # Handle list of transcript entries
            full_text = " ".join([entry['text'] for entry in transcript_data if isinstance(entry, dict) and 'text' in entry])
        elif isinstance(transcript_data, str):
            # Handle direct text input
            full_text = transcript_data
        else:
            print(f"Unexpected transcript data type: {type(transcript_data)}")
            return None
            
        print(f"Extracted text length: {len(full_text)}")
        
        if not full_text:
            print("No text found in transcript data")
            return None
        
        # Get popular phrases
        print("Getting popular phrases...")
        popular_phrases = extract_popular_phrases(full_text)
        
        if not popular_phrases:
            print("No popular phrases found")
            return None
        
        # Create memory cards
        print("Creating memory cards...")
        created_cards = []
        for phrase, count in popular_phrases:
            try:
                print(f"Translating phrase: {phrase}")
                # Translate phrase
                translation = translator(phrase)[0]['translation_text']
                
                # Create memory card in database
                card = MemoryCard.create(
                    video_id=video_id,
                    video_title=video_title,
                    phrase=phrase,
                    translation=translation,
                    frequency=count
                )
                
                if card:
                    created_cards.append(card)
                    print(f"Created card for: {phrase}")
            except Exception as e:
                print(f"Error creating memory card for phrase '{phrase}': {str(e)}")
                continue
        
        if not created_cards:
            print("No memory cards were created")
            return None
            
        print(f"Successfully created {len(created_cards)} memory cards")
        return created_cards
        
    except Exception as e:
        print(f"Error in create_memory_cards: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())
        return None

def save_transcript(transcript_data, video_id):
    try:
        print("Starting to save transcript...")
        # Get video title
        try:
            video_title = get_video_title(video_id)
            print(f"Got video title: {video_title}")
        except Exception as e:
            print(f"Error getting video title: {str(e)}")
            video_title = f"video_{video_id}"
            print(f"Using fallback title: {video_title}")

        # Format the transcript with proper line breaks
        formatted_text = ""
        current_paragraph = []
        last_start_time = 0
        
        for entry in transcript_data:
            if isinstance(entry, dict) and 'text' in entry:
                if "[음악]" in entry['text'] or \
                   (entry.get('start', 0) - last_start_time > 3) or \
                   len(current_paragraph) >= 3:
                    if current_paragraph:
                        formatted_text += " ".join(current_paragraph) + "\n\n"
                        current_paragraph = []
                
                current_paragraph.append(entry['text'])
                last_start_time = entry.get('start', 0)
            elif isinstance(entry, str):
                if "[음악]" in entry or len(current_paragraph) >= 3:
                    if current_paragraph:
                        formatted_text += " ".join(current_paragraph) + "\n\n"
                        current_paragraph = []
                current_paragraph.append(entry)
        
        # Add any remaining text
        if current_paragraph:
            formatted_text += " ".join(current_paragraph) + "\n\n"
        
        print(f"Full transcript length: {len(formatted_text)}")
        
        # Check if transcript already exists
        existing_transcript = Transcript.get_by_video_id(video_id)
        
        if existing_transcript:
            print(f"Updating existing transcript for video {video_id}")
            # Update existing transcript
            transcript = Transcript.update(
                video_id=video_id,
                video_title=video_title,
                korean_text=formatted_text
            )
        else:
            print(f"Creating new transcript for video {video_id}")
            # Create new transcript
            transcript = Transcript.create(
                video_id=video_id,
                video_title=video_title,
                korean_text=formatted_text
            )
        
        if not transcript:
            raise Exception("Failed to save transcript to database")

        # Translate and save English transcript
        print("Translating and saving English transcript...")
        try:
            if translator is None:
                raise Exception("Local translator not loaded")
            
            # Split text into smaller chunks for translation
            paragraphs = formatted_text.split("\n\n")
            translated_paragraphs = []
            
            batch_size = 8
            for i in range(0, len(paragraphs), batch_size):
                batch = [p for p in paragraphs[i:i+batch_size] if p.strip()]
                if not batch:
                    continue
                print(f"Translating paragraphs {i+1}-{i+len(batch)} of {len(paragraphs)}")
                try:
                    results = translator(batch)
                    for result in results:
                        translated_paragraphs.append(result['translation_text'])
                except Exception as e:
                    print(f"Error translating batch {i//batch_size+1}: {str(e)}")
                    translated_paragraphs.extend(batch)
            
            english_text = "\n\n".join(translated_paragraphs)
            print("Translation completed successfully")
            
            # Update transcript with English translation
            Transcript.update_english_text(video_id, english_text)
            
            # Create memory cards
            print("Creating memory cards...")
            popular_phrases = extract_popular_phrases(formatted_text)
            
            if popular_phrases:
                for phrase, count in popular_phrases:
                    try:
                        translation = translator(phrase)[0]['translation_text']
                        MemoryCard.create(
                            video_id=video_id,
                            video_title=video_title,
                            phrase=phrase,
                            translation=translation,
                            frequency=count
                        )
                    except Exception as e:
                        print(f"Error creating memory card for phrase '{phrase}': {str(e)}")
                        continue

            return transcript
        except Exception as e:
            print(f"Error during translation: {str(e)}")
            print("Full traceback:")
            print(traceback.format_exc())
            return transcript

    except Exception as e:
        print(f"Error in save_transcript: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())
        return None

@app.post("/translate")
async def translate_video(video_request: VideoRequest):
    try:
        print(f"📝 Starting translation request for URL: {video_request.url}")
        video_id = get_video_id(video_request.url)
        print(f"📌 Extracted video ID: {video_id}")
        
        try:
            video_title = get_video_title(video_id)
            print(f"📺 Got video title: {video_title}")
        except Exception as e:
            print(f"⚠️ Error getting video title: {str(e)}")
            video_title = f"video_{video_id}"
            print(f"Using fallback title: {video_title}")
        
        # Check if transcript already exists in database
        existing_transcript = Transcript.get_by_video_id(video_id)
        if existing_transcript:
            print(f"Found existing transcript for video {video_id}")
            transcript_file = existing_transcript
        else:
            # Get captions if transcript doesn't exist
            try:
                transcript_data, source_language = get_captions(video_id)
                if not transcript_data:
                    raise HTTPException(status_code=404, detail="No transcript found for this video. Please make sure the video has captions enabled.")
                print(f"📄 Got transcript in {source_language}")
                
                # Save transcript
                transcript_file = save_transcript(transcript_data, video_id)
                if not transcript_file:
                    raise HTTPException(status_code=500, detail="Failed to save transcript")
            except Exception as e:
                print(f"⚠️ Error getting captions: {str(e)}")
                raise HTTPException(status_code=404, detail=f"Could not get captions for this video: {str(e)}")
        
        # Extract text from transcript data
        transcript_text = existing_transcript["korean_text"] if existing_transcript else " ".join([item["text"] for item in transcript_data])
        
        # Save frequent phrases automatically
        memory_cards = save_all_phrases_to_json(
            video_id=video_id,
            video_title=video_title,
            transcript_text=transcript_text,
            top_n=20
        )
        
        # Translate the transcript
        target_lang = video_request.target_lang
        model_name = LANG_MODEL_MAP.get(target_lang)
        
        if not model_name:
            raise HTTPException(status_code=400, detail=f"Unsupported target language: {target_lang}")
        
        # Check if translation already exists
        existing_translation = load_existing_translation(video_id, target_lang)
        if existing_translation:
            print(f"Found existing translation for {target_lang}")
            return {
                "video_id": video_id,
                "video_title": video_title,
                "transcript_file": transcript_file,
                "translation_file": existing_translation.get("file", ""),
                "source_language": source_language if not existing_transcript else "Korean",
                "target_language": target_lang,
                "memory_cards": memory_cards
            }
        
        # Translate the text
        try:
            if HF_ENABLED:
                translated_text = translate_with_huggingface_api(transcript_text, model_name)
            else:
                # Use local translator
                if translator is None:
                    raise HTTPException(status_code=500, detail="Translation model not loaded")
                translated_text = translator(transcript_text)[0]["translation_text"]
        except Exception as e:
            print(f"⚠️ Error during translation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
        
        # Save translation
        translation_data = [{"text": translated_text}]
        translation_file = save_translation(translation_data, video_id, source_language if not existing_transcript else "Korean", target_lang)
        if not translation_file:
            raise HTTPException(status_code=500, detail="Failed to save translation")
        
        return {
            "video_id": video_id,
            "video_title": video_title,
            "transcript_file": transcript_file,
            "translation_file": translation_file,
            "source_language": source_language if not existing_transcript else "Korean",
            "target_language": target_lang,
            "memory_cards": memory_cards
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"❌ Error in translate_video: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

def get_captions(video_id: str) -> tuple:
    """Get captions from YouTube video using YouTube Transcript API"""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to get Korean captions first
        try:
            transcript = transcript_list.find_transcript(['ko'])
            return transcript.fetch(), "Korean"
        except:
            pass
            
        # Try English if Korean not available
        try:
            transcript = transcript_list.find_transcript(['en'])
            return transcript.fetch(), "English"
        except:
            pass
            
        # Get first available caption
        available_transcripts = list(transcript_list)
        if available_transcripts:
            transcript = available_transcripts[0]
            return transcript.fetch(), transcript.language
            
        return None, None
        
    except Exception as e:
        print(f"Error getting captions: {str(e)}")
        return None, None

@app.post("/get-transcript")
async def get_transcript(video_request: VideoRequest):
    try:
        video_id = get_video_id(video_request.url)
        video_title = get_video_title(video_id)
        
        # Get captions
        transcript_data, source_language = get_captions(video_id)
        if not transcript_data:
            raise HTTPException(status_code=404, detail="No transcript found")
        
        # Save transcript
        transcript_file = save_transcript(transcript_data, video_id)
        
        # Extract text from transcript data
        transcript_text = " ".join([item["text"] for item in transcript_data])
        
        # Save frequent phrases automatically
        memory_cards = save_all_phrases_to_json(
            video_id=video_id,
            video_title=video_title,
            transcript_text=transcript_text,
            top_n=20
        )
        
        return {
            "video_id": video_id,
            "video_title": video_title,
            "transcript_file": transcript_file,
            "source_language": source_language,
            "memory_cards": memory_cards
        }
        
    except Exception as e:
        print(f"Error in get_transcript: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/read-transcript")
async def read_transcript(video_request: VideoRequest):
    try:
        video_id = get_video_id(video_request.url)
        print(f"Reading transcript for video ID: {video_id}")
        
        # Try to get transcript from database
        transcript = Transcript.get_by_video_id(video_id)
        
        if transcript:
            print("Found transcript in database")
            return {
                "status": "success",
                "message": "Loaded saved transcript.",
                "korean_transcript": transcript["korean_text"],
                "english_transcript": transcript["english_text"]
            }
        
        # If not found in database, fetch and save new transcript
        print("Fetching new transcript...")
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            try:
                transcript_obj = transcript_list.find_transcript(['ko'])
                language = "Korean"
            except:
                try:
                    transcript_obj = transcript_list.find_transcript(['en'])
                    language = "English"
                except:
                    available_transcripts = list(transcript_list)
                    if not available_transcripts:
                        raise HTTPException(status_code=404, detail="No transcripts available for this video.")
                    transcript_obj = available_transcripts[0]
                    language = transcript_obj.language
            
            try:
                transcript_data = transcript_obj.fetch()
                print(f"Found {language} transcript with {len(transcript_data)} lines")
                
                # Save transcript
                transcript = save_transcript(transcript_data, video_id)
                
                if transcript:
                    return {
                        "status": "success",
                        "message": f"Fetched and saved new transcript.",
                        "korean_transcript": transcript["korean_text"],
                        "english_transcript": transcript["english_text"]
                    }
            except Exception as e:
                error_msg = str(e)
                print(f"Error fetching transcript: {error_msg}")
                if "no element found" in error_msg:
                    # Try alternative method using pytube
                    caption, caption_language = get_captions(video_id)
                    if caption:
                        caption_text = caption.generate_srt_captions()
                        # Convert SRT to readable text
                        formatted_text = ""
                        for line in caption_text.split('\n'):
                            if line.strip() and not line.isdigit() and not '-->' in line:
                                formatted_text += line.strip() + "\n"
                        
                        # Save the transcript
                        transcript = save_transcript([{"text": formatted_text}], video_id)
                        
                        if transcript:
                            return {
                                "status": "success",
                                "message": f"Fetched and saved new transcript using alternative method.",
                                "korean_transcript": transcript["korean_text"],
                                "english_transcript": transcript["english_text"]
                            }
                
                raise HTTPException(status_code=404, detail=f"No subtitles or captions found for this video. {error_msg}")
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error in read_transcript: {error_msg}")
            print("Full traceback:")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
                
    except Exception as e:
        print(f"Error in read_transcript: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize-transcript")
async def summarize_video_transcript(request: SummarizeRequest):
    """Summarize a video transcript using AI"""
    try:
        print(f"Received summarization request for transcript of length: {len(request.transcript)}")
        
        if not request.transcript or len(request.transcript.strip()) < 50:
            raise HTTPException(status_code=400, detail="Transcript is too short to summarize")
        
        # Generate summary
        summary = summarize_transcript(request.transcript)
        
        return {
            "status": "success",
            "message": "Transcript summarized successfully",
            "summary": summary,
            "original_length": len(request.transcript),
            "summary_length": len(summary)
        }
        
    except Exception as e:
        print(f"Error in summarize_video_transcript: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "YouTube Translator API is running"}

@app.get("/memory-cards/{video_id}")
async def get_memory_cards(video_id: str):
    try:
        # Get memory cards from database
        cards = MemoryCard.get_by_video_id(video_id)
        if cards:
            return {"status": "success", "cards": cards}
        else:
            return {"status": "error", "message": "Memory cards not found"}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}
