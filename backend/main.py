import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
import asyncio
from concurrent.futures import ThreadPoolExecutor
from .database.local_storage import save_frequent_phrases_from_transcript, get_memory_cards_by_video_id, memory_card_exists_for_video, save_all_phrases_to_json, get_latest_memory_cards_file
from hangul_romanize import Transliter
from hangul_romanize.rule import academic

app = FastAPI()

# Hugging Face Inference API Configuration
HF_API_BASE = "https://api-inference.huggingface.co/models"
HF_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")  # Get from environment variable
HF_ENABLED = HF_API_KEY is not None

print(f"🔧 Hugging Face API: {'Enabled' if HF_ENABLED else 'Disabled (no API key)'}")

# Update CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get port from environment variable
port = int(os.environ.get("PORT", 8080))  # Changed default to 8080

os.makedirs("model_cache", exist_ok=True)
os.makedirs("output", exist_ok=True)
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
    try:
        if "youtu.be" in url:
            return url.split("/")[-1].split("?")[0]
        elif "youtube.com" in url:
            return url.split("v=")[1].split("&")[0]
        else:
            raise ValueError("Invalid YouTube URL format")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL format")

def get_video_title(video_id: str) -> str:
    """Get the title of a YouTube video"""
    try:
        # First try using pytube
        try:
            yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
            return yt.title
        except Exception as e:
            print(f"Pytube failed to get title: {str(e)}")
            
        # Fallback: Try using YouTube API
        try:
            response = requests.get(
                f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
            )
            if response.status_code == 200:
                return response.json()['title']
        except Exception as e:
            print(f"YouTube API failed to get title: {str(e)}")
            
        # If all methods fail, return a fallback title
        return f"video_{video_id}"
        
    except Exception as e:
        print(f"Error getting video title: {str(e)}")
        return f"video_{video_id}"

def get_next_output_dir(base_dir="output"):
    existing = [int(d) for d in os.listdir(base_dir) if d.isdigit()]
    next_num = max(existing, default=0) + 1
    new_dir = os.path.join(base_dir, str(next_num))
    os.makedirs(new_dir, exist_ok=True)
    return new_dir

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

def load_existing_translation(video_id: str, target_lang: str) -> dict:
    """Check if a translation already exists for this video and target language"""
    try:
        translations_dir = os.path.join(os.getcwd(), "translations")
        if not os.path.exists(translations_dir):
            return None
        
        # Look for translation files for this video ID
        pattern = os.path.join(translations_dir, f"*{video_id}*.json")
        files = sorted(glob(pattern), reverse=True)  # Most recent first
        
        if files:
            print(f"Found existing translation files: {files}")
            
            # Try each file until we find one with translations
            for file in files:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("translations") and len(data["translations"]) > 0:
                        print(f"Loaded cached translation with {len(data['translations'])} lines from {file}")
                        return data
                    else:
                        print(f"Skipping empty translation file: {file}")
            
            print("No non-empty translation files found")
            return None
        
        return None
        
    except Exception as e:
        print(f"Error loading existing translation: {str(e)}")
        return None

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
        
        # Create filename with video ID and target language
        filename = f"translation_{video_id}_{target_lang}.json"
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

def save_transcript(transcript_data, video_id):
    try:
        print("Starting to save transcript...")
        # Get video title
        try:
            video = YouTube(f"https://www.youtube.com/watch?v={video_id}")
            video_title = video.title
            print(f"Got video title: {video_title}")
        except Exception as e:
            print(f"Error getting video title: {str(e)}")
            video_title = f"video_{video_id}"
            print(f"Using fallback title: {video_title}")

        # Create transcripts directory if it doesn't exist
        transcripts_dir = os.path.join(os.path.dirname(__file__), "transcripts")
        print(f"Creating transcripts directory at: {transcripts_dir}")
        os.makedirs(transcripts_dir, exist_ok=True)

        # Sanitize the title for use in filenames
        sanitized_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        print(f"Sanitized title: {sanitized_title}")

        # Format the transcript
        formatted_transcript = ""
        current_paragraph = []

        for line in transcript_data:
            if isinstance(line, dict):
                text = line.get('text', '')
            else:
                text = line.text if hasattr(line, 'text') else str(line)
            
            if text:
                current_paragraph.append(text)
                # Start new paragraph if we hit a music marker or long pause
                if any("[음악]" in text for text in current_paragraph) or (len(current_paragraph) > 1 and isinstance(line, dict) and 'start' in line and isinstance(transcript_data[transcript_data.index(line)-1], dict) and 'start' in transcript_data[transcript_data.index(line)-1] and line['start'] - transcript_data[transcript_data.index(line)-1]['start'] > 2):
                    formatted_transcript += " ".join(current_paragraph) + "\n\n"
                    current_paragraph = []

        # Add any remaining text
        if current_paragraph:
            formatted_transcript += " ".join(current_paragraph)
        
        print(f"Full transcript length: {len(formatted_transcript)}")
        
        # Save Korean transcript
        ko_file = os.path.join(transcripts_dir, f"{sanitized_title}_{video_id}_ko.txt")
        print("Writing Korean transcript to file...")
        with open(ko_file, "w", encoding="utf-8") as f:
            f.write(f"Video Title: {video_title}\n")
            f.write(f"Video ID: {video_id}\n")
            f.write("="*50 + "\n")
            f.write(formatted_transcript)
        print("Korean transcript saved successfully")

        # Translate and save English transcript
        print("Translating and saving English transcript...")
        try:
            if translator is None:
                raise Exception("Local translator not loaded")
            
            # Split text into smaller chunks for translation
            chunks = [formatted_transcript[i:i+500] for i in range(0, len(formatted_transcript), 500)]
            translated_chunks = []
            
            for i, chunk in enumerate(chunks):
                print(f"Translating chunk {i+1}/{len(chunks)}")
                try:
                    translated = translator(chunk)[0]['translation_text']
                    translated_chunks.append(translated)
                except Exception as e:
                    print(f"Error translating chunk {i+1}: {str(e)}")
                    translated_chunks.append(chunk)  # Keep original if translation fails
            
            english_text = " ".join(translated_chunks)
            print("Translation completed successfully")
            
            # Save English transcript
            en_file = os.path.join(transcripts_dir, f"{sanitized_title}_{video_id}_en.txt")
            print("Writing English transcript to file...")
            with open(en_file, "w", encoding="utf-8") as f:
                f.write(f"Video Title: {video_title}\n")
                f.write(f"Video ID: {video_id}\n")
                f.write("="*50 + "\n")
                f.write(english_text)
            print("English transcript saved successfully")
            
            # Also save the translation data for the translate endpoint
            translations = []
            for i, line in enumerate(transcript_data):
                text = line.get('text', '') if isinstance(line, dict) else getattr(line, 'text', '')
                start = line.get('start', 0) if isinstance(line, dict) else getattr(line, 'start', 0)
                duration = line.get('duration', 0) if isinstance(line, dict) else getattr(line, 'duration', 0)
                try:
                    translated = translator(text)[0]['translation_text']
                    print(f"[Translation] Line {i+1}/{len(transcript_data)}: {text} -> {translated}")
                except Exception as e:
                    translated = text  # fallback
                    print(f"[Translation] Line {i+1}/{len(transcript_data)}: {text} -> [FAILED] {str(e)}")
                translations.append({
                    "original": text,
                    "translated": translated,
                    "start": start,
                    "duration": duration
                })
            
            # Save the translation data
            save_translation(translations, video_id, "Korean", "eng_Latn")
            
            return ko_file, en_file
        except Exception as e:
            print(f"Error during translation: {str(e)}")
            print("Full traceback:")
            print(traceback.format_exc())
            # If translation fails, still return the Korean file
            return ko_file, None

    except Exception as e:
        print(f"Error in save_transcript: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())
        return None, None

executor = ThreadPoolExecutor(max_workers=1)  # Single worker for translations

async def process_remaining_translations(transcript_data, video_id, source_language, target_lang, start_index=20):
    """Process remaining translations in the background"""
    try:
        print(f"Starting background translation from line {start_index}")
        translations = []
        formatted_transcript = ""
        current_paragraph = []
        
        for line in transcript_data[start_index:]:
            try:
                # Access transcript data correctly
                if isinstance(line, dict):
                    text = line.get('text', '')
                    start = line.get('start', 0)
                    duration = line.get('duration', 0)
                else:
                    text = getattr(line, 'text', '')
                    start = getattr(line, 'start', 0)
                    duration = getattr(line, 'duration', 0)
                
                # Translate the text
                translated = translator(text)[0]['translation_text']
                
                # Add to translations list
                translations.append({
                    "original": text,
                    "translated": translated,
                    "start": start,
                    "duration": duration
                })
                
                # Build formatted transcript
                current_paragraph.append(translated)
                if any("[음악]" in text for text in current_paragraph) or (len(current_paragraph) > 1 and isinstance(line, dict) and 'start' in line and isinstance(transcript_data[transcript_data.index(line)-1], dict) and 'start' in transcript_data[transcript_data.index(line)-1] and line['start'] - transcript_data[transcript_data.index(line)-1]['start'] > 2):
                    formatted_transcript += " ".join(current_paragraph) + "\n\n"
                    current_paragraph = []
                    
            except Exception as e:
                print(f"Error translating line in background: {str(e)}")
                continue

        # Add any remaining text
        if current_paragraph:
            formatted_transcript += " ".join(current_paragraph)

        # Load existing translations
        existing_translation = load_existing_translation(video_id, target_lang)
        if existing_translation:
            # Combine with existing translations
            all_translations = existing_translation["translations"][:start_index] + translations
        else:
            all_translations = translations

        # Save the full translation
        save_translation(all_translations, video_id, source_language, target_lang)
        
        # Update the English transcript file
        try:
            video = YouTube(f"https://www.youtube.com/watch?v={video_id}")
            video_title = video.title
        except Exception as e:
            print(f"Error getting video title: {str(e)}")
            video_title = f"video_{video_id}"
        
        sanitized_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        en_file = os.path.join("transcripts", f"{sanitized_title}_{video_id}_en.txt")
        
        # Read existing content
        with open(en_file, "r", encoding="utf-8") as f:
            existing_content = f.read()
        
        # Append new content
        with open(en_file, "w", encoding="utf-8") as f:
            f.write(existing_content)
            f.write(formatted_transcript)
            
        print(f"Completed background translation of {len(translations)} lines")
        
    except Exception as e:
        print(f"Error in background translation: {str(e)}")
        print(traceback.format_exc())

@app.post("/translate")
async def translate_video(video_request: VideoRequest):
    try:
        print(f"📝 Starting translation request for URL: {video_request.url}")
        video_url = video_request.url
        video_id = get_video_id(video_url)
        print(f"📌 Extracted video ID: {video_id}")
        
        target_lang = video_request.target_lang or "eng_Latn"
        print(f" Target language: {target_lang}")

        # First check for existing translation
        existing_translation = load_existing_translation(video_id, target_lang)
        if existing_translation:
            print(f"Using cached translation for video {video_id}")
            # Always try to generate memory cards if not present
            try:
                if not memory_card_exists_for_video(video_id):
                    full_transcript = " ".join([item["original"] for item in existing_translation["translations"]])
                    video_title = existing_translation.get("video_title", f"video_{video_id}")
                    translation_dict = {item["original"]: item["translated"] for item in existing_translation["translations"]}
                    save_all_phrases_to_json(
                        video_id=video_id,
                        video_title=video_title,
                        transcript_text=full_transcript,
                        translations=translation_dict
                    )
                    print(f"✅ Memory cards generated and saved for video {video_id} (from cache)")
            except Exception as e:
                print(f"❌ Error generating memory cards from cache: {str(e)}")

            # Return only first 20 lines for the translation view
            return {
                "status": "success",
                "message": "Loaded existing translation from cache.",
                "subtitle_count": existing_translation["subtitle_count"],
                "translation_method": "📂 Cached",
                "source_language": existing_translation["source_language"],
                "result": existing_translation["translations"][:20]  # Only return first 20 lines
            }

        # If no translation found, proceed with new translation
        try:
            # Get transcript using YouTube Transcript API
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            print("Available transcripts:", transcript_list)
            
            # Try Korean first, then English, then any other language
            try:
                transcript = transcript_list.find_transcript(['ko'])
                source_language = "Korean"
                print(f"Found {source_language} transcript")
            except:
                try:
                    transcript = transcript_list.find_transcript(['en'])
                    source_language = "English"
                    print(f"Found {source_language} transcript")
                except:
                    # Get first available transcript
                    transcript = transcript_list.find_transcript()
                    source_language = transcript.language
                    print(f"Found {source_language} transcript")
            
            try:
                transcript_data = transcript.fetch()
                print(f"Got {len(transcript_data)} lines of transcript")
            except Exception as e:
                print(f"Error fetching transcript: {str(e)}")
                # Try alternative method using pytube
                caption, language = get_captions(video_id)
                if caption:
                    print(f"Using pytube captions in {language}")
                    caption_text = caption.generate_srt_captions()
                    # Convert SRT to list of dicts
                    transcript_data = []
                    current_text = []
                    for line in caption_text.split('\n'):
                        if '-->' in line:
                            if current_text:
                                transcript_data.append({
                                    'text': ' '.join(current_text),
                                    'start': parse_srt_time(line.split('-->')[0].strip()),
                                    'duration': parse_srt_time(line.split('-->')[1].strip()) - parse_srt_time(line.split('-->')[0].strip())
                                })
                                current_text = []
                        elif line.strip() and not line.isdigit():
                            current_text.append(line.strip())
                else:
                    raise HTTPException(status_code=404, detail="No subtitles or captions found for this video.")

            # Process first 20 lines for immediate response
            translations = []
            formatted_transcript = ""
            current_paragraph = []
            
            for line in transcript_data[:20]:
                if isinstance(line, dict):
                    text = line.get('text', '')
                    start = line.get('start', 0)
                    duration = line.get('duration', 0)
                else:
                    text = getattr(line, 'text', '')
                    start = getattr(line, 'start', 0)
                    duration = getattr(line, 'duration', 0)
                translated = translator(text)[0]['translation_text']
                translations.append({
                    "original": text,
                    "translated": translated,
                    "start": start,
                    "duration": duration
                })
                current_paragraph.append(translated)
                if any("[음악]" in text for text in current_paragraph) or (len(current_paragraph) > 1 and isinstance(line, dict) and 'start' in line and isinstance(transcript_data[transcript_data.index(line)-1], dict) and 'start' in transcript_data[transcript_data.index(line)-1] and line['start'] - transcript_data[transcript_data.index(line)-1]['start'] > 2):
                    formatted_transcript += " ".join(current_paragraph) + "\n\n"
                    current_paragraph = []

            # Add any remaining text
            if current_paragraph:
                formatted_transcript += " ".join(current_paragraph)

            # Save initial translation
            save_translation(translations, video_id, source_language, target_lang)
            
            # Save initial English transcript
            try:
                video = YouTube(f"https://www.youtube.com/watch?v={video_id}")
                video_title = video.title
            except Exception as e:
                print(f"Error getting video title: {str(e)}")
                video_title = f"video_{video_id}"
            
            sanitized_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            en_file = os.path.join("transcripts", f"{sanitized_title}_{video_id}_en.txt")
            
            with open(en_file, "w", encoding="utf-8") as f:
                f.write(f"Video Title: {video_title}\n")
                f.write(f"Video ID: {video_id}\n")
                f.write("="*50 + "\n")
                f.write(formatted_transcript)

            # Generate memory cards
            try:
                video_title = get_video_title(video_id)
                full_transcript = " ".join([line['text'] if isinstance(line, dict) else line.text for line in transcript_data])
                translation_dict = {item["original"]: item["translated"] for item in translations} if translations else {}
                save_all_phrases_to_json(
                    video_id=video_id,
                    video_title=video_title,
                    transcript_text=full_transcript,
                    translations=translation_dict
                )
                print(f"✅ Memory cards generated and saved for video {video_id}")
            except Exception as e:
                print(f"❌ Error generating memory cards: {str(e)}")

            # Start background translation for remaining lines
            asyncio.create_task(process_remaining_translations(transcript_data, video_id, source_language, target_lang))

            return {
                "status": "success",
                "message": f"Translated first 20 lines from {source_language}. Continuing translation in background...",
                "subtitle_count": len(transcript_data),
                "translation_method": "🔄 Local Model",
                "source_language": source_language,
                "result": translations
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error in translate_video: {error_msg}")
            print("📋 Full traceback:")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error in translate_video: {error_msg}")
        print("📋 Full traceback:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def parse_srt_time(time_str: str) -> float:
    """Convert SRT time format (HH:MM:SS,mmm) to seconds"""
    try:
        hours, minutes, seconds = time_str.replace(',', '.').split(':')
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    except:
        return 0

@app.post("/get-transcript")
async def get_transcript(video_request: VideoRequest):
    try:
        print(f"Received transcript request for URL: {video_request.url}")
        video_url = video_request.url
        video_id = get_video_id(video_url)
        print(f"Extracted video ID: {video_id}")
        
        try:
            # Try to get captions first
            caption, language = get_captions(video_id)
            
            if caption:
                print(f"Found {language} captions")
                # Get the caption text
                caption_text = caption.generate_srt_captions()
                
                # Convert SRT format to readable text
                formatted_transcript = ""
                current_paragraph = []
                
                for line in caption_text.split('\n'):
                    line = line.strip()
                    if line and not line.isdigit() and not '-->' in line:
                        current_paragraph.append(line)
                        if "[음악]" in line or len(current_paragraph) > 3:
                            formatted_transcript += " ".join(current_paragraph) + "\n\n"
                            current_paragraph = []
                
                # Add any remaining text
                if current_paragraph:
                    formatted_transcript += " ".join(current_paragraph)
                
                return {
                    "status": "success",
                    "message": f"Retrieved {language} captions",
                    "language": language,
                    "transcript": formatted_transcript
                }
            
            # If no captions found, try transcript API as fallback
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            print("No captions found, trying transcript API...")
            
            # Try Korean first, then English, then any other language
            try:
                transcript = transcript_list.find_transcript(['ko'])
                language = "Korean"
            except:
                try:
                    transcript = transcript_list.find_transcript(['en'])
                    language = "English"
                except:
                    # Get the first available transcript
                    transcript = transcript_list.find_transcript()
                    language = transcript.language
            
            transcript_data = transcript.fetch()
            print(f"Found {language} transcript with {len(transcript_data)} lines")
            
            # Format transcript into paragraphs
            formatted_transcript = ""
            current_paragraph = []
            
            for line in transcript_data:
                if isinstance(line, dict) and 'text' in line:
                    current_paragraph.append(line['text'])
                elif isinstance(line, str):
                    current_paragraph.append(line)
                # Start new paragraph if we hit a music marker or long pause
                if any("[음악]" in text for text in current_paragraph) or (len(current_paragraph) > 1 and isinstance(line, dict) and 'start' in line and isinstance(transcript_data[transcript_data.index(line)-1], dict) and 'start' in transcript_data[transcript_data.index(line)-1] and line['start'] - transcript_data[transcript_data.index(line)-1]['start'] > 2):
                    formatted_transcript += " ".join(current_paragraph) + "\n\n"
                    current_paragraph = []
            
            # Add any remaining text
            if current_paragraph:
                formatted_transcript += " ".join(current_paragraph)
            
            return {
                "status": "success",
                "message": f"Retrieved {language} transcript",
                "language": language,
                "subtitle_count": len(transcript_data),
                "transcript": formatted_transcript
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error getting transcript: {error_msg}")
            if "Video unavailable" in error_msg:
                raise HTTPException(status_code=404, detail="Video is unavailable. It might be private or restricted.")
            else:
                raise HTTPException(status_code=404, detail="No subtitles or captions found for this video.")
                
    except Exception as e:
        print(f"Error in get_transcript: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/read-transcript")
async def read_transcript(video_request: VideoRequest):
    try:
        video_id = get_video_id(video_request.url)
        print(f"Reading transcript for video ID: {video_id}")
        
        # First, check for existing transcript files
        pattern = os.path.join("transcripts", f"*{video_id}*.txt")
        files = sorted(glob(pattern), reverse=True)
        
        if files:
            print(f"Found existing transcript files")
            try:
                # Get both Korean and English transcripts
                ko_file = next((f for f in files if f.endswith("_ko.txt")), None)
                en_file = next((f for f in files if f.endswith("_en.txt")), None)
                
                if ko_file and en_file:
                    with open(ko_file, "r", encoding="utf-8") as f:
                        ko_content = f.read()
                    with open(en_file, "r", encoding="utf-8") as f:
                        en_content = f.read()
                    
                    # Extract only the paragraphs (after the separator)
                    ko_paragraph = ko_content.split("="*50, 1)[-1].strip() if "="*50 in ko_content else ko_content
                    en_paragraph = en_content.split("="*50, 1)[-1].strip() if "="*50 in en_content else en_content

                    # Always try to generate memory cards if not present
                    try:
                        if not memory_card_exists_for_video(video_id):
                            video_title = f"video_{video_id}"
                            transcript_text = ko_paragraph
                            save_all_phrases_to_json(
                                video_id=video_id,
                                video_title=video_title,
                                transcript_text=transcript_text,
                                translations={}
                            )
                            print(f"✅ Memory cards generated and saved for video {video_id} (from cached transcript)")
                    except Exception as e:
                        print(f"❌ Error generating memory cards from cached transcript: {str(e)}")

                    return {
                        "status": "success",
                        "message": "Loaded saved transcripts.",
                        "korean_transcript": ko_paragraph,
                        "english_transcript": en_paragraph
                    }
            except Exception as e:
                print(f"Error reading existing transcript files: {str(e)}")
        
        # If not found or reading failed, try to fetch and save transcript
        print("Fetching new transcript...")
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try Korean first, then English, then any other language
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
            except Exception as e:
                print(f"Error fetching transcript: {str(e)}")
                # Try alternative method using pytube
                caption, caption_language = get_captions(video_id)
                if caption:
                    print(f"Using pytube captions in {caption_language}")
                    caption_text = caption.generate_srt_captions()
                    # Convert SRT to list of dicts
                    transcript_data = []
                    current_text = []
                    for line in caption_text.split('\n'):
                        if '-->' in line:
                            if current_text:
                                transcript_data.append({
                                    'text': ' '.join(current_text),
                                    'start': parse_srt_time(line.split('-->')[0].strip()),
                                    'duration': parse_srt_time(line.split('-->')[1].strip()) - parse_srt_time(line.split('-->')[0].strip())
                                })
                                current_text = []
                        elif line.strip() and not line.isdigit():
                            current_text.append(line.strip())
                else:
                    raise HTTPException(status_code=404, detail="No subtitles or captions found for this video.")
            
            # Format transcript into paragraphs
            formatted_transcript = ""
            current_paragraph = []
            
            for line in transcript_data:
                if isinstance(line, dict):
                    text = line.get('text', '')
                else:
                    text = line.text if hasattr(line, 'text') else str(line)
                
                if text:
                    current_paragraph.append(text)
                    # Start new paragraph if we hit a music marker or long pause
                    if any("[음악]" in text for text in current_paragraph) or (len(current_paragraph) > 1 and isinstance(line, dict) and 'start' in line and isinstance(transcript_data[transcript_data.index(line)-1], dict) and 'start' in transcript_data[transcript_data.index(line)-1] and line['start'] - transcript_data[transcript_data.index(line)-1]['start'] > 2):
                        formatted_transcript += " ".join(current_paragraph) + "\n\n"
                        current_paragraph = []
            
            # Add any remaining text
            if current_paragraph:
                formatted_transcript += " ".join(current_paragraph)
            
            # Save both original and translated transcripts
            ko_file, en_file = save_transcript(transcript_data, video_id)
            
            if ko_file and en_file:
                with open(ko_file, "r", encoding="utf-8") as f:
                    ko_content = f.read()
                with open(en_file, "r", encoding="utf-8") as f:
                    en_content = f.read()
                
                ko_paragraph = ko_content.split("="*50, 1)[-1].strip() if "="*50 in ko_content else ko_content
                en_paragraph = en_content.split("="*50, 1)[-1].strip() if "="*50 in en_content else en_content
                
                # Generate memory cards
                try:
                    video_title = get_video_title(video_id)
                    full_transcript = " ".join([line['text'] if isinstance(line, dict) else line.text for line in transcript_data])
                    save_all_phrases_to_json(
                        video_id=video_id,
                        video_title=video_title,
                        transcript_text=full_transcript,
                        translations={}
                    )
                    print(f"✅ Memory cards generated and saved for video {video_id}")
                except Exception as e:
                    print(f"❌ Error generating memory cards: {str(e)}")

                return {
                    "status": "success",
                    "message": f"Fetched and saved new transcripts.",
                    "korean_transcript": ko_paragraph,
                    "english_transcript": en_paragraph
                }
                
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

def get_captions(video_id: str):
    """Get captions using pytube"""
    try:
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        caption = yt.captions.get_by_language_code('ko') or yt.captions.get_by_language_code('en')
        return caption, caption.language_code if caption else None
    except Exception as e:
        print(f"Error getting captions: {str(e)}")
        return None, None

@app.get("/memory-cards/{video_id}")
async def get_memory_cards(video_id: str):
    """Get memory cards for a specific video"""
    try:
        latest_file = get_latest_memory_cards_file(video_id)
        if not latest_file:
            return {"status": "success", "message": "No memory cards found", "memory_cards": []}
        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "status": "success",
            "message": f"Retrieved {len(data.get('phrases', []))} memory cards",
            "memory_cards": data.get('phrases', [])
        }
    except Exception as e:
        print(f"Error getting memory cards: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
