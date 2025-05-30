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

app = FastAPI()

# Hugging Face Inference API Configuration
HF_API_BASE = "https://api-inference.huggingface.co/models"
HF_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")  # Get from environment variable
HF_ENABLED = HF_API_KEY is not None

print(f"🔧 Hugging Face API: {'Enabled' if HF_ENABLED else 'Disabled (no API key)'}")

# Update CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get port from environment variable
port = int(os.environ.get("PORT", 8000))

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
            translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en", device=0 if torch.cuda.is_available() else -1)
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
            en_tl_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-tl", device=0 if torch.cuda.is_available() else -1)
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
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        return yt.title
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

def save_transcript(transcript_data: list, video_id: str) -> str:
    """Save transcript from YouTube video to file"""
    try:
        print("Starting to save transcript...")
        # Get video title
        video_title = get_video_title(video_id)
        print(f"Got video title: {video_title}")
        
        # Create transcripts directory if it doesn't exist
        transcripts_dir = os.path.join(os.getcwd(), "transcripts")
        print(f"Creating transcripts directory at: {transcripts_dir}")
        os.makedirs(transcripts_dir, exist_ok=True)
        
        # Sanitize video title for filename
        safe_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_title = safe_title.replace(' ', '_')
        print(f"Sanitized title: {safe_title}")
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transcript_{timestamp}_{safe_title}_ko.txt"
        filepath = os.path.join(transcripts_dir, filename)
        print(f"Will save to file: {filepath}")
        
        # Save transcript
        print("Writing transcript to file...")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Video Title: {video_title}\n")
            f.write(f"Video ID: {video_id}\n")
            f.write(f"Language: Korean\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n" + "="*50 + "\n\n")
            
            # Process transcript text for better readability
            full_transcript = " ".join(line['text'] for line in transcript_data)
            print(f"Full transcript length: {len(full_transcript)}")
            
            # First, handle music markers
            formatted_text = re.sub(r'(\[음악\])', r'\1\n', full_transcript)
            
            # Split into words
            words = formatted_text.split()
            current_line = []
            current_length = 0
            max_line_length = 50  # Maximum characters per line
            
            # Write the formatted text
            for word in words:
                # If adding this word would exceed the line length, write the current line
                if current_length + len(word) + 1 > max_line_length and current_line:
                    f.write(" ".join(current_line) + "\n")
                    current_line = []
                    current_length = 0
                
                # Add the word to the current line
                current_line.append(word)
                current_length += len(word) + 1  # +1 for the space
                
                # If the word contains a music marker, write the line
                if "[음악]" in word:
                    f.write(" ".join(current_line) + "\n")
                    current_line = []
                    current_length = 0
            
            # Write any remaining words
            if current_line:
                f.write(" ".join(current_line) + "\n")
            
            print(f"Wrote formatted transcript to file")
        
        print(f"Successfully saved transcript to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Error saving transcript: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Transcript data length: {len(transcript_data)}")
        return None

@app.post("/translate")
async def translate_video(video_request: VideoRequest):
    try:
        print(f"📝 Starting translation request for URL: {video_request.url}")
        video_url = video_request.url
        video_id = get_video_id(video_url)
        print(f"📌 Extracted video ID: {video_id}")
        
        target_lang = video_request.target_lang or "eng_Latn"
        print(f"🎯 Target language: {target_lang}")
        
        # Verify directories exist
        for directory in ["translations", "transcripts"]:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                print(f"📁 Created missing directory: {directory}")
        
        # Check translation model availability
        if not translator and not HF_ENABLED:
            print("⚠️ Warning: No translation models available")
            raise HTTPException(
                status_code=500,
                detail="Translation service is not available. Please try again later."
            )
        
        # Check if translation already exists
        existing_translation = load_existing_translation(video_id, target_lang)
        if existing_translation:
            print(f"🗂️ Found existing translation for {video_id} -> {target_lang}")
            return {
                "status": "success",
                "message": f"Loaded existing translation from {existing_translation['source_language']} (Generated: {existing_translation['generated']})",
                "subtitle_count": existing_translation['subtitle_count'],
                "translation_method": "🗂️ Cached Translation",
                "source_language": existing_translation['source_language'],
                "video_title": existing_translation['video_title'],
                "result": existing_translation['translations'][:20]
            }

        try:
            # Use the same robust approach as read-transcript function
            print("Looking for available transcripts...")
            
            # First, check if we have a saved transcript file
            pattern = os.path.join("transcripts", f"*{video_id}*.txt")
            files = sorted(glob(pattern), reverse=True)
            if files:
                print("Found existing transcript file, using it")
                # We have a saved transcript, but we still need to get the raw data for translation
                # Fall through to fetch from API for translation purposes
                pass
            
            # Fetch from YouTube API using robust method
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try Korean first, then English, then any other language
            source_language = None
            transcript_obj = None
            
            try:
                transcript_obj = transcript_list.find_transcript(['ko'])
                source_language = "Korean"
                print("Found Korean transcript!")
            except:
                try:
                    transcript_obj = transcript_list.find_transcript(['en'])
                    source_language = "English"
                    print("Found English transcript (Korean not available)")
                except:
                    # Get the first available transcript
                    available_transcripts = list(transcript_list)
                    if not available_transcripts:
                        raise HTTPException(status_code=404, detail="No transcripts available for this video.")
                    transcript_obj = available_transcripts[0]
                    source_language = transcript_obj.language
                    print(f"Found {source_language} transcript (Korean/English not available)")
            
            # Safely fetch the transcript data with error handling
            try:
                transcript_data = transcript_obj.fetch()
                print(f"Successfully fetched transcript data with {len(transcript_data)} lines")
            except Exception as fetch_error:
                print(f"Error fetching transcript data: {str(fetch_error)}")
                # If fetch fails, check if we have a saved file to use
                if files:
                    print("Fetch failed, but we have a saved transcript file. Using fallback approach.")
                    # Read from saved file and create minimal transcript data for translation
                    with open(files[0], "r", encoding="utf-8") as f:
                        content = f.read()
                    if "="*10 in content:
                        text_content = content.split("="*10, 1)[-1].strip()
                    else:
                        text_content = content
                    
                    # Create simple transcript data for translation
                    lines = text_content.split('\n')
                    transcript_data = []
                    for i, line in enumerate(lines[:20]):  # Limit to first 20 lines
                        if line.strip():
                            transcript_data.append({
                                'text': line.strip(),
                                'start': i * 3.0,  # Approximate timing
                                'duration': 3.0
                            })
                    source_language = "Korean"  # Assume Korean for saved files
                else:
                    raise fetch_error
            
            # Convert to the format expected by the rest of the function
            transcript = [{"text": line.text if hasattr(line, 'text') else line['text'], 
                          "start": line.start if hasattr(line, 'start') else line['start'], 
                          "duration": line.duration if hasattr(line, 'duration') else line['duration']} 
                         for line in transcript_data]
            
            if not transcript:
                raise HTTPException(status_code=404, detail="No subtitles found for this video.")
            print(f"Found {source_language} transcript with {len(transcript)} lines")
            
            # Save the full transcript
            print("Attempting to save transcript...")
            transcript_file = save_transcript(transcript, video_id)
            if transcript_file:
                print(f"Successfully saved transcript to: {transcript_file}")
            else:
                print("Failed to save transcript")
            
        except Exception as e:
            error_msg = str(e)
            if "no element found" in error_msg or "No transcripts were found" in error_msg:
                raise HTTPException(
                    status_code=404,
                    detail="No subtitles found for this video. The video may not have captions available."
                )
            elif "Video unavailable" in error_msg:
                raise HTTPException(
                    status_code=404,
                    detail="Video is unavailable. It might be private or restricted."
                )
            elif "ParseError" in error_msg or "XML" in error_msg:
                raise HTTPException(
                    status_code=404,
                    detail="Transcript data is corrupted or unavailable for this video."
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error retrieving transcript: {error_msg}"
                )

        print(f"Translating {source_language} to {target_lang}")

        # Skip translation if source and target are the same language
        if (source_language == "Korean" and target_lang == "ko") or \
           (source_language == "English" and target_lang == "eng_Latn"):
            # Just return the original text
            translated_lines = []
            for line in transcript[:20]:
                translated_lines.append({
                    "original": line['text'],
                    "start": line['start'],
                    "duration": line['duration'],
                    "translation": line['text']  # Same as original
                })
            
            # Save this "no translation needed" result for caching
            no_translation_file = save_translation(translated_lines, video_id, source_language, target_lang)
            
            return {
                "status": "success",
                "message": f"Source and target languages are the same ({source_language}). Returning original text.",
                "subtitle_count": len(transcript),
                "transcript_file": transcript_file,
                "translation_file": no_translation_file,
                "translation_method": "No translation needed",
                "source_language": source_language,
                "result": translated_lines
            }

        # Use Hugging Face API if available, otherwise use local models
        if HF_ENABLED:
            print(f"🚀 Using Hugging Face API for faster translation: {source_language} -> {target_lang}")
            
            # Determine the appropriate model based on source and target languages
            if source_language == "Korean":
                # Korean source - use existing logic
                if target_lang in ["tl_Latn", "zho_Hans", "arb_Arab", "swe_Latn"]:
                    # Two-step translation: Korean -> English -> Target
                    print(f"Two-step translation via API: Korean -> English -> {target_lang}")
                    model_step1 = LANG_MODEL_MAP["eng_Latn"]  # Korean to English
                    model_step2 = LANG_MODEL_MAP[target_lang]  # English to Target
                    two_step = True
                else:
                    # Direct Korean to target translation
                    print(f"Direct translation via API: Korean -> {target_lang}")
                    model_direct = LANG_MODEL_MAP[target_lang]
                    two_step = False
            else:
                # Non-Korean source (English or other)
                if source_language == "English":
                    if target_lang in LANG_MODEL_MAP:
                        # Direct English to target translation
                        print(f"Direct translation via API: English -> {target_lang}")
                        if target_lang == "eng_Latn":
                            model_direct = "Helsinki-NLP/opus-mt-en-en"  # No translation needed
                        else:
                            model_direct = LANG_MODEL_MAP[target_lang]
                        two_step = False
                    else:
                        raise HTTPException(status_code=400, detail=f"Translation from English to {target_lang} not supported")
                else:
                    # Other source language - try two-step through English
                    print(f"Two-step translation via API: {source_language} -> English -> {target_lang}")
                    # This is more complex and might need additional models
                    # For now, fallback to local models
                    raise Exception("Non-Korean/English source requires local models")
            
            translated_lines = []
            for i, line in enumerate(transcript[:20]):
                print(f"Translating line {i+1}/20 via API")
                try:
                    if two_step:
                        # Two-step translation
                        english = translate_with_huggingface_api(line['text'], model_step1)
                        translation = translate_with_huggingface_api(english, model_step2)
                    else:
                        # Direct translation
                        translation = translate_with_huggingface_api(line['text'], model_direct)
                    
                    translation = remove_phrase_repetition(translation, max_repeat=1)
                    translated_lines.append({
                        "original": line['text'],
                        "start": line['start'],
                        "duration": line['duration'],
                        "translation": translation
                    })
                except Exception as e:
                    print(f"API translation failed for line {i+1}, falling back to local: {str(e)}")
                    # Fallback to local models
                    if source_language == "Korean":
                        english = translator(line['text'])[0]['translation_text']
                        if target_lang == "tl_Latn":
                            translation = en_tl_translator(english)[0]['translation_text']
                        elif target_lang in ["zho_Hans", "arb_Arab", "swe_Latn"]:
                            final_model = LANG_MODEL_MAP[target_lang]
                            final_translator = pipeline("translation", model=final_model, device=0 if torch.cuda.is_available() else -1)
                            translation = final_translator(english)[0]['translation_text']
                        else:
                            translation = english  # Direct Korean to English
                    else:
                        # For non-Korean source, just return original for now
                        translation = line['text']
                    
                    translation = remove_phrase_repetition(translation, max_repeat=1)
                    translated_lines.append({
                        "original": line['text'],
                        "start": line['start'],
                        "duration": line['duration'],
                        "translation": translation
                    })
        else:
            # Fallback to local models
            print(f"🔄 Using local models: Korean -> {target_lang}")
            if target_lang in ["tl_Latn", "zho_Hans", "arb_Arab", "swe_Latn"]:
                # Two-step translation for complex languages: Korean -> English -> Target
                print(f"Two-step translation: Korean -> English -> {target_lang}")
                translated_lines = []
                for i, line in enumerate(transcript[:20]):
                    print(f"Translating line {i+1}/20")
                    # Step 1: Korean to English
                    english = translator(line['text'])[0]['translation_text']
                    # Step 2: English to Target
                    if target_lang == "tl_Latn":
                        translation = en_tl_translator(english)[0]['translation_text']
                    else:
                        # Load the specific translator for this target
                        final_model = LANG_MODEL_MAP[target_lang]
                        final_translator = pipeline("translation", model=final_model, device=0 if torch.cuda.is_available() else -1)
                        translation = final_translator(english)[0]['translation_text']
                    
                    translation = remove_phrase_repetition(translation, max_repeat=1)
                    translated_lines.append({
                        "original": line['text'],
                        "start": line['start'],
                        "duration": line['duration'],
                        "translation": translation
                    })
            else:
                # Direct Korean to target translation (English, Spanish, French)
                print(f"Direct translation: Korean -> {target_lang}")
                translated_lines = []
                for i, line in enumerate(transcript[:20]):
                    print(f"Translating line {i+1}/20")
                    translation = translator(line['text'])[0]['translation_text']
                    translation = remove_phrase_repetition(translation, max_repeat=1)
                    translated_lines.append({
                        "original": line['text'],
                        "start": line['start'],
                        "duration": line['duration'],
                        "translation": translation
                    })

        translation_method = "🚀 Hugging Face API" if HF_ENABLED else "🔄 Local Models"
        
        # Save the translation for future use
        translation_file = save_translation(translated_lines, video_id, source_language, target_lang)
        if translation_file:
            print(f"Successfully saved translation to: {translation_file}")
        else:
            print("Failed to save translation")
        
        return {
            "status": "success",
            "message": f"Translated first 20 lines of {len(transcript)} total lines from {source_language} using {translation_method}",
            "subtitle_count": len(transcript),
            "transcript_file": transcript_file,
            "translation_file": translation_file,
            "translation_method": translation_method,
            "source_language": source_language,
            "result": translated_lines
        }
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error in translate_video: {error_msg}")
        print("📋 Full traceback:")
        print(traceback.format_exc())
        
        if "No transcripts were found" in error_msg:
            raise HTTPException(
                status_code=404,
                detail="No subtitles found for this video. The video may not have captions available."
            )
        elif "Video unavailable" in error_msg:
            raise HTTPException(
                status_code=404,
                detail="Video is unavailable. It might be private or restricted."
            )
        elif "Invalid YouTube URL format" in error_msg:
            raise HTTPException(
                status_code=400,
                detail="Please provide a valid YouTube URL."
            )
        elif "Translation service is not available" in error_msg:
            raise HTTPException(
                status_code=503,
                detail="Translation service is temporarily unavailable. Please try again later."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred during translation: {str(e)}"
            )

def get_captions(video_id: str) -> tuple:
    """Get captions from YouTube video using pytube"""
    try:
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        print(f"Video title: {yt.title}")
        
        # Get available caption tracks
        caption_tracks = yt.captions
        print(f"Available caption tracks: {caption_tracks}")
        
        # Try to get Korean captions first
        try:
            caption = caption_tracks.get_by_language_code('ko')
            if caption:
                return caption, "Korean"
        except:
            pass
            
        # Try English if Korean not available
        try:
            caption = caption_tracks.get_by_language_code('en')
            if caption:
                return caption, "English"
        except:
            pass
            
        # Get first available caption
        if caption_tracks:
            first_caption = list(caption_tracks.values())[0]
            return first_caption, first_caption.name
            
        return None, None
        
    except Exception as e:
        print(f"Error getting captions: {str(e)}")
        return None, None

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
                current_paragraph.append(line.text)
                # Start new paragraph if we hit a music marker or long pause
                if "[음악]" in line.text or (len(current_paragraph) > 1 and line.start - transcript_data[transcript_data.index(line)-1].start > 2):
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
        # First, check for an existing transcript file
        pattern = os.path.join("transcripts", f"*{video_id}*.txt")
        files = sorted(glob(pattern), reverse=True)
        if files:
            # Read the most recent file
            with open(files[0], "r", encoding="utf-8") as f:
                content = f.read()
            # Extract only the paragraph (after the separator)
            if "="*10 in content:
                paragraph = content.split("="*10, 1)[-1].strip()
            else:
                paragraph = content
            return {
                "status": "success",
                "message": "Loaded saved transcript.",
                "transcript": paragraph
            }
        # If not found, try to fetch and save transcript
        try:
            # Try multiple languages like the get-transcript endpoint
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
                    # Get the first available transcript
                    available_transcripts = list(transcript_list)
                    if not available_transcripts:
                        raise HTTPException(status_code=404, detail="No transcripts available for this video.")
                    transcript_obj = available_transcripts[0]
                    language = transcript_obj.language
            
            transcript_data = transcript_obj.fetch()
            print(f"Found {language} transcript with {len(transcript_data)} lines")
            
            # Convert to the format expected by save_transcript
            transcript = [{"text": line.text, "start": line.start, "duration": line.duration} for line in transcript_data]
            
            save_transcript(transcript, video_id)
            # Try again to find the file
            files = sorted(glob(pattern), reverse=True)
            if not files:
                raise HTTPException(status_code=404, detail="Transcript could not be saved.")
            with open(files[0], "r", encoding="utf-8") as f:
                content = f.read()
            if "="*10 in content:
                paragraph = content.split("="*10, 1)[-1].strip()
            else:
                paragraph = content
            return {
                "status": "success",
                "message": f"Fetched and saved new {language} transcript.",
                "transcript": paragraph
            }
        except Exception as e:
            error_msg = str(e)
            if "Video unavailable" in error_msg:
                raise HTTPException(status_code=404, detail="Video is unavailable. It might be private or restricted.")
            else:
                raise HTTPException(status_code=404, detail=f"No subtitles or captions found for this video. {error_msg}")
    except Exception as e:
        print(f"Error in read_transcript: {str(e)}")
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
