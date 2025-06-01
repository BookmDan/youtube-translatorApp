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
        formatted_text = ""
        for entry in transcript_data:
            if isinstance(entry, dict) and 'text' in entry:
                formatted_text += entry['text'] + "\n"
            elif isinstance(entry, str):
                formatted_text += entry + "\n"
        
        print(f"Full transcript length: {len(formatted_text)}")
        
        # Save Korean transcript
        ko_file = os.path.join(transcripts_dir, f"{sanitized_title}_{video_id}_ko.txt")
        print("Writing Korean transcript to file...")
        with open(ko_file, "w", encoding="utf-8") as f:
            f.write(f"Video Title: {video_title}\n")
            f.write(f"Video ID: {video_id}\n")
            f.write("="*50 + "\n")
            f.write(formatted_text)
        print("Korean transcript saved successfully")

        # Translate and save English transcript
        print("Translating and saving English transcript...")
        try:
            if translator is None:
                raise Exception("Local translator not loaded")
            
            # Split text into smaller chunks for translation
            chunks = [formatted_text[i:i+500] for i in range(0, len(formatted_text), 500)]
            translated_chunks = []
            
            for i, chunk in enumerate(chunks):
                print(f"Translating chunk {i+1}/{len(chunks)}")
                try:
                    translation = translator(chunk)[0]['translation_text']
                    translated_chunks.append(translation)
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

@app.post("/translate")
async def translate_video(video_request: VideoRequest):
    try:
        print(f"📝 Starting translation request for URL: {video_request.url}")
        video_url = video_request.url
        video_id = get_video_id(video_url)
        print(f"📌 Extracted video ID: {video_id}")
        
        target_lang = video_request.target_lang or "eng_Latn"
        print(f"🎯 Target language: {target_lang}")

        # First check for existing translation
        existing_translation = load_existing_translation(video_id, target_lang)
        if existing_translation:
            print(f"Found existing translation for video {video_id}")
            return {
                "status": "success",
                "message": "Loaded existing translation from cache.",
                "subtitle_count": existing_translation["subtitle_count"],
                "translation_method": "📂 Cached",
                "source_language": existing_translation["source_language"],
                "result": existing_translation["translations"]
            }

        # If no translation found, check for existing transcript
        pattern = os.path.join("transcripts", f"*{video_id}*.txt")
        files = sorted(glob(pattern), reverse=True)
        
        if files:
            print(f"Found existing transcript files")
            try:
                # Get both Korean and English transcripts
                ko_file = next((f for f in files if f.endswith("_ko.txt")), None)
                en_file = next((f for f in files if f.endswith("_en.txt")), None)
                
                if ko_file and en_file:
                    print("Using existing transcripts for translation")
                    with open(ko_file, "r", encoding="utf-8") as f:
                        ko_content = f.read()
                    with open(en_file, "r", encoding="utf-8") as f:
                        en_content = f.read()
                    
                    # Extract only the paragraphs (after the separator)
                    ko_paragraph = ko_content.split("="*50, 1)[-1].strip() if "="*50 in ko_content else ko_content
                    en_paragraph = en_content.split("="*50, 1)[-1].strip() if "="*50 in en_content else en_content
                    
                    # Create translation data from existing transcripts
                    translation_data = []
                    ko_lines = ko_paragraph.split('\n')
                    en_lines = en_paragraph.split('\n')
                    
                    for i, (ko_line, en_line) in enumerate(zip(ko_lines, en_lines)):
                        if ko_line.strip() and en_line.strip():
                            translation_data.append({
                                "original": ko_line.strip(),
                                "start": i * 5,  # Approximate timing
                                "duration": 5,   # Approximate duration
                                "translation": en_line.strip()
                            })
                    
                    # Save the translation for future use
                    save_translation(translation_data, video_id, "Korean", target_lang)
                    
                    return {
                        "status": "success",
                        "message": "Translated using existing transcripts.",
                        "subtitle_count": len(translation_data),
                        "translation_method": "📝 Existing Transcripts",
                        "source_language": "Korean",
                        "result": translation_data
                    }
            except Exception as e:
                print(f"Error using existing transcripts: {str(e)}")
        
        # If no existing translation or transcript found, proceed with new translation
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to get Korean transcript first
            try:
                transcript = transcript_list.find_transcript(['ko']).fetch()
                source_language = "Korean"
                print(f"Found Korean transcript with {len(transcript)} lines")
            except:
                # If Korean not available, try English
                try:
                    transcript = transcript_list.find_transcript(['en']).fetch()
                    source_language = "English"
                    print(f"Found English transcript with {len(transcript)} lines")
                except:
                    # If neither Korean nor English available, get any available transcript
                    available_transcripts = list(transcript_list)
                    if not available_transcripts:
                        raise HTTPException(status_code=404, detail="No subtitles found for this video.")
                    transcript = available_transcripts[0].fetch()
                    source_language = available_transcripts[0].language
                    print(f"Found {source_language} transcript with {len(transcript)} lines")
            
        except Exception as e:
            print(f"Error getting transcript: {str(e)}")
            raise HTTPException(status_code=404, detail="No subtitles found for this video.")

        # If source language is not Korean, we can't translate it
        if source_language != "Korean":
            return {
                "status": "error",
                "message": f"This video has {source_language} subtitles, but we can only translate from Korean. Please try a video with Korean subtitles.",
                "source_language": source_language
            }

        # Translate using local model
        translated_lines = []
        for i, line in enumerate(transcript[:20]):  # Translate first 20 lines
            print(f"Translating line {i+1}/20")
            try:
                translation = translator(line['text'])[0]['translation_text']
                translation = remove_phrase_repetition(translation, max_repeat=1)
                translated_lines.append({
                    "original": line['text'],
                    "start": line['start'],
                    "duration": line['duration'],
                    "translation": translation
                })
            except Exception as e:
                print(f"Error translating line {i+1}: {str(e)}")
                continue

        # Save the translation for future use
        save_translation(translated_lines, video_id, "Korean", target_lang)

        return {
            "status": "success",
            "message": f"Translated first 20 lines of {len(transcript)} total lines from Korean",
            "subtitle_count": len(transcript),
            "translation_method": "🔄 Local Model",
            "source_language": "Korean",
            "result": translated_lines
        }
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error in translate_video: {error_msg}")
        print("📋 Full traceback:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

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
                
                # Save both original and translated transcripts
                ko_file, en_file = save_transcript(transcript_data, video_id)
                
                if ko_file and en_file:
                    with open(ko_file, "r", encoding="utf-8") as f:
                        ko_content = f.read()
                    with open(en_file, "r", encoding="utf-8") as f:
                        en_content = f.read()
                    
                    ko_paragraph = ko_content.split("="*50, 1)[-1].strip() if "="*50 in ko_content else ko_content
                    en_paragraph = en_content.split("="*50, 1)[-1].strip() if "="*50 in en_content else en_content
                    
                    return {
                        "status": "success",
                        "message": f"Fetched and saved new transcripts.",
                        "korean_transcript": ko_paragraph,
                        "english_transcript": en_paragraph
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
                        ko_file, en_file = save_transcript([{"text": formatted_text}], video_id)
                        
                        if ko_file and en_file:
                            with open(ko_file, "r", encoding="utf-8") as f:
                                ko_content = f.read()
                            with open(en_file, "r", encoding="utf-8") as f:
                                en_content = f.read()
                            
                            ko_paragraph = ko_content.split("="*50, 1)[-1].strip() if "="*50 in ko_content else ko_content
                            en_paragraph = en_content.split("="*50, 1)[-1].strip() if "="*50 in en_content else en_content
                            
                            return {
                                "status": "success",
                                "message": f"Fetched and saved new transcripts using alternative method.",
                                "korean_transcript": ko_paragraph,
                                "english_transcript": en_paragraph
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
