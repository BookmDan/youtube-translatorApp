from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import os
import json
import traceback
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from datetime import timedelta, datetime
import torch
import re
from pytube import YouTube
from glob import glob

app = FastAPI()

# Create cache directory if it doesn't exist
os.makedirs("model_cache", exist_ok=True)

# Set environment variable for HuggingFace cache
os.environ["TRANSFORMERS_CACHE"] = "model_cache"

# Load the translator once at startup
print("Loading translation model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = model.to(device)

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en", device=0 if torch.cuda.is_available() else -1)

en_tl_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-tl", device=0 if torch.cuda.is_available() else -1)

LANG_MODEL_MAP = {
    "eng_Latn": "Helsinki-NLP/opus-mt-ko-en",
    "spa_Latn": "Helsinki-NLP/opus-mt-ko-es",
    "fra_Latn": "Helsinki-NLP/opus-mt-ko-fr",
    "zho_Hans": "Helsinki-NLP/opus-mt-en-zh",   # English to Chinese (Simplified)
    "arb_Arab": "Helsinki-NLP/opus-mt-en-ar",   # English to Arabic
    "tl_Latn": "Helsinki-NLP/opus-mt-en-tl",    # English to Tagalog
    "swe_Latn": "Helsinki-NLP/opus-mt-en-sv",   # English to Swedish
}

# Update CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "https://youtube-translator-app-s9q9.vercel.app"  # Vercel frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get port from environment variable
port = int(os.environ.get("PORT", 8000))

os.makedirs("output", exist_ok=True)
app.mount("/output", StaticFiles(directory="output"), name="output")

class VideoRequest(BaseModel):
    url: str
    start_time: Optional[int] = 0
    end_time: Optional[int] = None
    target_lang: Optional[str] = "eng_Latn"  # Default to English

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
        print(f"Received request for URL: {video_request.url}")
        video_url = video_request.url
        video_id = get_video_id(video_url)
        print(f"Extracted video ID: {video_id}")
        
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko'])
            if not transcript:
                raise HTTPException(status_code=404, detail="No Korean subtitles found for this video.")
            print(f"Found transcript with {len(transcript)} lines")
            
            # Save the full transcript
            print("Attempting to save transcript...")
            transcript_file = save_transcript(transcript, video_id)
            if transcript_file:
                print(f"Successfully saved transcript to: {transcript_file}")
            else:
                print("Failed to save transcript")
            
        except Exception as e:
            error_msg = str(e)
            if "no element found" in error_msg:
                raise HTTPException(status_code=404, detail="No Korean subtitles found for this video. Please ensure the video has Korean subtitles available.")
            elif "Video unavailable" in error_msg:
                raise HTTPException(status_code=404, detail="Video is unavailable. It might be private or restricted.")
            else:
                raise HTTPException(status_code=500, detail=f"Error retrieving transcript: {error_msg}")

        target_lang = video_request.target_lang or "eng_Latn"

        if target_lang in ["tl_Latn", "zho_Hans", "arb_Arab", "swe_Latn"]:
            # Step 1: Korean to English
            intermediate_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en", device=0 if torch.cuda.is_available() else -1)
            # Step 2: English to Target
            final_model = LANG_MODEL_MAP[target_lang]
            final_translator = pipeline("translation", model=final_model, device=0 if torch.cuda.is_available() else -1)
        else:
            model_name = LANG_MODEL_MAP.get(target_lang, "Helsinki-NLP/opus-mt-ko-en")
            intermediate_translator = pipeline("translation", model=model_name, device=0 if torch.cuda.is_available() else -1)
            final_translator = None

        # Translate only first 20 lines
        preview_lines = transcript[:20]
        translated_lines = []
        for i, line in enumerate(preview_lines):
            print(f"Translating line {i+1}/20")
            if target_lang in ["tl_Latn", "zho_Hans", "arb_Arab", "swe_Latn"]:
                # Korean → English → Target
                english = intermediate_translator(line['text'])[0]['translation_text']
                translation = final_translator(english)[0]['translation_text']
            else:
                translation = intermediate_translator(line['text'])[0]['translation_text']
            translation = remove_phrase_repetition(translation, max_repeat=1)
            translated_lines.append({
                "original": line['text'],
                "start": line['start'],
                "duration": line['duration'],
                "translation": translation
            })

        return {
            "status": "success",
            "message": f"Translated first 20 lines of {len(transcript)} total lines",
            "subtitle_count": len(transcript),
            "transcript_file": transcript_file,
            "result": translated_lines
        }
    except Exception as e:
        print(f"Error in translate_video: {str(e)}")
        print("Full traceback:")
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
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            if not transcript:
                raise HTTPException(status_code=404, detail="No subtitles found for this video.")
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
                "message": "Fetched and saved new transcript.",
                "transcript": paragraph
            }
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"No subtitles or captions found for this video. {str(e)}")
    except Exception as e:
        print(f"Error in read_transcript: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
