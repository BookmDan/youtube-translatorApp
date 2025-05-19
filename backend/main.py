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
from datetime import timedelta
import torch
import re

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
        "https://your-frontend-domain.vercel.app"  # Your Vercel frontend
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

        # Save to a new numbered output directory
        output_dir = get_next_output_dir()
        output_file = os.path.join(output_dir, "translated_subtitles.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(translated_lines, f, ensure_ascii=False, indent=2)

        txt_file = os.path.join(output_dir, "translated_subtitles.txt")
        with open(txt_file, "w", encoding="utf-8") as f:
            for line in translated_lines:
                f.write(f"[{format_timestamp(line['start'])}]\n")
                f.write(f"Korean: {line['original']}\n")
                f.write(f"English: {line['translation']}\n\n")

        return {
            "status": "success",
            "message": f"Translated first 20 lines of {len(transcript)} total lines",
            "subtitle_count": len(transcript),
            "output_file": output_file,
            "result": translated_lines  # Return all 20 translations for preview
        }
    except Exception as e:
        print(f"Error in translate_video: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
