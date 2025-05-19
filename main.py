from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import os
import json
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from datetime import timedelta

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("output", exist_ok=True)
app.mount("/output", StaticFiles(directory="output"), name="output")

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en")

class VideoRequest(BaseModel):
    url: str
    start_time: Optional[int] = 0
    end_time: Optional[int] = None

def get_video_id(url: str) -> str:
    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    return url.split("v=")[1].split("&")[0]

def get_next_output_dir(base_dir="output"):
    existing = [int(d) for d in os.listdir(base_dir) if d.isdigit()]
    next_num = max(existing, default=0) + 1
    new_dir = os.path.join(base_dir, str(next_num))
    os.makedirs(new_dir, exist_ok=True)
    return new_dir

@app.post("/translate")
async def translate_video(video_request: VideoRequest):
    try:
        video_url = video_request.url
        video_id = get_video_id(video_url)
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko'])
        if not transcript:
            raise HTTPException(status_code=404, detail="No transcript found.")

        # Translate all lines
        translated_lines = []
        for line in transcript:
            translation = translator(line['text'])[0]['translation_text']
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
            "message": "All lines translated",
            "subtitle_count": len(translated_lines),
            "output_file": output_file,
            "result": translated_lines  # Optionally return the translations in the response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
