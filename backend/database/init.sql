 -- Create memory_cards table
CREATE TABLE IF NOT EXISTS public.memory_cards (
    id BIGSERIAL PRIMARY KEY,
    video_id TEXT NOT NULL,
    video_title TEXT NOT NULL,
    phrase TEXT NOT NULL,
    translation TEXT NOT NULL,
    frequency INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create transcripts table
CREATE TABLE IF NOT EXISTS public.transcripts (
    id BIGSERIAL PRIMARY KEY,
    video_id TEXT NOT NULL UNIQUE,
    video_title TEXT NOT NULL,
    korean_text TEXT NOT NULL,
    english_text TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create translations table
CREATE TABLE IF NOT EXISTS public.translations (
    id BIGSERIAL PRIMARY KEY,
    video_id TEXT NOT NULL,
    source_language TEXT NOT NULL,
    target_language TEXT NOT NULL,
    translation_data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);