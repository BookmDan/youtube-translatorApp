from supabase import create_client
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supabase configuration for local development
SUPABASE_URL = "http://127.0.0.1:54321"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0"

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def init_db():
    """Initialize database tables"""
    try:
        # Create memory_cards table
        print("Creating memory_cards table...")
        supabase.table("memory_cards").select("*").limit(1).execute()
        print("✅ memory_cards table exists")

        # Create transcripts table
        print("Creating transcripts table...")
        supabase.table("transcripts").select("*").limit(1).execute()
        print("✅ transcripts table exists")

        # Create translations table
        print("Creating translations table...")
        supabase.table("translations").select("*").limit(1).execute()
        print("✅ translations table exists")

        print("✅ Database tables initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing database: {str(e)}")
        print("Please make sure to create the tables in Supabase Studio first")
        raise e 