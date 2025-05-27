#!/usr/bin/env python3
"""
Setup script for Hugging Face Inference API integration

This script helps you set up the Hugging Face API key for 10x faster translations.
"""

import os
import sys

def setup_huggingface_api():
    print("🚀 Setting up Hugging Face Inference API for faster translations")
    print("=" * 60)
    print()
    
    print("Benefits of using Hugging Face API:")
    print("• 10x faster translations (hosted models vs local)")
    print("• No GPU/CPU load on your machine")
    print("• Automatic fallback to local models if API fails")
    print("• Free tier: 30,000 characters/month")
    print()
    
    # Check if API key already exists
    existing_key = os.environ.get("HUGGINGFACE_API_KEY")
    if existing_key:
        print(f"✅ API key already configured: {existing_key[:8]}...{existing_key[-4:]}")
        print("The app will use Hugging Face API for faster translations!")
        return
    
    print("To get your FREE Hugging Face API key:")
    print("1. Go to: https://huggingface.co/settings/tokens")
    print("2. Sign up/Login to Hugging Face")
    print("3. Click 'New token' → 'Read' permissions → Create")
    print("4. Copy the token")
    print()
    
    api_key = input("Enter your Hugging Face API key (or press Enter to use local models): ").strip()
    
    if not api_key:
        print("⚠️  No API key provided. Using local models (slower but works offline)")
        return
    
    # Validate API key format
    if not api_key.startswith("hf_"):
        print("❌ Invalid API key format. Should start with 'hf_'")
        return
    
    # Set environment variable for current session
    os.environ["HUGGINGFACE_API_KEY"] = api_key
    
    # Create/update .env file
    env_file = ".env"
    env_content = ""
    
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            lines = f.readlines()
        
        # Remove existing HUGGINGFACE_API_KEY lines
        lines = [line for line in lines if not line.startswith("HUGGINGFACE_API_KEY=")]
        env_content = "".join(lines)
    
    # Add new API key
    env_content += f"HUGGINGFACE_API_KEY={api_key}\n"
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print("✅ API key saved successfully!")
    print(f"🔧 Added to {env_file}")
    print()
    print("To use the API key, restart your backend server:")
    print("1. Stop the current server (Ctrl+C)")
    print("2. Run: source .env && python main.py")
    print()
    print("🚀 Your translations will now be 10x faster!")

if __name__ == "__main__":
    setup_huggingface_api() 