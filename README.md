# 🎬 YouTube Korean Translator

A powerful web application that translates Korean YouTube video subtitles into multiple languages using state-of-the-art machine learning models.

## ✨ Features

- **Multi-language Support**: Translate Korean subtitles to:
  - English (direct translation)
  - Spanish (direct translation)
  - French (direct translation)
  - Chinese (via English)
  - Arabic (via English)
  - Tagalog (via English)
  - Swedish (via English)

- **Real-time Translation**: Get instant translations using Helsinki-NLP models
- **Beautiful UI**: Modern Material-UI interface
- **Subtitle Extraction**: Automatically extracts Korean subtitles from YouTube videos
- **Translation Preview**: Shows first 20 lines of translation for quick review

## 🎥 Demo

[Watch the Demo Video](https://www.youtube.com/watch?v=XVEquB0Cydo)


### Screenshots
![youtube-translator](https://github.com/user-attachments/assets/bb58b9a1-57dd-41f6-9fe0-a6b563edef9b)

## 🛠️ Tech Stack

### Backend
- FastAPI
- PyTorch
- HuggingFace Transformers
- Helsinki-NLP Translation Models
- YouTube Transcript API

### Frontend
- React.js
- Material-UI
- Axios

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Node.js 14+
- npm or yarn

### Installation

1. Clone the repository:
```bash
git clone https://github.com/BookmDan/youtube-translator-app.git
cd youtube-translator-app
```

2. Set up the backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up the frontend:
```bash
cd frontend
npm install
```

### Running the Application

1. Start the backend server:
```bash
cd backend
uvicorn main:app --reload
```

2. Start the frontend development server:
```bash
cd frontend
npm start
```

3. Open your browser and navigate to `http://localhost:3000`

## 📝 Usage

1. Enter a YouTube URL with Korean subtitles
2. Select your target language
3. Click "Translate Video"
4. View the translated subtitles in real-time

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Helsinki-NLP](https://huggingface.co/Helsinki-NLP) for the translation models
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api) for subtitle extraction
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [Material-UI](https://mui.com/) for the frontend components

