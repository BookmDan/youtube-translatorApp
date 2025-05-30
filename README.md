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
<a href="https://www.youtube.com/watch?v=XVEquB0Cydo" target="_blank">
  <img src="https://github.com/user-attachments/assets/bb58b9a1-57dd-41f6-9fe0-a6b563edef9b" alt="youtube-translator" width="400"/>
</a>


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

## Local Development Setup

### Backend Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
cd backend
pip install -r requirements.txt

# Start backend server
uvicorn main:app --reload
```

### Frontend Setup
```bash
# Install dependencies
cd frontend
npm install

# Start frontend development server
npm start
```

## Docker Development

### Backend
```bash
# Build backend container
cd backend
docker build -t youtube-translator-backend .

# Run backend container
docker run -p 8000:8000 youtube-translator-backend
```

## GCP Deployment Setup

### Prerequisites
1. Install Google Cloud SDK
2. Install Terraform
3. Enable required GCP APIs:
```bash
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  containerregistry.googleapis.com \
  storage.googleapis.com
```

### Initial Setup
1. Create a new GCP project or select an existing one:
```bash
gcloud projects create [PROJECT_ID]
gcloud config set project [PROJECT_ID]
```

2. Configure GCP authentication:
```bash
gcloud auth login
gcloud auth application-default login
```

### Infrastructure Deployment (Terraform)
```bash
# Initialize Terraform
terraform init

# Plan changes
terraform plan

# Apply infrastructure
terraform apply
```

### Manual Deployment
```bash
# Build and push backend
docker build -t gcr.io/[PROJECT_ID]/youtube-translator-backend ./backend
docker push gcr.io/[PROJECT_ID]/youtube-translator-backend

# Deploy to Cloud Run
gcloud run deploy youtube-translator \
  --image gcr.io/[PROJECT_ID]/youtube-translator-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

# Deploy frontend to Storage
cd frontend
npm run build
gsutil -m cp -r build/* gs://[YOUR_BUCKET_NAME]/
```

### Automated Deployment (Cloud Build)
The repository includes Cloud Build configuration that automatically:
1. Builds the backend container
2. Deploys to Cloud Run
3. Builds the frontend
4. Deploys to Cloud Storage

Just push to the main branch:
```bash
git push origin main
```

## Environment Variables

### Backend (.env)
```
HUGGINGFACE_API_KEY=your_api_key_here
```

### Frontend (.env)
```
REACT_APP_API_URL=your_backend_url
```

## Architecture
- Frontend: React.js hosted on Google Cloud Storage
- Backend: FastAPI on Cloud Run
- CI/CD: Cloud Build
- Infrastructure: Terraform

