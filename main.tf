terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

provider "google" {
  project = "YOUR_PROJECT_ID"
  region  = "us-central1"
}

# Frontend storage bucket
resource "google_storage_bucket" "frontend" {
  name          = "your-app-frontend"
  location      = "US"
  force_destroy = true

  website {
    main_page_suffix = "index.html"
    not_found_page   = "index.html"
  }

  cors {
    origin          = ["*"]
    method          = ["GET", "HEAD", "OPTIONS"]
    response_header = ["*"]
    max_age_seconds = 3600
  }
}

# Cloud Run service
resource "google_cloud_run_service" "backend" {
  name     = "youtube-translator"
  location = "us-central1"

  template {
    spec {
      containers {
        image = "gcr.io/YOUR_PROJECT_ID/youtube-translator"
        resources {
          limits = {
            cpu    = "2000m"
            memory = "2Gi"
          }
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

# Make the Cloud Run service public
resource "google_cloud_run_service_iam_member" "public" {
  service  = google_cloud_run_service.backend.name
  location = google_cloud_run_service.backend.location
  role     = "roles/run.invoker"
  member   = "allUsers"
} 