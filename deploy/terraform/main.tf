terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Variables
variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region"
  type        = string
  default     = "us-central1"
}

variable "frontend_domain" {
  description = "The domain for the frontend"
  type        = string
  default     = ""
}

variable "huggingface_api_key" {
  description = "The HuggingFace API key"
  type        = string
  sensitive   = true
}

# Frontend Storage Bucket
resource "google_storage_bucket" "frontend" {
  name          = "${var.project_id}-frontend"
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

  uniform_bucket_level_access = true
}

# Make bucket public
resource "google_storage_bucket_iam_member" "public_read" {
  bucket = google_storage_bucket.frontend.name
  role   = "roles/storage.objectViewer"
  member = "allUsers"
}

# Cloud Run service
resource "google_cloud_run_service" "backend" {
  name     = "youtube-translator"
  location = var.region

  template {
    spec {
      containers {
        image = "gcr.io/${var.project_id}/youtube-translator-backend"
        
        resources {
          limits = {
            cpu    = "2000m"
            memory = "2Gi"
          }
        }

        env {
          name  = "HUGGINGFACE_API_KEY"
          value = var.huggingface_api_key
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

# Make Cloud Run service public
resource "google_cloud_run_service_iam_member" "public" {
  service  = google_cloud_run_service.backend.name
  location = google_cloud_run_service.backend.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Cloud Build trigger
resource "google_cloudbuild_trigger" "deploy" {
  name = "deploy-youtube-translator"

  github {
    owner = "BookmDan"
    name  = "youtube-translatorApp"
    push {
      branch = "^main$"
    }
  }

  filename = "cloudbuild.yaml"
}

# Outputs
output "frontend_url" {
  value = "https://storage.googleapis.com/${google_storage_bucket.frontend.name}/index.html"
}

output "backend_url" {
  value = google_cloud_run_service.backend.status[0].url
}

# Optional: Cloud CDN setup if frontend_domain is provided
resource "google_compute_backend_bucket" "frontend" {
  count       = var.frontend_domain != "" ? 1 : 0
  name        = "frontend-backend-bucket"
  bucket_name = google_storage_bucket.frontend.name
  enable_cdn  = true
} 