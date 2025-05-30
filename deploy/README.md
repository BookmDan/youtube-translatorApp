# Deployment Configuration

This directory contains all deployment-related configurations for the YouTube Translator application.

## Directory Structure

```
deploy/
├── docker/           # Docker-related files
│   ├── Dockerfile.dev   # Development environment
│   └── Dockerfile.prod  # Production environment
├── terraform/        # Infrastructure as Code
│   ├── main.tf         # Main Terraform configuration
│   └── variables.tf    # Variable definitions
└── cloudbuild/      # CI/CD configurations
    └── cloudbuild.yaml # Google Cloud Build config
```

## Usage

### Local Development
```bash
# Build and run development container
docker build -f deploy/docker/Dockerfile.dev -t youtube-translator-dev .
docker run -p 8080:8080 youtube-translator-dev
```

### Production Deployment
The production deployment is handled automatically by Google Cloud Build when changes are pushed to the main branch.

### Infrastructure Management
```bash
# Initialize Terraform
cd deploy/terraform
terraform init

# Plan changes
terraform plan

# Apply changes
terraform apply
```

## Configuration Files

- `Dockerfile.dev`: Includes hot-reload and debugging features for development
- `Dockerfile.prod`: Optimized for production with proper health checks and worker configuration
- `cloudbuild.yaml`: Defines the CI/CD pipeline for Google Cloud Build
- Terraform files: Define and manage the cloud infrastructure

## Environment Variables

The following environment variables are required:
- `HUGGINGFACE_API_KEY`: API key for Hugging Face services
- Additional environment variables are set in the Cloud Run configuration 