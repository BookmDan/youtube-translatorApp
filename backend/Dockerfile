FROM python:3.9-slim

WORKDIR /app

# Create necessary directories first
RUN mkdir -p translations transcripts model_cache output

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure directories exist and have proper permissions
RUN chmod -R 777 translations transcripts model_cache output

# Expose the port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 