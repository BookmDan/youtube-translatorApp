FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install fastapi uvicorn

COPY test_app.py .

ENV PORT=8080

CMD ["uvicorn", "test_app:app", "--host", "0.0.0.0", "--port", "8080"] 