FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application files
COPY requirements.txt .
COPY api.py .
COPY kokoro_tts/__init__.py ./kokoro_tts/__init__.py

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p uploads outputs processing

# Port for FastAPI
EXPOSE 8080

# Command to start the API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]