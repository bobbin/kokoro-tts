# Kokoro TTS API

A REST API service for converting EPUB books to audio using the Kokoro TTS engine.

## Features

- EPUB to MP3 conversion
- Multiple language and voice support
- Background processing with status tracking
- RESTful API endpoints
- Progress monitoring
- Detailed error reporting

## Prerequisites

- Python 3.x
- Required Python packages:
  - fastapi
  - uvicorn
  - python-multipart
  - soundfile
  - sounddevice
  - kokoro_onnx
  - ebooklib
  - beautifulsoup4

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/kokoro-tts-api.git
cd kokoro-tts-api
```

2. Install required packages:
```bash
pip install fastapi uvicorn python-multipart soundfile sounddevice kokoro_onnx ebooklib beautifulsoup4
```

3. Download the required model files:
```bash
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx
```

## Running the API

Start the API server:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API documentation will be available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### POST /convert
Convert EPUB file to audio

**Parameters:**
- `file`: EPUB file (multipart/form-data)
- `voice`: Voice ID (string)
- `speed`: Speech speed (float, default: 1.0)
- `lang`: Language code (string, default: "en-us")

**Response:**
```json
{
    "job_id": "uuid",
    "message": "Conversion started"
}
```

### GET /status/{job_id}
Get conversion job status

**Response:**
```json
{
    "job_id": "uuid",
    "status": "queued|processing|completed|failed",
    "progress": 45.5,
    "output_file": "path/to/output.mp3",
    "error": "error message if failed"
}
```

### GET /voices
Get list of available voices

**Response:**
```json
{
    "voices": ["voice1", "voice2", ...]
}
```

### GET /languages
Get list of supported languages

**Response:**
```json
{
    "languages": ["en-us", "es-es", ...]
}
```

## Example Usage

Using curl:
```bash
# Convert EPUB to audio
curl -X POST "http://localhost:8000/convert" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@book.epub" \
     -F "voice=af_sarah" \
     -F "speed=1.0" \
     -F "lang=en-us"

# Check conversion status
curl "http://localhost:8000/status/job-id-here"

# List available voices
curl "http://localhost:8000/voices"

# List supported languages
curl "http://localhost:8000/languages"
```

Using Python requests:
```python
import requests

# Convert EPUB to audio
files = {'file': open('book.epub', 'rb')}
data = {'voice': 'af_sarah', 'speed': 1.0, 'lang': 'en-us'}
response = requests.post('http://localhost:8000/convert', files=files, data=data)
job_id = response.json()['job_id']

# Check conversion status
status = requests.get(f'http://localhost:8000/status/{job_id}')
print(status.json())
```

## Directory Structure

- `/uploads`: Temporary storage for uploaded EPUB files
- `/processing`: Temporary directory for processing files
- `/outputs`: Directory containing generated MP3 files

## Error Handling

The API uses standard HTTP status codes:
- 200: Success
- 400: Bad Request (invalid parameters)
- 404: Not Found (job ID not found)
- 500: Internal Server Error

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
