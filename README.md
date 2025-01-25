# Kokoro TTS API

API for converting EPUB books to audio using Kokoro TTS.

## Features

- EPUB to audio conversion
- Multiple voices and languages support
- Secure payment processing with Stripe
- Admin dashboard for transaction management
- Email notifications
- Progress tracking
- Chunked audio processing

## Environment Setup

Create a `.env` file in the root directory with the following variables:

```env
# Brevo (Email)
BREVO_API_KEY=your_brevo_api_key

# Hugging Face
HUGGING_FACE_TOKEN=your_huggingface_token

# API Configuration
API_BASE_URL=http://localhost:8000

# Stripe Configuration
STRIPE_SECRET_KEY=sk_test_your_stripe_secret_key
STRIPE_WEBHOOK_SECRET=whsec_your_stripe_webhook_secret

# Admin Configuration
ADMIN_API_KEY=your_secure_admin_api_key
```

## Payment System

The API uses Stripe for secure payment processing. The payment flow is as follows:

1. Upload EPUB and get price:
```bash
POST /book-summary
Content-Type: multipart/form-data
file: your_book.epub

Response:
{
  "success": true,
  "summary": {
    "total_chapters": 10,
    "total_words": 50000,
    "total_duration": 333.33,
    "price": 6.47
  },
  "chapters": [...]
}
```

2. Create payment intent:
```bash
POST /create-payment-intent
Content-Type: application/json
{
  "job_id": "your_job_id",
  "email": "user@example.com"
}

Response:
{
  "clientSecret": "pi_..._secret_...",
  "amount": 647,
  "transaction_id": "..."
}
```

3. Process payment in frontend:
```javascript
// Using Stripe Elements
const {error} = await stripe.confirmPayment({
  clientSecret,
  elements
});
```

4. Webhook notification:
- Configure your Stripe webhook to point to `/webhook`
- The API will automatically start processing the book after successful payment

## Admin Dashboard

Secure admin endpoints for managing transactions:

### List Transactions

```bash
GET /admin/transactions
X-Admin-API-Key: your_admin_api_key

# Optional status filter
GET /admin/transactions?status=completed
```

### Transaction Details

```bash
GET /admin/transaction/{transaction_id}
X-Admin-API-Key: your_admin_api_key

Response:
{
  "transaction": {
    "transaction_id": "...",
    "stripe_payment_intent_id": "...",
    "user_email": "...",
    "amount": 6.47,
    "status": "completed",
    "created_at": "2024-...",
    "epub_title": "...",
    "word_count": 50000
  },
  "stripe_details": {...},
  "job": {...}
}
```

### Process Refund

```bash
POST /admin/refund/{transaction_id}
X-Admin-API-Key: your_admin_api_key

Response:
{
  "status": "success",
  "refund": {...}
}
```

## Pricing

The pricing is calculated based on word count:

- Up to 10,000 words: $3.00
- Between 10,001 and 100,000 words: Proportional from $3.00 to $9.95
- Over 100,000 words: $9.95 + $0.10 per additional 1,000 words

## Security Features

- Price verification before payment processing
- Secure webhook handling with signature verification
- Admin API key authentication
- Database transaction safety
- Automatic cleanup of temporary files
- Email notifications for important events

## Error Handling

The API uses standard HTTP status codes and returns detailed error messages:

```json
{
  "detail": {
    "type": "value_error",
    "msg": "Error description"
  }
}
```

Common error types:
- `value_error`: Invalid input data
- `upload_error`: File upload issues
- `processing_error`: EPUB processing problems
- `payment_error`: Payment-related issues

## Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`

3. Run the server:
```bash
uvicorn api:app --reload
```

4. Configure Stripe webhook:
```bash
stripe listen --forward-to localhost:8000/webhook
```

## Production Considerations

1. Use proper SSL/TLS certificates
2. Set up proper CORS configuration
3. Use production Stripe keys
4. Configure proper database connection pooling
5. Set up monitoring and logging
6. Use secure admin API keys
7. Configure proper email templates
8. Set up backup systems for audio files

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

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
