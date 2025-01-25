# Kokoro TTS API Documentation

## Base URL
```
http://localhost:8000
```

## Authentication

### Admin Endpoints
Admin endpoints require the `X-Admin-API-Key` header:
```
X-Admin-API-Key: your_admin_api_key
```

## Endpoints

### Book Analysis

#### Get Book Summary
```http
POST /book-summary
Content-Type: multipart/form-data
```

Analyzes an EPUB file and returns summary information including price.

**Parameters:**
- `file`: EPUB file (required)

**Response:**
```json
{
  "success": true,
  "summary": {
    "total_chapters": 10,
    "total_words": 50000,
    "total_duration": 333.33,
    "price": 6.47
  },
  "chapters": [
    {
      "order": 1,
      "title": "Chapter 1",
      "word_count": 5000,
      "estimated_duration": 33.33
    }
  ]
}
```

### Book Conversion

#### Start Conversion
```http
POST /convert
Content-Type: multipart/form-data
```

Initiates the book conversion process.

**Parameters:**
- `file`: EPUB file (required)
- `voice`: Voice ID (required)
- `speed`: Speech speed (optional, default: 1.0)
- `lang`: Language code (optional, default: "en-us")
- `email`: User email (required)

**Response:**
```json
{
  "job_id": "uuid-string",
  "message": "Conversion started"
}
```

#### Check Conversion Status
```http
GET /status/{job_id}
```

Gets the current status of a conversion job.

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "queued|processing|completed|failed",
  "progress": 45.5,
  "output_file": "path/to/output.mp3",
  "error": "error message if failed"
}
```

### Payment Processing

#### Create Payment Intent
```http
POST /create-payment-intent
Content-Type: application/json
```

Creates a Stripe payment intent for processing payment.

**Request Body:**
```json
{
  "job_id": "uuid-string",
  "email": "user@example.com"
}
```

**Response:**
```json
{
  "clientSecret": "pi_..._secret_...",
  "amount": 647,
  "transaction_id": "uuid-string"
}
```

#### Stripe Webhook
```http
POST /webhook
```

Handles Stripe webhook events. Must be configured in Stripe dashboard.

**Headers:**
- `stripe-signature`: Webhook signature from Stripe

**Response:**
```json
{
  "status": "success"
}
```

### Voice and Language Options

#### List Available Voices
```http
GET /voices
```

Returns list of available voices.

**Response:**
```json
{
  "voices": [
    "voice_id_1",
    "voice_id_2"
  ]
}
```

#### List Supported Languages
```http
GET /languages
```

Returns list of supported languages.

**Response:**
```json
{
  "languages": [
    "en-us",
    "es-es",
    "fr-fr"
  ]
}
```

### Admin Dashboard

#### List Transactions
```http
GET /admin/transactions
X-Admin-API-Key: your_admin_api_key
```

Lists all transactions with optional status filter.

**Query Parameters:**
- `status`: Filter by status (optional)

**Response:**
```json
[
  {
    "transaction_id": "uuid-string",
    "stripe_payment_intent_id": "pi_...",
    "user_email": "user@example.com",
    "amount": 6.47,
    "status": "completed",
    "created_at": "2024-...",
    "epub_title": "Book Title",
    "word_count": 50000,
    "error": null
  }
]
```

#### Get Transaction Details
```http
GET /admin/transaction/{transaction_id}
X-Admin-API-Key: your_admin_api_key
```

Gets detailed information about a specific transaction.

**Response:**
```json
{
  "transaction": {
    "transaction_id": "uuid-string",
    "stripe_payment_intent_id": "pi_...",
    "user_email": "user@example.com",
    "amount": 6.47,
    "status": "completed",
    "created_at": "2024-...",
    "epub_title": "Book Title",
    "word_count": 50000,
    "error": null
  },
  "stripe_details": {
    // Full Stripe payment intent details
  },
  "job": {
    // Associated job details
  }
}
```

#### Process Refund
```http
POST /admin/refund/{transaction_id}
X-Admin-API-Key: your_admin_api_key
```

Processes a refund for a completed transaction.

**Response:**
```json
{
  "status": "success",
  "refund": {
    // Stripe refund details
  }
}
```

## Error Responses

All endpoints return error responses in the following format:

```json
{
  "detail": {
    "type": "error_type",
    "msg": "Error description"
  }
}
```

Common error types:
- `value_error`: Invalid input data
- `upload_error`: File upload issues
- `processing_error`: EPUB processing problems
- `payment_error`: Payment-related issues
- `auth_error`: Authentication issues

## Status Codes

- `200`: Success
- `201`: Created
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `500`: Internal Server Error

## Rate Limiting

- Admin endpoints: 100 requests per minute
- Public endpoints: 60 requests per minute per IP
- Webhook endpoint: Unlimited

## Webhook Events

The following Stripe webhook events should be configured:
- `payment_intent.succeeded`
- `payment_intent.payment_failed`
- `charge.refunded`

## Testing

Test cards for Stripe:
- Success: `4242 4242 4242 4242`
- Decline: `4000 0000 0000 0002`
- Authentication Required: `4000 0025 0000 3155` 