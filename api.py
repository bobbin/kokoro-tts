from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks, Form, File, Depends, Header, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
import stripe
import shutil
import os
import uuid
from typing import Optional, List
from pydantic import BaseModel, EmailStr
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import asyncio
from gradio_client import Client
import sib_api_v3_sdk
from sib_api_v3_sdk import Configuration, ApiClient, TransactionalEmailsApi, SendSmtpEmail
from sqlalchemy import create_engine, Column, String, Float, DateTime, Text, Boolean, Integer, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.pool import QueuePool
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Stripe configuration
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

if not all([STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET, ADMIN_API_KEY]):
    raise ValueError("Missing required environment variables for Stripe/Admin configuration")

stripe.api_key = STRIPE_SECRET_KEY

# Admin API key security
api_key_header = APIKeyHeader(name="X-Admin-API-Key")

async def verify_admin_api_key(api_key: str = Depends(api_key_header)):
    if api_key != ADMIN_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )
    return api_key

# Import Kokoro processing functions
from kokoro_tts import (
    convert_text_to_audio,
    validate_voice,
    calculate_book_summary
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Kokoro TTS API",
    description="API for converting EPUB books to audio using Kokoro TTS",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, cambia esto por la URL específica de tu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
PROCESSING_DIR = "processing"
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, PROCESSING_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Initialize thread pool
thread_pool = ThreadPoolExecutor(max_workers=3)  # Ajusta según necesidades

# Initialize Hugging Face client
try:
    client = Client("bobbin28/Kokoro-TTS-Zero", hf_token=os.getenv("HUGGING_FACE_TOKEN"))
except Exception as e:
    logger.error(f"Failed to initialize Hugging Face client: {e}")
    raise

# Database configuration
DATABASE_URL = "postgresql://neondb_owner:npg_yEa2POCRgNf5@ep-tight-dust-a4gcy3r9.us-east-1.aws.neon.tech/neondb"
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True,
    connect_args={
        "sslmode": "require",
        "connect_timeout": 10
    }
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Brevo configuration
BREVO_API_KEY = os.getenv("BREVO_API_KEY")
if not BREVO_API_KEY:
    raise ValueError("BREVO_API_KEY environment variable is not set")

try:
    brevo_configuration = Configuration()
    brevo_configuration.api_key['api-key'] = BREVO_API_KEY
    brevo_api = TransactionalEmailsApi(ApiClient(brevo_configuration))
    
    # Test the API key with a simple API call
    brevo_api.get_account()
    logger.info("Brevo API connection successful")
except Exception as e:
    logger.error(f"Failed to initialize Brevo API: {e}")
    raise

# Database Models
class Job(Base):
    __tablename__ = "jobs"

    job_id = Column(UUID(as_uuid=True), primary_key=True)
    user_email = Column(String, nullable=False)
    status = Column(String, nullable=False)
    progress = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    epub_title = Column(String)
    voice = Column(String)
    speed = Column(Float)
    language = Column(String)
    output_file = Column(String)
    error = Column(Text)
    transaction_id = Column(UUID(as_uuid=True), ForeignKey('transactions.transaction_id'), nullable=True)
    transaction = relationship("Transaction", back_populates="job")

class Transaction(Base):
    __tablename__ = "transactions"

    transaction_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    stripe_payment_intent_id = Column(String, unique=True)
    user_email = Column(String, nullable=False)
    amount = Column(Float, nullable=False)  # Amount in USD
    status = Column(String, nullable=False)  # pending, completed, failed, refunded
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    epub_title = Column(String)
    word_count = Column(Integer)
    error = Column(Text)
    job = relationship("Job", back_populates="transaction", uselist=False)

# Create tables
Base.metadata.create_all(bind=engine)

# Retry decorator for database operations
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def get_db_with_retry():
    db = SessionLocal()
    try:
        # Test the connection
        db.execute("SELECT 1")
        return db
    except Exception as e:
        db.close()
        raise e

# Modified helper function to get database session with retry
def get_db():
    db = get_db_with_retry()
    try:
        yield db
    finally:
        db.close()

async def send_completion_email(email: str, job_id: str, output_url: str):
    """Send completion email to user using Brevo template"""
    try:
        # Initialize Brevo client
        configuration = Configuration()
        configuration.api_key['api-key'] = os.getenv("BREVO_API_KEY")
        api_instance = TransactionalEmailsApi(ApiClient(configuration))
        
        # Get template ID by name
        templates = api_instance.get_smtp_templates()
        template_id = None
        template_name = "Audiobook"
        
        for template in templates.templates:
            if template.name == template_name:
                template_id = template.id
                break
        
        if template_id is None:
            logger.error(f"❌ Template '{template_name}' not found")
            logger.info("Available templates:")
            for template in templates.templates:
                logger.info(f"  - {template.name} (ID: {template.id})")
            raise ValueError(f"Template '{template_name}' not found")
        
        logger.info(f"Found template: {template_name} (ID: {template_id})")
        
        # Prepare email parameters
        sender = {"name": "Kokoro TTS", "email": "jaldao27@gmail.com"}
        to = [{"email": email}]
        
        # Parameters that will replace variables in the template
        params = {
            "download_url": output_url,
            "user_email": email
        }
        
        # Create email object using template
        send_smtp_email = SendSmtpEmail(
            to=to,
            sender=sender,
            template_id=template_id,
            params=params,
            reply_to={"email": "jaldao27@gmail.com", "name": "Kokoro TTS Support"}
        )
        
        # Send email
        logger.info(f"Sending template email to {email}...")
        logger.info("Email configuration:")
        logger.info(f"- Template: {template_name} (ID: {template_id})")
        logger.info(f"- From: {sender['name']} <{sender['email']}>")
        logger.info(f"- To: {email}")
        logger.info(f"- Parameters: {params}")
        
        response = api_instance.send_transac_email(send_smtp_email)
        logger.info(f"✅ Email sent successfully to {email} for job {job_id}")
        return True
    except Exception as e:
        logger.error(f"Error sending email for job {job_id}: {str(e)}")
        # Update job status to indicate email failure
        try:
            db = get_db_with_retry()
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if job:
                job.error = f"Job completed but failed to send email: {str(e)}"
                db.commit()
        except Exception as db_error:
            logger.error(f"Error updating job status after email failure: {db_error}")
        finally:
            if 'db' in locals():
                db.close()
        return False

class ConversionRequest(BaseModel):
    email: EmailStr

def process_epub_sync(job_id: str, epub_path: str, voice: str, speed: float = 1.0, lang: str = "en-us", email: str = None):
    """Synchronous processing function to run in thread pool"""
    try:
        db = get_db_with_retry()
        # Update job status in database
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found in database")
            return

        job.status = "processing"
        job.progress = 0.0
        db.commit()

        # Create job-specific directory for split output
        split_output = os.path.join(PROCESSING_DIR, job_id)
        os.makedirs(split_output, exist_ok=True)
        
        # Extract EPUB title
        import ebooklib
        from ebooklib import epub
        book = epub.read_epub(epub_path)
        
        # Try to get title from different sources
        title = None
        try:
            if book.get_metadata('DC', 'title'):
                title = book.get_metadata('DC', 'title')[0][0]
            if not title:
                title = os.path.splitext(os.path.basename(epub_path))[0]
            title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
            title = title.replace(' ', '_')
            
            # Update job with EPUB title
            job.epub_title = title
            db.commit()
        except Exception as e:
            logger.warning(f"Could not extract title from EPUB: {e}")
            title = "audiobook"
        
        # Create job output directory
        job_output_dir = os.path.join(OUTPUT_DIR, str(job_id))
        os.makedirs(job_output_dir, exist_ok=True)
        
        output_file = os.path.join(job_output_dir, f"{title}.mp3")
        
        def progress_callback(current_chapter: int, total_chapters: int):
            progress = (current_chapter / total_chapters) * 90
            job.progress = progress
            db.commit()
            logger.info(f"Job {job_id} progress: {progress:.1f}%")
        
        # Process EPUB
        convert_text_to_audio(
            input_file=epub_path,
            output_file=None,
            voice=voice,
            speed=speed,
            lang=lang,
            split_output=split_output,
            format="mp3",
            debug=True,
            interactive=False,
            progress_callback=progress_callback
        )
        
        # Merge chunks
        logger.info(f"Merging chunks for job {job_id}")
        from kokoro_tts import merge_chunks_to_chapters
        merge_chunks_to_chapters(split_output, format="mp3")
        
        # Concatenate chapters
        from pydub import AudioSegment
        chapter_files = sorted([
            os.path.join(split_output, f) 
            for f in os.listdir(split_output) 
            if f.startswith("chapter_") and f.endswith(".mp3")
        ])
        
        if chapter_files:
            combined = AudioSegment.empty()
            for i, chapter_file in enumerate(chapter_files, 1):
                audio = AudioSegment.from_mp3(chapter_file)
                combined += audio
                merge_progress = 90 + (i / len(chapter_files) * 10)
                job.progress = merge_progress
                db.commit()
            
            combined.export(output_file, format="mp3")
            
            # Update job status
            job.status = "completed"
            job.progress = 100.0
            job.output_file = output_file
            db.commit()

            # Send completion email
            if email:
                # Generate download URL based on your domain and setup
                base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
                download_url = f"{base_url}/download/{job_id}"
                # Use asyncio.run since we're in a sync function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(send_completion_email(email, str(job_id), download_url))
                finally:
                    loop.close()
        else:
            raise FileNotFoundError("No chapter files found to merge")
        
        # Cleanup
        shutil.rmtree(split_output)
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")
        try:
            job.status = "failed"
            job.error = str(e)
            db.commit()
        except Exception as commit_error:
            logger.error(f"Error updating job status: {commit_error}")
        
        if os.path.exists(job_output_dir):
            shutil.rmtree(job_output_dir)
        
    finally:
        if os.path.exists(epub_path):
            os.remove(epub_path)
        if os.path.exists(split_output):
            shutil.rmtree(split_output)
        if 'db' in locals():
            db.close()

async def process_epub(job_id: str, epub_path: str, voice: str, speed: float = 1.0, lang: str = "en-us", email: str = None):
    """Asynchronous wrapper for processing function"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        thread_pool,
        process_epub_sync,
        job_id,
        epub_path,
        voice,
        speed,
        lang,
        email
    )

@app.post("/convert")
async def convert_epub(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    voice: str = Form(...),
    speed: float = Form(1.0),
    lang: str = Form("en-us"),
    email: str = Form(...)
):
    """Convert EPUB to audio using Kokoro TTS"""
    try:
        db = get_db_with_retry()
        logger.info(f"Received request - File: {file.filename}, Voice: {voice}, Speed: {speed}, Lang: {lang}, Email: {email}")
        
        if not file.filename.endswith('.epub'):
            raise HTTPException(
                status_code=400,
                detail={"type": "value_error", "loc": ["file"], "msg": "Only EPUB files are supported"}
            )
        
        try:
            voice = validate_voice(voice)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail={"type": "value_error", "loc": ["voice"], "msg": str(e)}
            )
        
        job_id = str(uuid.uuid4())
        epub_path = os.path.join(UPLOAD_DIR, f"{job_id}.epub")
        
        try:
            with open(epub_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={"type": "upload_error", "msg": f"Error saving uploaded file: {str(e)}"}
            )
        
        # Create job in database with retry
        try:
            new_job = Job(
                job_id=job_id,
                user_email=email,
                status="queued",
                progress=0.0,
                voice=voice,
                speed=speed,
                language=lang
            )
            db.add(new_job)
            db.commit()
        except Exception as e:
            logger.error(f"Error creating job in database: {e}")
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail={"type": "database_error", "msg": "Error creating job in database"}
            )
        
        # Start processing in background
        background_tasks.add_task(
            process_epub,
            job_id,
            epub_path,
            voice,
            speed,
            lang,
            email
        )
        
        return {"job_id": job_id}
        
    except Exception as e:
        logger.error(f"Error in convert_epub: {e}")
        raise HTTPException(
            status_code=500,
            detail={"type": "server_error", "msg": str(e)}
        )
    finally:
        if 'db' in locals():
            db.close()

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get status of a conversion job"""
    try:
        db = get_db_with_retry()
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return {
            "job_id": job.job_id,
            "status": job.status,
            "progress": job.progress,
            "output_file": job.output_file,
            "error": job.error
        }
    finally:
        if 'db' in locals():
            db.close()

@app.get("/voices")
async def list_voices():
    """Get list of available voices"""
    try:
        response = client.predict(api_name="/initialize_model")
        voices = [voice[0] for voice in response['choices']]
        return {"voices": voices}
    except Exception as e:
        logger.error(f"Error getting voices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/languages")
async def list_languages():
    """Get list of supported languages"""
    # Since we don't have direct access to supported languages through the API,
    # we'll maintain a list of known supported languages
    languages = ["en-us", "es-es", "fr-fr", "de-de", "it-it", "pt-br", "ja-jp"]
    return {"languages": languages}

@app.post("/book-summary")
async def get_book_summary(file: UploadFile = File(...)):
    """Calculate book summary from EPUB file"""
    try:
        # Log received file
        logger.info(f"Received request for book summary - File: {file.filename}")
        
        # Validate file type
        if not file.filename.endswith('.epub'):
            raise HTTPException(
                status_code=400,
                detail={"type": "value_error", "loc": ["file"], "msg": "Only EPUB files are supported"}
            )
        
        # Generate temporary ID for the file
        temp_id = str(uuid.uuid4())
        
        # Save uploaded file temporarily
        epub_path = os.path.join(UPLOAD_DIR, f"{temp_id}.epub")
        try:
            with open(epub_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={"type": "upload_error", "msg": f"Error saving uploaded file: {str(e)}"}
            )
        
        try:
            # Calculate book summary
            result = calculate_book_summary(epub_path)
            
            if not result["success"]:
                raise HTTPException(
                    status_code=400,
                    detail={"type": "processing_error", "msg": result["error"]}
                )
                
            return result
            
        finally:
            # Clean up - remove temporary file
            if os.path.exists(epub_path):
                os.remove(epub_path)
                logger.info(f"Removed temporary file: {epub_path}")
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_book_summary: {e}")
        raise HTTPException(
            status_code=500,
            detail={"type": "server_error", "msg": str(e)}
        )

class PaymentIntentRequest(BaseModel):
    job_id: str
    email: EmailStr

class AdminTransactionResponse(BaseModel):
    transaction_id: str
    stripe_payment_intent_id: str
    user_email: str
    amount: float
    status: str
    created_at: datetime
    epub_title: str
    word_count: int
    error: Optional[str]

@app.post("/create-payment-intent")
async def create_payment_intent(request: PaymentIntentRequest):
    """Create a Stripe PaymentIntent for a book conversion job"""
    try:
        db = get_db_with_retry()
        
        # Get job details
        job = db.query(Job).filter(Job.job_id == request.job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Verify email matches
        if job.user_email != request.email:
            raise HTTPException(status_code=403, detail="Email does not match job")
        
        # Re-calculate price to prevent manipulation
        epub_path = os.path.join(UPLOAD_DIR, f"{job.job_id}.epub")
        if not os.path.exists(epub_path):
            raise HTTPException(status_code=404, detail="EPUB file not found")
        
        summary = calculate_book_summary(epub_path)
        if not summary["success"]:
            raise HTTPException(status_code=400, detail="Could not calculate book price")
        
        amount = int(summary["summary"]["price"] * 100)  # Convert to cents for Stripe
        
        # Create Stripe PaymentIntent
        try:
            intent = stripe.PaymentIntent.create(
                amount=amount,
                currency="usd",
                metadata={
                    "job_id": str(job.job_id),
                    "email": request.email,
                    "epub_title": job.epub_title,
                    "word_count": summary["summary"]["total_words"]
                }
            )
            
            # Create transaction record
            transaction = Transaction(
                transaction_id=uuid.uuid4(),
                stripe_payment_intent_id=intent.id,
                user_email=request.email,
                amount=amount / 100,  # Store in dollars
                status="pending",
                epub_title=job.epub_title,
                word_count=summary["summary"]["total_words"]
            )
            db.add(transaction)
            
            # Link transaction to job
            job.transaction_id = transaction.transaction_id
            db.commit()
            
            return {
                "clientSecret": intent.client_secret,
                "amount": amount,
                "transaction_id": str(transaction.transaction_id)
            }
            
        except stripe.error.StripeError as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    finally:
        if 'db' in locals():
            db.close()

@app.post("/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events"""
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    if event["type"] == "payment_intent.succeeded":
        payment_intent = event["data"]["object"]
        
        try:
            db = get_db_with_retry()
            
            # Update transaction status
            transaction = db.query(Transaction).filter(
                Transaction.stripe_payment_intent_id == payment_intent.id
            ).first()
            
            if transaction:
                transaction.status = "completed"
                
                # Get associated job
                job = transaction.job
                if job:
                    # Start processing in background
                    background_tasks = BackgroundTasks()
                    epub_path = os.path.join(UPLOAD_DIR, f"{job.job_id}.epub")
                    
                    background_tasks.add_task(
                        process_epub,
                        str(job.job_id),
                        epub_path,
                        job.voice,
                        job.speed,
                        job.language,
                        job.user_email
                    )
                
                db.commit()
                
        finally:
            if 'db' in locals():
                db.close()
    
    return {"status": "success"}

# Admin Dashboard Endpoints
@app.get("/admin/transactions", response_model=List[AdminTransactionResponse])
async def get_transactions(
    status: Optional[str] = None,
    api_key: str = Depends(verify_admin_api_key)
):
    """Get all transactions with optional status filter"""
    try:
        db = get_db_with_retry()
        query = db.query(Transaction)
        
        if status:
            query = query.filter(Transaction.status == status)
        
        transactions = query.order_by(Transaction.created_at.desc()).all()
        return transactions
    finally:
        if 'db' in locals():
            db.close()

@app.get("/admin/transaction/{transaction_id}")
async def get_transaction(
    transaction_id: str,
    api_key: str = Depends(verify_admin_api_key)
):
    """Get detailed information about a specific transaction"""
    try:
        db = get_db_with_retry()
        transaction = db.query(Transaction).filter(
            Transaction.transaction_id == transaction_id
        ).first()
        
        if not transaction:
            raise HTTPException(status_code=404, detail="Transaction not found")
        
        # Get Stripe payment details
        try:
            payment_intent = stripe.PaymentIntent.retrieve(transaction.stripe_payment_intent_id)
        except stripe.error.StripeError:
            payment_intent = None
        
        return {
            "transaction": transaction,
            "stripe_details": payment_intent,
            "job": transaction.job
        }
    finally:
        if 'db' in locals():
            db.close()

@app.post("/admin/refund/{transaction_id}")
async def refund_transaction(
    transaction_id: str,
    api_key: str = Depends(verify_admin_api_key)
):
    """Refund a transaction"""
    try:
        db = get_db_with_retry()
        transaction = db.query(Transaction).filter(
            Transaction.transaction_id == transaction_id
        ).first()
        
        if not transaction:
            raise HTTPException(status_code=404, detail="Transaction not found")
        
        if transaction.status != "completed":
            raise HTTPException(status_code=400, detail="Transaction cannot be refunded")
        
        try:
            refund = stripe.Refund.create(
                payment_intent=transaction.stripe_payment_intent_id
            )
            
            transaction.status = "refunded"
            db.commit()
            
            return {"status": "success", "refund": refund}
            
        except stripe.error.StripeError as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    finally:
        if 'db' in locals():
            db.close() 