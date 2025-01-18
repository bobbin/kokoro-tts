from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks, Form, File
from fastapi.responses import JSONResponse
import shutil
import os
import uuid
from typing import Optional
from pydantic import BaseModel
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Import Kokoro processing functions
from kokoro_tts import (
    convert_text_to_audio,
    Kokoro,
    validate_voice
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

# Configuration
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
PROCESSING_DIR = "processing"
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, PROCESSING_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Initialize thread pool
thread_pool = ThreadPoolExecutor(max_workers=3)  # Ajusta según necesidades

# Initialize Kokoro model
try:
    kokoro = Kokoro("kokoro-v0_19.onnx", "voices.json")
except Exception as e:
    logger.error(f"Failed to initialize Kokoro model: {e}")
    raise

class ConversionStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    output_file: Optional[str] = None
    error: Optional[str] = None

# Store job statuses in memory (in production, use a proper database)
conversion_jobs = {}

def process_epub_sync(job_id: str, epub_path: str, voice: str, speed: float = 1.0, lang: str = "en-us"):
    """Synchronous processing function to run in thread pool"""
    try:
        # Update job status
        conversion_jobs[job_id].status = "processing"
        
        # Create job-specific directory for split output
        split_output = os.path.join(PROCESSING_DIR, job_id)
        os.makedirs(split_output, exist_ok=True)
        
        logger.info(f"Processing EPUB file for job {job_id}")
        
        # Definir el archivo de salida final
        output_file = os.path.join(OUTPUT_DIR, f"{job_id}.mp3")
        
        # Procesar el EPUB
        convert_text_to_audio(
            input_file=epub_path,
            output_file=None,  # No generamos el output aquí
            voice=voice,
            speed=speed,
            lang=lang,
            split_output=split_output,
            format="mp3",
            debug=True,
            interactive=False
        )
        
        # Merge chunks después de procesar
        logger.info(f"Merging chunks for job {job_id}")
        from kokoro_tts import merge_chunks_to_chapters
        merge_chunks_to_chapters(split_output, format="mp3")
        
        # Concatenar todos los capítulos en un solo archivo MP3
        from pydub import AudioSegment
        
        # Crear lista de archivos de capítulos ordenados
        chapter_files = sorted([
            os.path.join(split_output, f) 
            for f in os.listdir(split_output) 
            if f.startswith("chapter_") and f.endswith(".mp3")
        ])
        
        if chapter_files:
            logger.info(f"Found {len(chapter_files)} chapters to merge")
            
            # Combinar todos los capítulos
            combined = AudioSegment.empty()
            for chapter_file in chapter_files:
                audio = AudioSegment.from_mp3(chapter_file)
                combined += audio
                logger.info(f"Added chapter: {os.path.basename(chapter_file)}")
            
            # Exportar el archivo final
            combined.export(output_file, format="mp3")
            logger.info(f"Successfully exported merged file to {output_file}")
            
            # Update job status
            conversion_jobs[job_id].status = "completed"
            conversion_jobs[job_id].output_file = output_file
        else:
            raise FileNotFoundError("No chapter files found to merge")
        
        # Cleanup
        shutil.rmtree(split_output)
        logger.info(f"Cleaned up processing directory: {split_output}")
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")
        conversion_jobs[job_id].status = "failed"
        conversion_jobs[job_id].error = str(e)
        
    finally:
        # Remove uploaded file
        if os.path.exists(epub_path):
            os.remove(epub_path)
            logger.info(f"Removed uploaded file: {epub_path}")

async def process_epub(job_id: str, epub_path: str, voice: str, speed: float = 1.0, lang: str = "en-us"):
    """Asynchronous wrapper for processing function"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        thread_pool,
        process_epub_sync,
        job_id,
        epub_path,
        voice,
        speed,
        lang
    )

@app.post("/convert")
async def convert_epub(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    voice: str = Form(...),
    speed: float = Form(1.0),
    lang: str = Form("en-us")
):
    """Convert EPUB to audio using Kokoro TTS"""
    try:
        # Log received parameters
        logger.info(f"Received request - File: {file.filename}, Voice: {voice}, Speed: {speed}, Lang: {lang}")
        
        # Validate file type
        if not file.filename.endswith('.epub'):
            raise HTTPException(
                status_code=400,
                detail={"type": "value_error", "loc": ["file"], "msg": "Only EPUB files are supported"}
            )
        
        # Validate voice
        try:
            voice = validate_voice(voice, kokoro)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail={"type": "value_error", "loc": ["voice"], "msg": str(e)}
            )
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file
        epub_path = os.path.join(UPLOAD_DIR, f"{job_id}.epub")
        try:
            with open(epub_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={"type": "upload_error", "msg": f"Error saving uploaded file: {str(e)}"}
            )
        
        # Initialize job status
        conversion_jobs[job_id] = ConversionStatus(
            job_id=job_id,
            status="queued",
            progress=0.0
        )
        
        # Start background processing
        background_tasks.add_task(process_epub, job_id, epub_path, voice, speed, lang)
        
        return {
            "job_id": job_id, 
            "message": "Conversion started",
            "status": "queued"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting conversion: {e}")
        raise HTTPException(
            status_code=500,
            detail={"type": "server_error", "msg": "Internal server error"}
        )

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get the status of a conversion job"""
    if job_id not in conversion_jobs:
        raise HTTPException(404, "Job not found")
    return conversion_jobs[job_id]

@app.get("/voices")
async def list_voices():
    """Get list of available voices"""
    try:
        voices = list(kokoro.get_voices())
        return {"voices": voices}
    except Exception as e:
        logger.error(f"Error getting voices: {e}")
        raise HTTPException(500, "Internal server error")

@app.get("/languages")
async def list_languages():
    """Get list of supported languages"""
    try:
        languages = list(kokoro.get_languages())
        return {"languages": languages}
    except Exception as e:
        logger.error(f"Error getting languages: {e}")
        raise HTTPException(500, "Internal server error") 