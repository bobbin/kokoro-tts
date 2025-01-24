#!/usr/bin/env python3

# Standard library imports
import os
import sys
import itertools
import threading
import time
import signal
import difflib
import warnings
from threading import Event

# Third-party imports
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
import soundfile as sf
import sounddevice as sd
import numpy as np
from gradio_client import Client

warnings.filterwarnings("ignore", category=UserWarning, module='ebooklib')
warnings.filterwarnings("ignore", category=FutureWarning, module='ebooklib')

# Global flag to stop the spinner and audio
stop_spinner = False
stop_audio = False

# Initialize Hugging Face client
client = Client("bobbin28/Kokoro-TTS-Zero", hf_token=os.environ.get("HUGGING_FACE_TOKEN"))

def spinning_wheel(message="Processing...", progress=None):
    """Display a spinning wheel with a message."""
    spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
    while not stop_spinner:
        spin = next(spinner)
        if progress is not None:
            sys.stdout.write(f"\r{message} {progress} {spin}")
        else:
            sys.stdout.write(f"\r{message} {spin}")
        sys.stdout.flush()
        time.sleep(0.1)
    # Clear the spinner line when done
    sys.stdout.write('\r' + ' ' * (len(message) + 50) + '\r')
    sys.stdout.flush()

def list_available_voices():
    """Get available voices from the API."""
    try:
        response = client.predict(api_name="/initialize_model")
        # La respuesta viene como un diccionario con una lista de pares [id, nombre]
        voices = [voice[0] for voice in response['choices']]
        print("Available voices:")
        for idx, voice in enumerate(voices):
            print(f"{idx + 1}. {voice}")
        return voices
    except Exception as e:
        print(f"Error getting available voices: {e}")
        return []

def extract_text_from_epub(epub_file):
    book = epub.read_epub(epub_file)
    full_text = ""
    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_body_content(), "html.parser")
            full_text += soup.get_text()
    return full_text

def chunk_text(text, chunk_size=2000):
    """Split text into chunks at sentence boundaries."""
    sentences = text.replace('\n', ' ').split('.')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        if not sentence.strip():
            continue  # Skip empty sentences
        
        sentence = sentence.strip() + '.'
        sentence_size = len(sentence)
        
        # Start new chunk if current one would be too large
        if current_size + sentence_size > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
        
        current_chunk.append(sentence)
        current_size += sentence_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def validate_language(lang):
    """Validate if the language is supported."""
    # Since we don't have direct access to supported languages through the API,
    # we'll maintain a list of known supported languages
    supported_languages = {"en-us", "es-es", "fr-fr", "de-de", "it-it", "pt-br", "ja-jp"}
    if lang not in supported_languages:
        supported_langs = ', '.join(sorted(supported_languages))
        raise ValueError(f"Unsupported language: {lang}\nSupported languages are: {supported_langs}")
    return lang

def print_usage():
    print("""
Usage: kokoro-tts <input_text_file> [<output_audio_file>] [options]

Commands:
    -h, --help         Show this help message
    --help-languages   List all supported languages
    --help-voices      List all available voices
    --merge-chunks     Merge existing chunks in split-output directory into chapter files

Options:
    --stream            Stream audio instead of saving to file
    --speed <float>     Set speech speed (default: 1.0)
    --lang <str>        Set language (default: en-us)
    --voice <str>       Set voice (default: interactive selection)
    --split-output <dir> Save each chunk as separate file in directory
    --format <str>      Audio format: wav or mp3 (default: wav)
    --debug             Show detailed debug information

Input formats:
    .txt               Text file input
    .epub              EPUB book input (will process chapters)

Examples:
    kokoro-tts input.txt output.wav --speed 1.2 --lang en-us --voice af_sarah
    kokoro-tts input.epub --split-output ./chunks/ --format mp3
    kokoro-tts input.txt --stream --speed 0.8
    kokoro-tts --merge-chunks --split-output ./chunks/ --format wav
    kokoro-tts --help-voices
    kokoro-tts --help-languages
    kokoro-tts input.epub --split-output ./chunks/ --debug
    """)

def print_supported_languages():
    """Print all supported languages."""
    try:
        # Since we don't have direct access to supported languages through the API,
        # we'll maintain a list of known supported languages
        languages = ["en-us", "es-es", "fr-fr", "de-de", "it-it", "pt-br", "ja-jp"]
        print("\nSupported languages:")
        for lang in sorted(languages):
            print(f"    {lang}")
        print()
    except Exception as e:
        raise RuntimeError(f"Error getting supported languages: {str(e)}")

def print_supported_voices():
    """Print all supported voices."""
    try:
        response = client.predict(api_name="/initialize_model")
        voices = [voice[0] for voice in response['choices']]
        print("\nSupported voices:")
        for idx, voice in enumerate(voices):
            print(f"    {idx + 1}. {voice}")
        print()
    except Exception as e:
        raise RuntimeError(f"Error getting supported voices: {str(e)}")

def validate_voice(voice):
    """Validate if the voice is supported."""
    try:
        response = client.predict(api_name="/initialize_model")
        supported_voices = {voice[0] for voice in response['choices']}
        
        # Check if it's a blend request (comma separated voices)
        if ',' in voice:
            voices = [v.strip() for v in voice.split(',')]
            if len(voices) != 2:
                raise ValueError("Voice blending requires exactly 2 voices separated by comma")
                
            # Validate each voice
            for v in voices:
                if v not in supported_voices:
                    supported_voices_list = ', '.join(sorted(supported_voices))
                    raise ValueError(f"Unsupported voice: {v}\nSupported voices are: {supported_voices_list}")
            
            # Voice blending is not supported in the API, so we'll use the first voice
            return voices[0]
            
        # Single voice validation
        if voice not in supported_voices:
            supported_voices_list = ', '.join(sorted(supported_voices))
            raise ValueError(f"Unsupported voice: {voice}\nSupported voices are: {supported_voices_list}")
        return voice
    except Exception as e:
        raise ValueError(f"Error validating voice: {str(e)}")

def process_chunk_sequential(chunk: str, voice: str, speed: float, lang: str) -> tuple[list[float] | None, int | None]:
    """Process a single chunk of text sequentially."""
    try:
        # Call the API to generate speech
        result = client.predict(
            text=chunk,
            voice_names=[voice],
            speed=speed,
            api_name="/generate_speech_from_ui"
        )
        
        # The API returns a tuple with the audio file path, metrics plot, and performance summary
        audio_path = result[0]
        
        # Read the generated audio file
        samples, sample_rate = sf.read(audio_path)
        return samples, sample_rate
    except Exception as e:
        raise RuntimeError(f"Error processing chunk: {str(e)}")

def convert_text_to_audio(input_file, output_file=None, voice=None, speed=1.0, lang="en-us", 
                         stream=False, split_output=None, format="mp3", debug=False, 
                         interactive=True, progress_callback=None):
    global stop_spinner
    
    try:
        # Validate language
        lang = validate_language(lang)
        
        # Handle voice selection
        if voice:
            voice = validate_voice(voice)
        else:
            # Interactive voice selection solo si interactive=True
            if interactive:
                voices = list_available_voices()
                print("\nTip: You can blend two voices by entering two numbers separated by comma (e.g., '7,11')")
                try:
                    voice_input = input("Choose voice(s) by number: ")
                    if ',' in voice_input:
                        v1, v2 = map(lambda x: int(x.strip()) - 1, voice_input.split(','))
                        if not (0 <= v1 < len(voices) and 0 <= v2 < len(voices)):
                            raise ValueError("Invalid voice numbers")
                        voice = f"{voices[v1]},{voices[v2]}"
                    else:
                        voice_choice = int(voice_input) - 1
                        if not (0 <= voice_choice < len(voices)):
                            raise ValueError("Invalid choice")
                        voice = voices[voice_choice]
                    voice = validate_voice(voice)
                except (ValueError, IndexError):
                    print("Invalid choice. Using default voice.")
                    voice = "af_sarah"  # default voice
            else:
                voice = "af_sarah"  # default voice cuando no es interactivo
        
        # Read the input file (handle .txt or .epub)
        if input_file.endswith('.epub'):
            chapters = extract_chapters_from_epub(input_file, debug)
            if not chapters:
                print("No chapters found in EPUB file.")
                sys.exit(1)
            
            # Print summary before starting
            total_words = sum(len(chapter['content'].split()) for chapter in chapters)
            print("\nBook Summary:")
            print(f"Total Chapters: {len(chapters)}")
            print(f"Total Words: {total_words:,}")
            print(f"Total Duration: {total_words / 150:.1f} minutes")
            
            if debug:
                print("\nDetailed Chapter List:")
                for chapter in chapters:
                    word_count = len(chapter['content'].split())
                    print(f"  • {chapter['title']}")
                    print(f"    Words: {word_count:,}")
                    print(f"    Duration: {word_count / 150:.1f} minutes")
            
            # Solo pedimos confirmación si interactive=True
            if interactive:
                print("\nPress Enter to start processing, or Ctrl+C to cancel...")
                input()
            
            if split_output:
                os.makedirs(split_output, exist_ok=True)
                
                # First create all chapter directories and info files
                print("\nCreating chapter directories and info files...")
                total_chapters = len(chapters)
                for chapter_num, chapter in enumerate(chapters, 1):
                    chapter_dir = os.path.join(split_output, f"chapter_{chapter_num:03d}")
                    os.makedirs(chapter_dir, exist_ok=True)
                    
                    # Write chapter info with more details
                    info_file = os.path.join(chapter_dir, "info.txt")
                    with open(info_file, "w", encoding="utf-8") as f:
                        f.write(f"Title: {chapter['title']}\n")
                        f.write(f"Order: {chapter['order']}\n")
                        f.write(f"Words: {len(chapter['content'].split())}\n")
                        f.write(f"Estimated Duration: {len(chapter['content'].split()) / 150:.1f} minutes\n")
                
                print("Created chapter directories and info files")
                
                # Continue with existing processing code...
                
                # Llamar al callback de progreso si existe
                if progress_callback:
                    progress_callback(chapter_num, total_chapters)
            
        else:
            with open(input_file, 'r', encoding='utf-8') as file:
                text = file.read()
            # Treat single text file as one chapter
            chapters = [{'title': 'Chapter 1', 'content': text}]

        if stream:
            # Streaming is not supported with the API
            print("Streaming is not supported when using the API")
            sys.exit(1)
        else:
            if split_output:
                os.makedirs(split_output, exist_ok=True)
                
                for chapter_num, chapter in enumerate(chapters, 1):
                    chapter_dir = os.path.join(split_output, f"chapter_{chapter_num:03d}")
                    
                    # Skip if chapter is already fully processed
                    if os.path.exists(chapter_dir):
                        info_file = os.path.join(chapter_dir, "info.txt")
                        if os.path.exists(info_file):
                            chunks = chunk_text(chapter['content'], chunk_size=50000)
                            total_chunks = len(chunks)
                            existing_chunks = len([f for f in os.listdir(chapter_dir) 
                                                if f.startswith("chunk_") and f.endswith(f".{format}")])
                            
                            if existing_chunks == total_chunks:
                                print(f"\nSkipping {chapter['title']}: Already completed ({existing_chunks} chunks)")
                                continue
                            else:
                                print(f"\nResuming {chapter['title']}: Found {existing_chunks}/{total_chunks} chunks")

                    print(f"\nProcessing: {chapter['title']}")
                    os.makedirs(chapter_dir, exist_ok=True)
                    
                    # Write chapter info if not exists
                    info_file = os.path.join(chapter_dir, "info.txt")
                    if not os.path.exists(info_file):
                        with open(info_file, "w", encoding="utf-8") as f:
                            f.write(f"Title: {chapter['title']}\n")
                    
                    chunks = chunk_text(chapter['content'], chunk_size=50000)
                    total_chunks = len(chunks)
                    processed_chunks = len([f for f in os.listdir(chapter_dir) 
                                         if f.startswith("chunk_") and f.endswith(f".{format}")])
                    
                    for chunk_num, chunk in enumerate(chunks, 1):
                        if stop_audio:  # Check for interruption
                            break
                        
                        # Skip if chunk file already exists (regardless of position)
                        chunk_file = os.path.join(chapter_dir, f"chunk_{chunk_num:03d}.{format}")
                        if os.path.exists(chunk_file):
                            continue  # Don't increment processed_chunks here since we counted them above
                        
                        # Create progress bar
                        filled = "■" * processed_chunks
                        remaining = "□" * (total_chunks - processed_chunks)
                        progress_bar = f"[{filled}{remaining}] ({processed_chunks}/{total_chunks})"
                        
                        stop_spinner = False
                        spinner_thread = threading.Thread(
                            target=spinning_wheel,
                            args=(f"Processing {chapter['title']}", progress_bar)
                        )
                        spinner_thread.start()
                        
                        try:
                            samples, sample_rate = process_chunk_sequential(chunk, voice, speed, lang)
                            if samples is not None:
                                sf.write(chunk_file, samples, sample_rate)
                                processed_chunks += 1
                        except Exception as e:
                            print(f"\nError processing chunk {chunk_num}: {e}")
                        
                        stop_spinner = True
                        spinner_thread.join()
                        
                        if stop_audio:  # Check for interruption
                            break
                    
                    print(f"\nCompleted {chapter['title']}: {processed_chunks}/{total_chunks} chunks processed")
                    
                    if stop_audio:  # Check for interruption
                        break
                
                print(f"\nCreated audio files for {len(chapters)} chapters in {split_output}/")
            else:
                # Combine all chapters into one file
                all_samples = []
                sample_rate = None
                
                for chapter_num, chapter in enumerate(chapters, 1):
                    print(f"\nProcessing: {chapter['title']}")
                    chunks = chunk_text(chapter['content'], chunk_size=50000)
                    processed_chunks = 0
                    total_chunks = len(chunks)
                    
                    for chunk_num, chunk in enumerate(chunks, 1):
                        if stop_audio:  # Check for interruption
                            break
                        
                        stop_spinner = False
                        spinner_thread = threading.Thread(
                            target=spinning_wheel,
                            args=(f"Processing chunk {chunk_num}/{total_chunks}",)
                        )
                        spinner_thread.start()
                        
                        try:
                            samples, sr = process_chunk_sequential(chunk, voice, speed, lang)
                            if samples is not None:
                                if sample_rate is None:
                                    sample_rate = sr
                                all_samples.extend(samples)
                                processed_chunks += 1
                        except Exception as e:
                            print(f"\nError processing chunk {chunk_num}: {e}")
                        
                        stop_spinner = True
                        spinner_thread.join()
                    
                    print(f"\nCompleted {chapter['title']}: {processed_chunks}/{total_chunks} chunks processed")
                
                if all_samples:
                    print("\nSaving complete audio file...")
                    if not output_file:
                        output_file = f"{os.path.splitext(input_file)[0]}.{format}"
                    sf.write(output_file, all_samples, sample_rate)
                    print(f"Created {output_file}")
    except Exception as e:
        raise RuntimeError(f"Error converting text to audio: {str(e)}")

def handle_ctrl_c(signum, frame):
    global stop_spinner, stop_audio
    print("\nCtrl+C detected, stopping...")
    stop_spinner = True
    stop_audio = True
    sys.exit(0)

# Register the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, handle_ctrl_c)

def merge_chunks_to_chapters(split_output_dir, format="mp3"):
    """Merge audio chunks into complete chapter files."""
    global stop_spinner

    if not os.path.exists(split_output_dir):
        print(f"Error: Directory {split_output_dir} does not exist.")
        return

    # Find all chapter directories
    chapter_dirs = sorted([d for d in os.listdir(split_output_dir) 
                          if d.startswith("chapter_") and os.path.isdir(os.path.join(split_output_dir, d))])
    
    if not chapter_dirs:
        print(f"No chapter directories found in {split_output_dir}")
        return

    for chapter_dir in chapter_dirs:
        chapter_path = os.path.join(split_output_dir, chapter_dir)
        chunk_files = sorted([f for f in os.listdir(chapter_path) 
                            if f.startswith("chunk_") and f.endswith(f".{format}")])
        
        if not chunk_files:
            print(f"No chunks found in {chapter_dir}")
            continue

        # Read chapter title from info.txt if available
        chapter_title = chapter_dir
        info_file = os.path.join(chapter_path, "info.txt")
        if os.path.exists(info_file):
            with open(info_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith("Title:"):
                        chapter_title = line.replace("Title:", "").strip()
                        break

        print(f"\nMerging chunks for {chapter_title}")
        
        # Initialize variables for merging
        all_samples = []
        sample_rate = None
        total_duration = 0
        
        # Create progress spinner
        total_chunks = len(chunk_files)
        processed_chunks = 0
        
        for chunk_file in chunk_files:
            chunk_path = os.path.join(chapter_path, chunk_file)
            
            # Display progress
            print(f"\rProcessing chunk {processed_chunks + 1}/{total_chunks}", end="")
            
            try:
                # Read audio data
                data, sr = sf.read(chunk_path)
                
                # Verify the audio data
                if len(data) == 0:
                    print(f"\nWarning: Empty audio data in {chunk_file}")
                    continue
                
                # Initialize sample rate or verify it matches
                if sample_rate is None:
                    sample_rate = sr
                elif sr != sample_rate:
                    print(f"\nWarning: Sample rate mismatch in {chunk_file}")
                    continue
                
                # Add chunk duration to total
                chunk_duration = len(data) / sr
                total_duration += chunk_duration
                
                # Append the audio data
                all_samples.extend(data)
                processed_chunks += 1
                
            except Exception as e:
                print(f"\nError processing {chunk_file}: {e}")
        
        print()  # New line after progress
        
        if all_samples:
            # Create merged file name
            merged_file = os.path.join(split_output_dir, f"{chapter_dir}.{format}")
            print(f"Saving merged chapter to {merged_file}")
            print(f"Total duration: {total_duration:.2f} seconds")
            
            try:
                # Ensure all_samples is a numpy array
                all_samples = np.array(all_samples)
                
                # Save merged audio
                sf.write(merged_file, all_samples, sample_rate)
                print(f"Successfully merged {processed_chunks}/{total_chunks} chunks")
                
                # Verify the output file
                if os.path.exists(merged_file):
                    output_data, output_sr = sf.read(merged_file)
                    output_duration = len(output_data) / output_sr
                    print(f"Verified output file: {output_duration:.2f} seconds")
                else:
                    print("Warning: Output file was not created")
                
            except Exception as e:
                print(f"Error saving merged file: {e}")
        else:
            print("No valid audio data to merge")

def get_valid_options():
    """Return a set of valid command line options"""
    return {
        '-h', '--help',
        '--help-languages',
        '--help-voices',
        '--merge-chunks',
        '--stream',
        '--speed',
        '--lang',
        '--voice',
        '--split-output',
        '--format',
        '--debug'  # Add debug option
    }

def extract_chapters_from_epub(epub_file, debug=False):
    """Extract chapters from epub file using ebooklib's metadata and TOC."""
    if not os.path.exists(epub_file):
        raise FileNotFoundError(f"EPUB file not found: {epub_file}")
    
    book = epub.read_epub(epub_file)
    chapters = []
    
    if debug:
        print("\nBook Metadata:")
        for key, value in book.metadata.items():
            print(f"  {key}: {value}")
        
        print("\nTable of Contents:")
        def print_toc(items, depth=0):
            for item in items:
                indent = "  " * depth
                if isinstance(item, tuple):
                    section_title, section_items = item
                    print(f"{indent}• Section: {section_title}")
                    print_toc(section_items, depth + 1)
                elif isinstance(item, epub.Link):
                    print(f"{indent}• {item.title} -> {item.href}")
        print_toc(book.toc)
    
    def get_chapter_content(soup, start_id, next_id=None):
        """Extract content between two fragment IDs"""
        content = []
        start_elem = soup.find(id=start_id)
        
        if not start_elem:
            return ""
        
        # Skip the heading itself if it's a heading
        if start_elem.name in ['h1', 'h2', 'h3', 'h4']:
            current = start_elem.find_next_sibling()
        else:
            current = start_elem
            
        while current:
            # Stop if we hit the next chapter
            if next_id and current.get('id') == next_id:
                break
            # Stop if we hit another chapter heading
            if current.name in ['h1', 'h2', 'h3'] and 'chapter' in current.get_text().lower():
                break
            content.append(current.get_text())
            current = current.find_next_sibling()
            
        return '\n'.join(content).strip()
    
    def process_toc_items(items, depth=0):
        processed = []
        for i, item in enumerate(items):
            if isinstance(item, tuple):
                section_title, section_items = item
                if debug:
                    print(f"{'  ' * depth}Processing section: {section_title}")
                processed.extend(process_toc_items(section_items, depth + 1))
            elif isinstance(item, epub.Link):
                if debug:
                    print(f"{'  ' * depth}Processing link: {item.title} -> {item.href}")
                
                # Skip if title suggests it's front matter
                if (item.title.lower() in ['copy', 'copyright', 'title page', 'cover'] or
                    item.title.lower().startswith('by')):
                    continue
                
                # Extract the file name and fragment from href
                href_parts = item.href.split('#')
                file_name = href_parts[0]
                fragment_id = href_parts[1] if len(href_parts) > 1 else None
                
                # Find the document
                doc = next((doc for doc in book.get_items_of_type(ITEM_DOCUMENT) 
                          if doc.file_name.endswith(file_name)), None)
                
                if doc:
                    content = doc.get_content().decode('utf-8')
                    soup = BeautifulSoup(content, "html.parser")
                    
                    # If no fragment ID, get whole document content
                    if not fragment_id:
                        text_content = soup.get_text().strip()
                    else:
                        # Get the next fragment ID if available
                        next_item = items[i + 1] if i + 1 < len(items) else None
                        next_fragment = None
                        if isinstance(next_item, epub.Link):
                            next_href_parts = next_item.href.split('#')
                            if next_href_parts[0] == file_name and len(next_href_parts) > 1:
                                next_fragment = next_href_parts[1]
                        
                        # Extract content between fragments
                        text_content = get_chapter_content(soup, fragment_id, next_fragment)
                    
                    if text_content:
                        chapters.append({
                            'title': item.title,
                            'content': text_content,
                            'order': len(processed) + 1
                        })
                        processed.append(item)
                        if debug:
                            print(f"{'  ' * depth}Added chapter: {item.title}")
                            print(f"{'  ' * depth}Content length: {len(text_content)} chars")
                            print(f"{'  ' * depth}Word count: {len(text_content.split())}")
        return processed
    
    # Process the table of contents
    process_toc_items(book.toc)
    
    # If no chapters were found through TOC, try processing all documents
    if not chapters:
        if debug:
            print("\nNo chapters found in TOC, processing all documents...")
        
        # Get all document items sorted by file name
        docs = sorted(
            book.get_items_of_type(ITEM_DOCUMENT),
            key=lambda x: x.file_name
        )
        
        for doc in docs:
            if debug:
                print(f"Processing document: {doc.file_name}")
            
            content = doc.get_content().decode('utf-8')
            soup = BeautifulSoup(content, "html.parser")
            
            # Try to find chapter divisions
            chapter_divs = soup.find_all(['h1', 'h2', 'h3'], class_=lambda x: x and 'chapter' in x.lower())
            if not chapter_divs:
                chapter_divs = soup.find_all(lambda tag: tag.name in ['h1', 'h2', 'h3'] and 
                                          ('chapter' in tag.get_text().lower() or
                                           'book' in tag.get_text().lower()))
            
            if chapter_divs:
                # Process each chapter division
                for i, div in enumerate(chapter_divs):
                    title = div.get_text().strip()
                    
                    # Get content until next chapter heading or end
                    content = ''
                    for tag in div.find_next_siblings():
                        if tag.name in ['h1', 'h2', 'h3'] and (
                            'chapter' in tag.get_text().lower() or
                            'book' in tag.get_text().lower()):
                            break
                        content += tag.get_text() + '\n'
                    
                    if content.strip():
                        chapters.append({
                            'title': title,
                            'content': content.strip(),
                            'order': len(chapters) + 1
                        })
                        if debug:
                            print(f"Added chapter: {title}")
            else:
                # No chapter divisions found, treat whole document as one chapter
                text_content = soup.get_text().strip()
                if text_content:
                    # Try to find a title
                    title_tag = soup.find(['h1', 'h2', 'title'])
                    title = title_tag.get_text().strip() if title_tag else f"Chapter {len(chapters) + 1}"
                    
                    if title.lower() not in ['copy', 'copyright', 'title page', 'cover']:
                        chapters.append({
                            'title': title,
                            'content': text_content,
                            'order': len(chapters) + 1
                        })
                        if debug:
                            print(f"Added chapter: {title}")
    
    # Print summary
    if chapters:
        print("\nSuccessfully extracted {} chapters:".format(len(chapters)))
        for chapter in chapters:
            print(f"  {chapter['order']}. {chapter['title']}")
        
        total_words = sum(len(chapter['content'].split()) for chapter in chapters)
        print("\nBook Summary:")
        print(f"Total Chapters: {len(chapters)}")
        print(f"Total Words: {total_words:,}")
        print(f"Total Duration: {total_words / 150:.1f} minutes")
        
        if debug:
            print("\nDetailed Chapter List:")
            for chapter in chapters:
                word_count = len(chapter['content'].split())
                print(f"  • {chapter['title']}")
                print(f"    Words: {word_count:,}")
                print(f"    Duration: {word_count / 150:.1f} minutes")
    else:
        print("\nWarning: No chapters were extracted!")
        if debug:
            print("\nAvailable documents:")
            for doc in book.get_items_of_type(ITEM_DOCUMENT):
                print(f"  • {doc.file_name}")
    
    return chapters

if __name__ == "__main__":
    # Validate command line options first
    valid_options = get_valid_options()
    unknown_options = []
    
    # Check for unknown options
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith('--') or arg.startswith('-'):
            # Check if it's a valid option
            if arg not in valid_options:
                unknown_options.append(arg)
            # Skip the next argument if it's a value for an option that takes parameters
            elif arg in {'--speed', '--lang', '--voice', '--split-output', '--format'}:
                i += 1
        i += 1
    
    # If unknown options were found, show error and help
    if unknown_options:
        print("Error: Unknown option(s):", ", ".join(unknown_options))
        print("\nDid you mean one of these?")
        for unknown in unknown_options:
            # Find similar valid options using string similarity
            similar = difflib.get_close_matches(unknown, valid_options, n=3, cutoff=0.4)
            if similar:
                print(f"  {unknown} -> {', '.join(similar)}")
        print("\n")  # Add extra newline for spacing
        print_usage()  # Show the full help text
        sys.exit(1)
    
    # Handle help commands first
    if len(sys.argv) == 2:
        if sys.argv[1] in ['-h', '--help']:
            print_usage()
            sys.exit(0)
        elif sys.argv[1] == '--help-languages':
            print_supported_languages()
            sys.exit(0)
        elif sys.argv[1] == '--help-voices':
            print_supported_voices()
            sys.exit(0)
    
    # Parse arguments
    input_file = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    else:
        output_file = None

    stream = '--stream' in sys.argv
    speed = 1.0  # default speed
    lang = "en-us"  # default language
    voice = None  # default to interactive selection
    split_output = None
    format = "mp3"  # default format
    merge_chunks = '--merge-chunks' in sys.argv
    
    # Parse optional arguments
    for i, arg in enumerate(sys.argv):
        if arg == '--speed' and i + 1 < len(sys.argv):
            try:
                speed = float(sys.argv[i + 1])
            except ValueError:
                print("Error: Speed must be a number")
                sys.exit(1)
        elif arg == '--lang' and i + 1 < len(sys.argv):
            lang = sys.argv[i + 1]
        elif arg == '--voice' and i + 1 < len(sys.argv):
            voice = sys.argv[i + 1]
        elif arg == '--split-output' and i + 1 < len(sys.argv):
            split_output = sys.argv[i + 1]
        elif arg == '--format' and i + 1 < len(sys.argv):
            format = sys.argv[i + 1].lower()
            if format not in ['wav', 'mp3']:
                print("Error: Format must be either 'wav' or 'mp3'")
                sys.exit(1)
    
    # Handle merge chunks operation
    if merge_chunks:
        if not split_output:
            print("Error: --split-output directory must be specified when using --merge-chunks")
            sys.exit(1)
        merge_chunks_to_chapters(split_output, format)
        sys.exit(0)
    
    # Normal processing mode
    if not input_file:
        print("Error: Input file required for text-to-speech conversion")
        print_usage()
        sys.exit(1)

    # Ensure the input file exists
    if not os.path.isfile(input_file):
        print(f"Error: The file {input_file} does not exist.")
        sys.exit(1)
    
    # Ensure the output file has a proper extension if specified
    if output_file and not output_file.endswith(('.' + format)):
        print(f"Error: Output file must have .{format} extension.")
        sys.exit(1)
    
    # Add debug flag
    debug = '--debug' in sys.argv
    
    # Convert text to audio with debug flag
    convert_text_to_audio(input_file, output_file, voice=voice, stream=stream, 
                         speed=speed, lang=lang, split_output=split_output, 
                         format=format, debug=debug)

