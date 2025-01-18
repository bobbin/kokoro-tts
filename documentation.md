# Kokoro TTS API Documentation

## Overview
Kokoro TTS API es un servicio REST que permite convertir libros EPUB a audio utilizando el motor de síntesis de voz Kokoro TTS. El servicio procesa los archivos de forma asíncrona y proporciona endpoints para monitorear el progreso de la conversión.

## Requisitos

### Sistema
- Python 3.x
- ffmpeg (para procesamiento de audio)
- Suficiente espacio en disco para archivos temporales y de salida

### Dependencias Python
```bash
pip install fastapi
pip install uvicorn
pip install python-multipart
pip install soundfile
pip install sounddevice
pip install kokoro_onnx
pip install ebooklib
pip install beautifulsoup4
pip install pydub
```

### Archivos del Modelo
Descargar los archivos del modelo necesarios:
```bash
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx
```

## Estructura de Directorios
```
/
├── api.py              # Código principal del API
├── kokoro_tts/        # Módulo de Kokoro TTS
├── uploads/           # Directorio temporal para archivos EPUB
├── processing/        # Directorio temporal para procesamiento
├── outputs/           # Directorio para archivos MP3 finales
├── voices.json        # Configuración de voces
└── kokoro-v0_19.onnx # Modelo de Kokoro
```

## Despliegue

### Desarrollo Local
1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Iniciar el servidor:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Producción
Para despliegue en producción, se recomienda usar Gunicorn con Uvicorn workers:

1. Instalar Gunicorn:
```bash
pip install gunicorn
```

2. Iniciar el servidor:
```bash
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

## API Endpoints

### POST /convert
Convierte un archivo EPUB a audio.

**Request:**
- Content-Type: multipart/form-data
- Body:
  - `file`: Archivo EPUB (required)
  - `voice`: ID de la voz a usar (required)
  - `speed`: Velocidad de habla (default: 1.0)
  - `lang`: Código de idioma (default: "en-us")

**Response:**
```json
{
    "job_id": "uuid",
    "message": "Conversion started",
    "status": "queued"
}
```

### GET /status/{job_id}
Obtiene el estado de un trabajo de conversión.

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
Lista las voces disponibles.

**Response:**
```json
{
    "voices": ["voice1", "voice2", ...]
}
```

### GET /languages
Lista los idiomas soportados.

**Response:**
```json
{
    "languages": ["en-us", "es-es", ...]
}
```

## Ejemplos de Uso

### Python
```python
import requests

# Convertir EPUB a audio
files = {'file': open('book.epub', 'rb')}
data = {'voice': 'af_sarah', 'speed': 1.0, 'lang': 'en-us'}
response = requests.post('http://localhost:8000/convert', files=files, data=data)
job_id = response.json()['job_id']

# Verificar estado
status = requests.get(f'http://localhost:8000/status/{job_id}')
print(status.json())
```

### cURL
```bash
# Convertir EPUB a audio
curl -X POST "http://localhost:8000/convert" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@book.epub" \
     -F "voice=af_sarah" \
     -F "speed=1.0" \
     -F "lang=en-us"

# Verificar estado
curl "http://localhost:8000/status/job-id-here"
```

## Manejo de Errores

El API usa códigos de estado HTTP estándar:
- 200: Éxito
- 400: Error en la petición (parámetros inválidos)
- 404: No encontrado (job_id no existe)
- 500: Error interno del servidor

Los errores incluyen detalles en el cuerpo de la respuesta:
```json
{
    "detail": {
        "type": "value_error",
        "loc": ["voice"],
        "msg": "Invalid voice selection"
    }
}
```

## Consideraciones de Producción

### Almacenamiento
- Monitorear el espacio en disco
- Implementar política de limpieza para archivos antiguos
- Considerar almacenamiento externo para archivos de salida

### Escalabilidad
- Ajustar `max_workers` en ThreadPoolExecutor según recursos
- Monitorear uso de memoria y CPU
- Considerar sistema de colas (como Celery) para alta carga

### Seguridad
- Implementar autenticación
- Limitar tamaño máximo de archivos
- Validar tipos de archivo
- Implementar rate limiting

### Monitoreo
- Logging de errores
- Métricas de uso
- Tiempo de procesamiento
- Estado del sistema

## Troubleshooting

### Problemas Comunes

1. Error "No file uploaded":
   - Verificar que el archivo se envía como multipart/form-data
   - Verificar que el campo se llama "file"

2. Error "Invalid voice":
   - Usar endpoint /voices para ver voces disponibles
   - Verificar que el ID de voz es correcto

3. Error en procesamiento:
   - Verificar logs del servidor
   - Comprobar permisos de directorios
   - Verificar espacio en disco

### Logs

Los logs se encuentran en la salida estándar y contienen información sobre:
- Inicio de trabajos
- Progreso de procesamiento
- Errores detallados
- Finalización de trabajos

## Mantenimiento

### Limpieza de Archivos
```python
# Ejemplo de script de limpieza
import os
import time

def cleanup_old_files(directory, max_age_days=7):
    now = time.time()
    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        if os.stat(path).st_mtime < now - max_age_days * 86400:
            os.remove(path)
```

### Actualización del Modelo
1. Descargar nuevos archivos del modelo
2. Reemplazar archivos existentes
3. Reiniciar el servidor

## Soporte

Para reportar problemas o solicitar ayuda:
1. Verificar la documentación
2. Revisar los logs del servidor
3. Proporcionar detalles del error
4. Incluir ID del trabajo si aplica
