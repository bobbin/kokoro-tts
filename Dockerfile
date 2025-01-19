FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copiar archivos necesarios
COPY requirements.txt .
COPY api.py .
COPY kokoro-v0_19.onnx .
COPY voices.json .

# Instalar dependencias de Python
RUN pip3 install --no-cache-dir -r requirements.txt

# Puerto para FastAPI
EXPOSE 8080

# Comando para iniciar la API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]