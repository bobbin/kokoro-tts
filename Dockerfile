FROM docker.io/nvidia/cuda:12.3.1-runtime-ubuntu20.04

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libportaudio2 \
    libasound-dev \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app
# Crear directorios necesarios y establecer permisos
RUN mkdir -p /app/uploads /app/outputs /app/processing && \
    chmod 777 /app/uploads /app/outputs /app/processing

# Copiar archivos necesarios
COPY requirements.txt .
COPY api.py .

# Instalar dependencias de Python
RUN pip3 install --no-cache-dir -r requirements.txt
# Variables de entorno para CUDA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV ONNX_PROVIDER=CUDAExecutionProvider
# Puerto para FastAPI
EXPOSE 8080

# Comando para iniciar la API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]