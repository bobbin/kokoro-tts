FROM ubuntu:22.04

# Evitar interacciones durante la instalación de paquetes
ENV DEBIAN_FRONTEND=noninteractive

# Instalar las herramientas básicas y CUDA
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Añadir repositorio NVIDIA CUDA
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb

# Añadir repositorio para Python 3.11
RUN add-apt-repository ppa:deadsnakes/ppa

# Instalar CUDA, Python 3.11 y dependencias del sistema
RUN apt-get update && apt-get install -y \
    cuda-runtime-12-3 \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Instalar pip para Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Establecer Python 3.11 como predeterminado
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11

# Directorio de trabajo
WORKDIR /app

# Crear directorios necesarios y establecer permisos
RUN mkdir -p /app/uploads /app/outputs /app/processing /app/models && \
    chmod 777 /app/uploads /app/outputs /app/processing /app/models

# Copiar archivos necesarios
COPY requirements.txt .
COPY api.py .

# Instalar dependencias de Python
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Variables de entorno para CUDA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV ONNX_PROVIDER=CUDAExecutionProvider
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Puerto para FastAPI
EXPOSE 8080

# Comando para iniciar la API usando Python 3.11
CMD ["python3.11", "-m", "uvicorn", "api:handler", "--host", "0.0.0.0", "--port", "8080"]