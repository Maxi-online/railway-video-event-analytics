# Используем базовый образ NVIDIA CUDA с Python 3 и Ubuntu 22.04
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем необходимые пакеты (включая зависимости для OpenCV)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libxrender1 \
        git \
        wget \
        unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Устанавливаем CLIP напрямую из GitHub репозитория
RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git

# Копируем и устанавливаем Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install onnxruntime-gpu==1.19.0

# Копируем исходный код приложения
COPY src /app/src
COPY main.py /app

# Создаем директорию для кеша CLIP и конфигов
RUN mkdir -p ~/.cache/clip && \
    ln -s /app/src/config /app/config
