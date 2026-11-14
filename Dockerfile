# Using the basic image of NVIDIA CUDA with Python 3 and Ubuntu 22.04
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set up a work directory.
WORKDIR /app

# Set up the necessary packages (including OpenCV relationships))
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

# Set up CLIP directly from GitHub Repository
RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git

# Copy and install Python Dependence
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install onnxruntime-gpu==1.19.0

# Copy the application reference code
COPY src /app/src
COPY main.py /app

# Create a CLIP Cache and Confiscation Directory
RUN mkdir -p ~/.cache/clip && \
    ln -s /app/src/config /app/config
