# Base image with PyTorch + CUDA (pre-installed)
FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies for OpenCV, video processing, and TensorFlow
RUN apt-get update && apt-get install -y \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # Video processing
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    # Image processing
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    # HDF5 for TensorFlow
    libhdf5-dev \
    # Build tools (needed for some Python packages)
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# Note: torch is already in base image, but we install others
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLO model (optional, speeds up first run)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')" || true

# Create models directory
RUN mkdir -p /app/models

# Copy model files (you need to add these to the directory before building)
# These are copied if they exist, ignored if not
COPY models/ /app/models/ 2>/dev/null || true

# Copy ML application code
COPY app/ /app/app/

# Copy handler
COPY handler.py .

# Copy test script (optional, useful for debugging in container)
COPY test_handler.py . 2>/dev/null || true

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MODEL_PATH=/app/models
ENV CUDA_VISIBLE_DEVICES=0

# Healthcheck (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('CUDA:', torch.cuda.is_available())" || exit 1

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]

