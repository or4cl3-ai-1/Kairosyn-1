# KAIROSYN-1 Docker Image
# ========================
# Multi-stage build for production deployment

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
    python3.11-dev \
    git \
    wget \
    libsndfile1 \
    ffmpeg \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.3.0 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY kairosyn/ ./kairosyn/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY setup.py .
RUN pip install --no-cache-dir -e .

# Create directories
RUN mkdir -p /app/checkpoints /app/results

# Environment variables
ENV PYTHONPATH=/app
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV TOKENIZERS_PARALLELISM=false

# Default port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost:8080/health || exit 1

# Default command: serve
CMD ["uvicorn", "kairosyn.api.server:app", "--host", "0.0.0.0", "--port", "8080"]
