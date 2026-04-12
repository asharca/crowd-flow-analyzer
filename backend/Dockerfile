# CUDA 12.8 runtime — must match torch+cu128 exactly to avoid cuBLAS version conflicts.
# Driver 575 supports up to CUDA 12.9, so 12.8 runtime runs fine.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# Switch apt sources to Aliyun mirror
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (includes Python management)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_PYTHON=3.11
ENV UV_COMPILE_BYTECODE=0
ENV UV_LINK_MODE=copy
ENV UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# HuggingFace mirror for faster model downloads in China
ENV HF_ENDPOINT=https://hf-mirror.com

WORKDIR /app

COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project

# Pre-download YOLO model weights (m=GPU default, n=CPU fallback)
RUN uv run --no-sync python -c "from ultralytics import YOLO; YOLO('yolo11m.pt'); YOLO('yolo11n.pt')"

# Pre-download MiVOLO v2 model
RUN uv run --no-sync python -c '\
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification; \
m = "iitolstykh/mivolo_v2"; \
AutoConfig.from_pretrained(m, trust_remote_code=True); \
AutoImageProcessor.from_pretrained(m, trust_remote_code=True); \
AutoModelForImageClassification.from_pretrained(m, trust_remote_code=True)'

COPY . .

# Create storage directories
RUN mkdir -p storage/uploads storage/db

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
