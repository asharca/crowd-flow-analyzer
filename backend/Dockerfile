# CUDA 12.1 + cuDNN 8 runtime — provides libcuda/libcudnn for PyTorch & ONNX Runtime GPU
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (includes Python management)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_PYTHON=3.11
ENV UV_COMPILE_BYTECODE=0
ENV UV_LINK_MODE=copy
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# Install Python deps
COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project

# Pre-download YOLOv8n model weights
RUN uv run python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Pre-download InsightFace buffalo_l model (SCRFD detector + genderage ONNX)
RUN uv run python -c 'import insightface.app as ia; app = ia.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"]); app.prepare(ctx_id=-1, det_size=(640, 640))'

COPY . .

# Create storage directories
RUN mkdir -p storage/uploads storage/db

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
