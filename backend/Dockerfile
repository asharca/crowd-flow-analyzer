FROM ubuntu:22.04

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
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

COPY . .

# Create storage directories
RUN mkdir -p storage/uploads storage/db

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
