# CUDA 12.9 runtime — matches driver 575 (supports up to CUDA 12.9).
# torch+cu128 is backwards-compatible with this runtime; cu130 requires driver >=580.
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu22.04

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
# MiVOLO's setup.py uses pkg_resources — uv's no-build-isolation can't resolve it.
# Workaround: pre-install setuptools + mivolo via pip, then uv sync skips rebuilding.
RUN uv venv && \
    uv pip install setuptools && \
    uv pip install --no-build-isolation "mivolo @ git+https://github.com/WildChlamydia/MiVOLO.git" && \
    uv sync --no-dev --no-install-project

COPY . .

# Create storage directories
RUN mkdir -p storage/uploads storage/db

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
