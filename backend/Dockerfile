FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# Switch apt sources to Aliyun mirror
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_PYTHON=3.11
ENV UV_COMPILE_BYTECODE=0
ENV UV_LINK_MODE=copy
ENV UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# Pass --build-arg HF_ENDPOINT=https://hf-mirror.com for faster downloads in China
ARG HF_ENDPOINT=https://huggingface.co
ENV HF_ENDPOINT=${HF_ENDPOINT}
# Model weights are stored in a persistent volume, not baked into the image
ENV HF_HOME=/app/model_weights/huggingface

WORKDIR /app

COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project && uv cache clean

COPY . .
RUN mkdir -p storage/uploads storage/db

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
