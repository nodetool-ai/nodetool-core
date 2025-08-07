# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04 AS base
COPY --from=ghcr.io/astral-sh/uv:0.8.5 /uv /uvx /bin/

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    SHELL=/bin/bash \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYTHON_VERSION=3.11 \
    VIRTUAL_ENV=/app/venv

# Install system dependencies, then Python 3.11 and related packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    # Essential for adding PPAs
    software-properties-common \
    locales \
    # Python build essentials and other tools
    build-essential \
    gcc \
    g++ \
    git \
    wget \
    curl \
    # System Libraries often needed for Python compilation or packages
    libssl-dev \
    libffi-dev \
    liblzma-dev \
    zlib1g-dev \
    # Scientific Computing Libraries
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    # Image Processing
    libopencv-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    libgif-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # Audio and Video Processing
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    libopus-dev \
    libx264-dev \
    libmp3lame-dev \
    libvorbis-dev \
    # Data Processing and Storage
    libxml2-dev \
    libxslt1-dev \
    libsqlite3-dev \
    # Document Processing
    tesseract-ocr && \
    # Clean up apt cache
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Ensure UTF-8 locale is generated
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen en_US.UTF-8

RUN uv python install $PYTHON_VERSION
RUN uv venv $VIRTUAL_ENV

# Set up the working directory
WORKDIR /app
ENV HF_HUB_CACHE=/runpod-volume/.cache/huggingface/hub
ENV HF_HOME=/runpod-volume/.cache/huggingface
ENV OLLAMA_MODELS=/runpod-volume/.ollama/models

RUN mkdir -p /runpod-volume/.cache/huggingface
RUN mkdir -p /runpod-volume/.ollama/models
RUN mkdir -p /runpod-volume/.cache/transformers

RUN echo "Installing nodetool packages..." && \
    uv pip install --no-cache-dir git+https://github.com/nodetool-ai/nodetool-core@cb3299e70064a6fbde1683d920b5bbaf0d05b3a6 && \
    uv pip install --no-cache-dir git+https://github.com/nodetool-ai/nodetool-base

# Install Ollama
RUN curl --fail --show-error --location --progress-bar \
        "https://github.com/ollama/ollama/releases/download/v0.11.3/ollama-linux-amd64.tgz" \
        | tar -xzf - -C /opt

COPY . /app
ADD start.sh /app/start.sh
ADD workflows /app/workflows
RUN chmod +x /app/start.sh

# Run the startup script
CMD ["/app/start.sh"]