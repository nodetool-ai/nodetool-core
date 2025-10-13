# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.0-base-ubuntu22.04 AS base

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:0.8.5 /uv /uvx /bin/

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    SHELL=/bin/bash \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYTHON_VERSION=3.11 \
    VIRTUAL_ENV=/app/venv \
    PATH="/app/venv/bin:$PATH"

# Install system dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    # Essential tools
    software-properties-common \
    locales \
    curl \
    wget \
    git \
    # Python and build essentials
    build-essential \
    gcc \
    g++ \
    # System libraries
    libssl-dev \
    libffi-dev \
    liblzma-dev \
    zlib1g-dev \
    # Scientific computing
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    # Image processing
    libopencv-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    libgif-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # Audio and video
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    libopus-dev \
    libx264-dev \
    libmp3lame-dev \
    libvorbis-dev \
    # Data processing
    libxml2-dev \
    libxslt1-dev \
    libsqlite3-dev \
    # OCR
    tesseract-ocr && \
    # Clean up
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Generate UTF-8 locale
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen en_US.UTF-8

# Install Python and create virtual environment
RUN uv python install $PYTHON_VERSION && \
    uv venv $VIRTUAL_ENV

# Set up working directory
WORKDIR /app

# Copy nodetool-core source
COPY . /app

# Install nodetool-core and dependencies using uv
RUN uv pip install --system /app

# Create default workspace directory structure
RUN mkdir -p /workspace/data \
    /workspace/assets \
    /workspace/temp \
    /hf-cache/huggingface/hub \
    /hf-cache/transformers

# Copy startup script
COPY src/nodetool/deploy/self-hosted/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the startup script
CMD ["/app/start.sh"]
