# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.0-base-ubuntu22.04 AS base
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
    # Playwright browser dependencies
    # libgtk-4-1 \
    # libgraphene-1.0-0 \
    # libwoff2-1.0.2 \
    # libevent-2.1-7 \
    # libgstreamer-gl1.0-0 \
    # libgstreamer-plugins-bad1.0-0 \
    # libavif13 \
    # libharfbuzz-icu0 \
    # libenchant-2-2 \
    # libsecret-1-0 \
    # libhyphen0 \
    # libmanette-0.2-0 && \
    # Clean up apt cache
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Ensure UTF-8 locale is generated
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen en_US.UTF-8

RUN uv python install $PYTHON_VERSION
RUN uv venv --python $PYTHON_VERSION $VIRTUAL_ENV

ENV PATH=$VIRTUAL_ENV/bin:$PATH

# Set up the working directory
WORKDIR /app

# Build argument to control installation method
# Set USE_LOCAL_WHEELS=1 to use local wheel files from ./dist/
ARG USE_LOCAL_WHEELS=0

# Copy project into the image so we install the local checkout
COPY . /app

# Install Python packages from the local repo and required registries
# These will now use the pip from the Python 3.11 virtual environment
RUN echo "Installing nodetool packages..." && \
    if [ "$USE_LOCAL_WHEELS" = "1" ]; then \
        echo "Using local wheel files from ./dist/" && \
        uv pip install --python $VIRTUAL_ENV/bin/python --no-cache-dir \
        --find-links /app/dist \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        --index-strategy unsafe-best-match \
        /app \
        nodetool-base \
        nodetool-huggingface; \
    else \
        echo "Using package index" && \
        uv pip install --python $VIRTUAL_ENV/bin/python --no-cache-dir \
        --extra-index-url https://nodetool-ai.github.io/nodetool-registry/simple/ \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        --index-strategy unsafe-best-match \
        /app \
        nodetool-base \
        nodetool-huggingface; \
    fi

# RUN /app/venv/bin/playwright install

# Expose port for the worker
EXPOSE 8000

# Run the NodeTool worker
CMD ["python", "-m", "nodetool.deploy.worker"]
