FROM nvidia/cuda:12.6.0-base-ubuntu22.04 AS base
COPY --from=ghcr.io/astral-sh/uv:0.8.5 /uv /uvx /bin/

# Note: llama-server is optional and may not be available on all platforms
# If you need llama-server support, you can add this line manually:
COPY --from=ghcr.io/ggml-org/llama.cpp:server-cuda /app/llama-server /usr/local/bin/llama-server

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    SHELL=/bin/bash \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYTHON_VERSION=3.11 \
    VIRTUAL_ENV=/app/venv

# Fix GPG keys and install system dependencies
RUN rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    apt-get update --allow-insecure-repositories || true && \
    apt-get install -y --allow-unauthenticated ca-certificates && \
    apt-get update && \
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


# Install external dependencies from GitHub releases (latest versions)
RUN echo "Installing external nodetool packages from GitHub releases..." && \
    uv pip install --python $VIRTUAL_ENV/bin/python --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    --index-strategy unsafe-best-match \
    git+https://github.com/nodetool-ai/nodetool-core.git@v0.6.2-rc.7 \
    git+https://github.com/nodetool-ai/nodetool-base.git@v0.6.2-rc.7 && \
    # Clean up pip, wheel, and other cached files to reduce image size
    find /root/.cache -type d -exec rm -rf {} + 2>/dev/null || true && \
    rm -rf /root/.cache/pip && \
    rm -rf /tmp/* && \
    rm -rf /var/tmp/*

# RUN /app/venv/bin/playwright install

# Expose port for the worker
EXPOSE 8000

# Run the NodeTool worker
CMD ["python", "-m", "nodetool.deploy.worker"]
