FROM mambaorg/micromamba:jammy AS base

# Install uv (used to install nodetool-core into the conda base env)
COPY --from=ghcr.io/astral-sh/uv:0.8.5 /uv /uvx /bin/

# Switch to root for system package installation
USER root

ENV DEBIAN_FRONTEND=noninteractive \
    SHELL=/bin/bash \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYTHON_VERSION=3.11 \
    # Point VIRTUAL_ENV to micromamba base environment
    VIRTUAL_ENV=/opt/conda

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
    python3-pip \
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

# Install Python and Pip via Micromamba
RUN micromamba install -y -n base -c conda-forge python=$PYTHON_VERSION pip && \
    micromamba clean --all --yes

# Ensure micromamba environment path is first
ENV PATH=$VIRTUAL_ENV/bin:$PATH


FROM base AS final

# nodetool-core version to install from PyPI. Override at build time:
#   docker build --build-arg NODETOOL_VERSION=0.7.2 .
ARG NODETOOL_VERSION=0.7.1

# Install nodetool-core (the Python node-runner worker) from PyPI.
# Node packages are layered separately on top of this image.
RUN uv pip install \
        --python $VIRTUAL_ENV \
        --index-url https://pypi.org/simple \
        "nodetool-core==${NODETOOL_VERSION}" && \
    # Clean up caches to keep the image small
    rm -rf /root/.cache/uv /root/.cache/pip /tmp/* /var/tmp/*

# Expose the worker's WebSocket port
EXPOSE 7777

# Health check — the worker is a WebSocket server (no HTTP /health route),
# so verify the port is accepting connections rather than curling an endpoint.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import socket; socket.create_connection(('127.0.0.1', 7777), timeout=3).close()" || exit 1

# Run the NodeTool Python worker (WebSocket transport, reachable from the TS server)
CMD ["python", "-m", "nodetool.worker", "--host", "0.0.0.0", "--port", "7777"]
