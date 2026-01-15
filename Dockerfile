FROM ubuntu:22.04 AS base
COPY --from=ghcr.io/astral-sh/uv:0.8.5 /uv /uvx /bin/

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


FROM base AS pip-deps

ARG USE_LOCAL_REPO=0

# Copy local repository if using local installation
COPY --chown=root:root . /tmp/nodetool-core

    # Install dependencies - either from local repo or from nodetool-registry
    RUN if [ "$USE_LOCAL_REPO" = "1" ]; then \
            echo "Installing from local repository..." && \
            uv pip install \
                --index-strategy unsafe-best-match \
                --index-url https://pypi.org/simple \
                --extra-index-url https://nodetool-ai.github.io/nodetool-registry/simple/ \
                nodetool-base && \
            cd /tmp/nodetool-core && \
            uv pip install \
                --index-strategy unsafe-best-match \
                --index-url https://pypi.org/simple \
                -e . ; \
        else \
            echo "Installing from nodetool-registry..." && \
            uv pip install \
                --index-strategy unsafe-best-match \
                --index-url https://pypi.org/simple \
                --extra-index-url https://nodetool-ai.github.io/nodetool-registry/simple/ \
                nodetool-core==0.6.2-rc.24 nodetool-base==0.6.2-rc.24 ; \
        fi && \
        # Clean up
        rm -rf /tmp/nodetool-core && \
        find /root/.cache -type d -exec rm -rf {} + 2>/dev/null || true && \
        rm -rf /root/.cache/pip && \
        rm -rf /tmp/* && \
        rm -rf /var/tmp/*

FROM base AS final

ARG USE_LOCAL_REPO=0

COPY --from=pip-deps $VIRTUAL_ENV $VIRTUAL_ENV

    # Copy source code if using local repo (for editable install)
    COPY --chown=root:root . /app/nodetool-core
    RUN if [ "$USE_LOCAL_REPO" = "1" ]; then \
            echo "Reinstalling local repository in final image..." && \
            cd /app/nodetool-core && \
            uv pip install -e . ; \
        else \
            rm -rf /app/nodetool-core ; \
        fi

# Install Playwright browsers
# Use /var/tmp for browser downloads to avoid /tmp space issues on some systems
RUN TMPDIR=/var/tmp $VIRTUAL_ENV/bin/python -m playwright install && \
    rm -rf /var/tmp/* /tmp/*

# Expose port for the worker
EXPOSE 7777

# Run the NodeTool worker
CMD ["python", "-m", "nodetool.deploy.worker"]
