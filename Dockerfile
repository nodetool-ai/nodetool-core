FROM mambaorg/micromamba:jammy AS base

# Install uv
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


FROM base AS pip-deps

ARG USE_LOCAL_REPO=1

# Copy local repository if using local installation
COPY --chown=root:root . /tmp/nodetool-core

    # Install dependencies - we use uv pip install into the conda environment
    RUN if [ "$USE_LOCAL_REPO" = "1" ]; then \
            echo "Installing from local repository..." && \
            uv pip install \
                --python $VIRTUAL_ENV \
                --index-strategy unsafe-best-match \
                --index-url https://pypi.org/simple \
                --extra-index-url https://nodetool-ai.github.io/nodetool-registry/simple/ \
                nodetool-base && \
            cd /tmp/nodetool-core && \
            uv pip install \
                --python $VIRTUAL_ENV \
                --index-strategy unsafe-best-match \
                --index-url https://pypi.org/simple \
                -e . ; \
        else \
            echo "Installing from nodetool-registry..." && \
            uv pip install \
                --python $VIRTUAL_ENV \
                --index-strategy unsafe-best-match \
                --index-url https://pypi.org/simple \
                --extra-index-url https://nodetool-ai.github.io/nodetool-registry/simple/ \
                nodetool-core==0.6.3-rc.18 nodetool-base==0.6.3-rc.18 ; \
        fi && \
        # Clean up
        rm -rf /tmp/nodetool-core && \
        find /root/.cache -type d -exec rm -rf {} + 2>/dev/null || true && \
        rm -rf /root/.cache/pip && \
        rm -rf /tmp/* && \
        rm -rf /var/tmp/*

FROM base AS final

ARG USE_LOCAL_REPO=1

# Micromamba image logic might persist, but we are copying VIRTUAL_ENV
# Note: copying a conda env between stages can sometimes break absolute paths if location changes
# But here location $VIRTUAL_ENV (/opt/conda) is kept same
COPY --from=pip-deps $VIRTUAL_ENV $VIRTUAL_ENV

    # Copy source code if using local repo (for editable install)
COPY --chown=root:root . /app/nodetool-core
RUN if [ "$USE_LOCAL_REPO" = "1" ]; then \
        echo "Reinstalling local repository in final image..." && \
        cd /app/nodetool-core && \
        uv pip install --python $VIRTUAL_ENV -e . ; \
    else \
        rm -rf /app/nodetool-core ; \
    fi

# Install Playwright browsers and system dependencies
# Use /var/tmp for browser downloads to avoid /tmp space issues on some systems
RUN TMPDIR=/var/tmp python -m playwright install-deps chromium firefox webkit && \
    TMPDIR=/var/tmp python -m playwright install && \
    rm -rf /var/tmp/* /tmp/*

# Expose port for the server
EXPOSE 7777

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7777/health || exit 1

# Run the NodeTool server
CMD ["python", "-m", "nodetool.api.run_server"]
