# Which install stage the runtime image is built from: `final` (nodetool-core
# from PyPI, used for releases) or `dev` (nodetool-core from the build context,
# used for main-branch builds of unreleased commits).
ARG WORKER_BASE=final

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


# Release image (default target): both packages come from PyPI, so nothing from
# the build context is needed and the layer caches on the version args alone.
#
#   docker build --build-arg NODETOOL_VERSION=0.7.2 --build-arg HF_VERSION=0.7.2 .
FROM base AS final

ARG NODETOOL_VERSION=0.7.1
ARG HF_VERSION=0.7.1

# Install nodetool-core and nodetool-huggingface from PyPI.
RUN uv pip install \
        --python $VIRTUAL_ENV \
        --index-url https://pypi.org/simple \
        "nodetool-core==${NODETOOL_VERSION}" \
        "nodetool-huggingface==${HF_VERSION}" && \
    rm -rf /root/.cache/uv /root/.cache/pip /tmp/* /var/tmp/*


# Development image (--build-arg WORKER_BASE=dev): nodetool-core is installed
# from the build context instead of PyPI, so unreleased commits (e.g. main) can
# be imaged. nodetool-huggingface still comes from PyPI.
FROM base AS dev

ARG HF_VERSION=0.7.1

# nodetool-huggingface first, then the local source — nodetool-huggingface
# depends on nodetool-core, so installing it second guarantees the build
# context's version is the one that ends up in the image.
COPY . /src

RUN uv pip install \
        --python $VIRTUAL_ENV \
        --index-url https://pypi.org/simple \
        "nodetool-huggingface==${HF_VERSION}" && \
    uv pip install \
        --python $VIRTUAL_ENV \
        --index-url https://pypi.org/simple \
        /src && \
    rm -rf /src /root/.cache/uv /root/.cache/pip /tmp/* /var/tmp/*


FROM ${WORKER_BASE} AS runtime

# Fail the build if the torch stack cannot import.
#
# torchvision and torchaudio ship CUDA-variant wheels that must match torch's.
# When nodetool-huggingface left them unpinned, the resolver installed
# torchaudio 2.11.0 (built against CUDA 13) next to torch 2.9.0+cu128, and the
# import died with `libcudart.so.13: cannot open shared object file`. Nothing
# here checked, so the image published fine and every HuggingFace node failed
# at execute time instead — the worker even reported zero load errors, because
# discovery does not import the extensions.
#
# This runs at build time on a CPU-only builder, so it must not touch a GPU.
# The import is the whole check: the .so loads against torch's CUDA runtime, or
# it does not.
RUN python -c "\
import torch, torchvision, torchaudio; \
print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'torchaudio', torchaudio.__version__)"

# Expose the worker's WebSocket port
EXPOSE 7777

# Health check — the worker is a WebSocket server (no HTTP route), so probe it
# with a real WebSocket handshake. A raw TCP connect gets rejected mid-handshake
# and spams the worker log with tracebacks; a proper ws:// connect that closes
# cleanly verifies liveness without the noise. `websockets` ships with nodetool-core.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from websockets.sync.client import connect; connect('ws://127.0.0.1:7777', open_timeout=5).close()" || exit 1

# Run the NodeTool Python worker (WebSocket transport, reachable from the TS server)
CMD ["python", "-m", "nodetool.worker", "--host", "0.0.0.0", "--port", "7777"]
