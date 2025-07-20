# Base image with CUDA and cuDNN runtime
FROM nvcr.io/nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04 AS base

# Set the working directory
WORKDIR /app

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    SHELL=/bin/bash \
    PYTHONPATH=/app \
    # Add /root/.local/bin to PATH for user-specific pip installs if any occur outside venv
    PATH="/root/.local/bin:$PATH" \
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
    tesseract-ocr \
    # Other System Libraries
    libcairo2-dev \
    libgl1 \
    libgl1-mesa-glx && \
    # Add the deadsnakes PPA for newer Python versions
    echo "Adding deadsnakes PPA for Python ${PYTHON_VERSION}" && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    # Install Python ${PYTHON_VERSION} and its development, venv, and tkinter packages
    echo "Installing Python ${PYTHON_VERSION}..." && \
    apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-tk && \
    # Clean up apt cache
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Ensure UTF-8 locale is generated
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen en_US.UTF-8

# Make python3.11 the default python3 and python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# Install pip for Python 3.11
# Using get-pip.py ensures we get a version of pip compatible with python3.11
RUN echo "Installing pip for Python ${PYTHON_VERSION}..." && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}

# Create virtual environment using python3.11
RUN echo "Creating virtual environment at ${VIRTUAL_ENV} using Python ${PYTHON_VERSION}..." && \
    python${PYTHON_VERSION} -m venv $VIRTUAL_ENV

# Add the virtual environment's bin directory to the PATH
# This ensures that 'pip', 'python' commands use the venv's versions
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip within the virtual environment
RUN echo "Upgrading pip in virtual environment..." && \
    pip install --upgrade pip setuptools wheel

# Install Python packages from GitHub repositories and PyPI
# These will now use the pip from the Python 3.11 virtual environment
RUN echo "Installing nodetool packages..." && \
    pip install --no-cache-dir git+https://github.com/nodetool-ai/nodetool-core && \
    pip install --no-cache-dir git+https://github.com/nodetool-ai/nodetool-base && \
    pip install --no-cache-dir git+https://github.com/nodetool-ai/nodetool-lib-audio && \
    pip install --no-cache-dir git+https://github.com/nodetool-ai/nodetool-lib-data && \
    pip install --no-cache-dir git+https://github.com/nodetool-ai/nodetool-lib-image && \
    pip install --no-cache-dir git+https://github.com/nodetool-ai/nodetool-lib-ml && \
    pip install --no-cache-dir git+https://github.com/nodetool-ai/nodetool-lib-network

RUN echo "Installing nodetool-huggingface with PyTorch..." && \
    pip install --no-cache-dir git+https://github.com/nodetool-ai/nodetool-huggingface --extra-index-url https://download.pytorch.org/whl/cu121

# # Install Playwright browser binaries
RUN echo "Installing Playwright browser binaries..." && \
    playwright install

# Optional: Add commands to verify versions (uncomment to use during build for debugging)
RUN python --version
RUN python3 --version
RUN pip --version
RUN pip list

# Set default command (optional, if your app has one)
# CMD ["python", "your_app_entrypoint.py"]

