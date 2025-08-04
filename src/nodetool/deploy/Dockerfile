# syntax=docker/dockerfile:1
FROM runpod/base:0.6.3-cuda11.8.0

# Set python3.11 as the default python
RUN ln -sf $(which python3.11) /usr/local/bin/python && \
    ln -sf $(which python3.11) /usr/local/bin/python3

# Set up the working directory
ARG WORKSPACE_DIR=/app
ENV WORKSPACE_DIR=${WORKSPACE_DIR}
WORKDIR $WORKSPACE_DIR

# Create virtual environment (cacheable layer)
RUN uv venv

# Install Python dependencies with cache mount
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-cache-dir runpod hf-transfer huggingface-hub \
       git+https://github.com/nodetool-ai/nodetool-core \
       git+https://github.com/nodetool-ai/nodetool-base

#    git+https://github.com/nodetool-ai/nodetool-huggingface

# Install Ollama with cache mount
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    curl -fsSL https://ollama.com/install.sh | sh

# Set up HuggingFace cache directory
RUN mkdir -p /app/.cache/huggingface/hub
ENV HF_HUB_CACHE=/app/.cache/huggingface/hub

# Copy dependency files first (better caching)
COPY models.json download_models.py ./

# Download HuggingFace models using the models.json specification with cache mount
RUN --mount=type=cache,target=/app/.cache/huggingface \
    /app/.venv/bin/python $WORKSPACE_DIR/download_models.py $WORKSPACE_DIR/models.json || echo "Model download completed with some errors"

# Copy and run Ollama model pull script
COPY pull_ollama_models.sh $WORKSPACE_DIR/pull_ollama_models.sh
RUN chmod +x $WORKSPACE_DIR/pull_ollama_models.sh
RUN --mount=type=cache,target=/root/.ollama \
    $WORKSPACE_DIR/pull_ollama_models.sh

# Copy remaining files
COPY . $WORKSPACE_DIR/

# Copy startup script and set permissions
COPY start.sh $WORKSPACE_DIR/start.sh
RUN chmod +x $WORKSPACE_DIR/start.sh

# Run the startup script
CMD ["/app/start.sh"]