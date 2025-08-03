FROM runpod/base:0.6.3-cuda11.8.0

# Set python3.11 as the default python
RUN ln -sf $(which python3.11) /usr/local/bin/python && \
    ln -sf $(which python3.11) /usr/local/bin/python3

# Set up the working directory
ARG WORKSPACE_DIR=/app
ENV WORKSPACE_DIR=${WORKSPACE_DIR}
WORKDIR $WORKSPACE_DIR

RUN uv venv
RUN uv pip install --no-cache-dir runpod hf-transfer huggingface-hub \
       git+https://github.com/nodetool-ai/nodetool-core \
       git+https://github.com/nodetool-ai/nodetool-base \
       git+https://github.com/nodetool-ai/nodetool-huggingface

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

RUN mkdir -p /app/.cache/huggingface/hub
ENV HF_HUB_CACHE=/app/.cache/huggingface/hub
    
# Copy all available files into the container
COPY . $WORKSPACE_DIR/

# Download HuggingFace models using the models.json specification
RUN /app/.venv/bin/python $WORKSPACE_DIR/download_models.py $WORKSPACE_DIR/models.json || echo "Model download completed with some errors"

# Copy and run Ollama model pull script
COPY pull_ollama_models.sh $WORKSPACE_DIR/pull_ollama_models.sh
RUN chmod +x $WORKSPACE_DIR/pull_ollama_models.sh && $WORKSPACE_DIR/pull_ollama_models.sh

# Copy startup script
COPY start.sh $WORKSPACE_DIR/start.sh
RUN chmod +x $WORKSPACE_DIR/start.sh

# Run the startup script
CMD ["/app/start.sh"]