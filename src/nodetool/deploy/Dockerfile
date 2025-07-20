FROM runpod/base:0.6.3-cuda11.8.0

# Set python3.11 as the default python
RUN ln -sf $(which python3.11) /usr/local/bin/python && \
    ln -sf $(which python3.11) /usr/local/bin/python3

# Set up the working directory
ARG WORKSPACE_DIR=/app
ENV WORKSPACE_DIR=${WORKSPACE_DIR}
WORKDIR $WORKSPACE_DIR

RUN uv venv
RUN uv pip install --no-cache-dir runpod hf-transfer \
       git+https://github.com/nodetool-ai/nodetool-core \
       git+https://github.com/nodetool-ai/nodetool-base \
       git+https://github.com/nodetool-ai/nodetool-huggingface

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

RUN mkdir -p /app/.cache/huggingface/hub
ENV HF_HUB_CACHE=/app/.cache/huggingface/hub
    
# Copy all of our files into the container
COPY runpod_handler.py $WORKSPACE_DIR/handler.py
COPY workflow.json $WORKSPACE_DIR/workflow.json
COPY models.json $WORKSPACE_DIR/models.json

# Copy HuggingFace models from local cache if they exist
COPY huggingface/hub/ /app/.cache/huggingface/hub/

# Copy and run Ollama model pull script
COPY pull_ollama_models.sh $WORKSPACE_DIR/pull_ollama_models.sh
RUN chmod +x $WORKSPACE_DIR/pull_ollama_models.sh && $WORKSPACE_DIR/pull_ollama_models.sh

# Run the handler
CMD ollama serve & /app/.venv/bin/python -u /app/handler.py --rp_log_level=DEBUG