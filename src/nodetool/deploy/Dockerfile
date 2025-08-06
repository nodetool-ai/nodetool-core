# syntax=docker/dockerfile:1
FROM runpod/base:0.6.3-cuda11.8.0

# Set python3.11 as the default python
RUN ln -sf $(which python3.11) /usr/local/bin/python && \
    ln -sf $(which python3.11) /usr/local/bin/python3

# Set up the working directory
ARG WORKSPACE_DIR=/app
ENV WORKSPACE_DIR=${WORKSPACE_DIR}
WORKDIR $WORKSPACE_DIR
ENV HF_HUB_CACHE=/runpod-volume/.cache/huggingface/hub
ENV HF_HOME=/runpod-volume/.cache/huggingface
ENV OLLAMA_MODELS=/runpod-volume/.ollama/models

RUN mkdir -p /runpod-volume/.cache/huggingface
RUN mkdir -p /runpod-volume/.ollama/models
RUN mkdir -p /runpod-volume/.cache/transformers

# Install Ollama with cache mount
RUN curl -fsSL https://ollama.com/install.sh | sh

RUN uv venv

# Install Python dependencies with cache mount
RUN uv pip install --no-cache-dir runpod hf-transfer huggingface-hub --upgrade \
       git+https://github.com/nodetool-ai/nodetool-core \
       git+https://github.com/nodetool-ai/nodetool-base

#    git+https://github.com/nodetool-ai/nodetool-huggingface


COPY . $WORKSPACE_DIR/
RUN chmod +x $WORKSPACE_DIR/start.sh

RUN mkdir /workflows

# Run the startup script
CMD ["/app/start.sh"]