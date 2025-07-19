FROM runpod/base:0.6.3-cuda11.8.0

# Set python3.11 as the default python
RUN ln -sf $(which python3.11) /usr/local/bin/python && \
    ln -sf $(which python3.11) /usr/local/bin/python3

# Set up the working directory
ARG WORKSPACE_DIR=/app
ENV WORKSPACE_DIR=${WORKSPACE_DIR}
WORKDIR $WORKSPACE_DIR

RUN uv venv
RUN uv pip install --no-cache-dir runpod && \
    uv pip install --no-cache-dir git+https://github.com/nodetool-ai/nodetool-core && \
    uv pip install --no-cache-dir git+https://github.com/nodetool-ai/nodetool-base
    
# Copy all of our files into the container
COPY runpod_handler.py $WORKSPACE_DIR/handler.py
COPY workflow.json $WORKSPACE_DIR/workflow.json

# Run the handler
CMD /app/.venv/bin/python -u /app/handler.py --rp_log_level=DEBUG