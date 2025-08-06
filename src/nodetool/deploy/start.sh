#!/bin/bash
set -e

# Start Ollama service in background with network volume models directory
ollama serve &
OLLAMA_PID=$!

# Start nodetool handler in foreground
exec /app/.venv/bin/python -m nodetool.deploy.fastapi_server "$@"