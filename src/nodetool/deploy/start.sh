#!/bin/bash
set -e

# Start Ollama service in background with network volume models directory
/opt/bin/ollama serve &
OLLAMA_PID=$!

# Start nodetool handler in foreground
exec /app/venv/bin/python -m nodetool.deploy.fastapi_server "$@"