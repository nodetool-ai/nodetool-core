#!/bin/bash
set -e

echo "🚀 Starting NodeTool Self-Hosted Container"

# Create workspace structure if it doesn't exist
mkdir -p /workspace/data
mkdir -p /workspace/assets
mkdir -p /workspace/temp

# Ensure HF cache directories exist (may be shared mount)
mkdir -p /hf-cache/huggingface/hub
mkdir -p /hf-cache/transformers

# Set environment variables for workspace paths
export DB_PATH="${DB_PATH:-/workspace/data/nodetool.db}"
export ASSET_FOLDER="${ASSET_FOLDER:-/workspace/assets}"
export ASSET_TEMP_BUCKET="${ASSET_TEMP_BUCKET:-/workspace/temp}"

# HF cache paths (shared across containers)
export HF_HOME="${HF_HOME:-/hf-cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/hf-cache/huggingface/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/hf-cache/transformers}"

# Set default environment (don't use production to avoid S3 requirement)
export ENV="${ENV:-development}"
export REMOTE_AUTH="${REMOTE_AUTH:-false}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Use llama.cpp as default provider (spawns subprocesses on demand)
export CHAT_PROVIDER="${CHAT_PROVIDER:-llama_cpp}"
export DEFAULT_MODEL="${DEFAULT_MODEL:-}"

# Server configuration
export PORT="${PORT:-8000}"
export NODETOOL_API_URL="${NODETOOL_API_URL:-http://localhost:8000}"

echo "📂 Workspace: /workspace"
echo "💾 Database: $DB_PATH"
echo "🗄️  Assets: $ASSET_FOLDER"
echo "🤖 HF Cache: $HF_HOME"
echo "🔧 Provider: $CHAT_PROVIDER"
if [ -n "$DEFAULT_MODEL" ]; then
    echo "🧠 Default Model: $DEFAULT_MODEL"
fi
echo ""

echo "🎯 Starting NodeTool FastAPI server on port $PORT..."

# Start NodeTool server
exec /app/venv/bin/python -m nodetool.deploy.fastapi_server
