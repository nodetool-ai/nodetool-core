# NodeTool Production Deployment Guide

This guide covers production deployment of the NodeTool worker server.

## Overview

The NodeTool worker (`python -m nodetool.deploy.worker`) is a complete, production-ready FastAPI server that includes:

- ✅ **OpenAI-compatible chat API** - `/v1/chat/completions`, `/v1/models`
- ✅ **Workflow execution** - Execute workflows via REST API
- ✅ **Admin endpoints** - Model management, cache, database operations
- ✅ **Built-in authentication** - Token-based authentication for all endpoints
- ✅ **Storage management** - File upload/download with asset management
- ✅ **Collection/RAG management** - ChromaDB integration for vector storage
- ✅ **Health checks** - `/health` and `/ping` endpoints

**No separate proxy is needed** - the worker handles everything.

## Quick Start with Docker Compose

The easiest way to deploy NodeTool in production is using Docker Compose:

```bash
# Copy and configure environment
cp .env.example .env.production
# Edit .env.production with your configuration

# Set required production variables
export ENV=production
export SECRETS_MASTER_KEY=your-secure-master-key-here
export WORKER_AUTH_TOKEN=your-secure-admin-token-here

# Start the services
docker-compose up -d

# Check logs
docker-compose logs -f api
```

The worker will be available at:
- HTTP: http://localhost (proxied via nginx)
- HTTPS: https://localhost (with self-signed certs, update for production)

## Environment Configuration

### Required for Production

```bash
# Environment mode
ENV=production

# Master key for encrypting secrets (REQUIRED in production)
SECRETS_MASTER_KEY=your-cryptographically-secure-key

# Admin authentication token (auto-generated if not set, but set explicitly for production)
WORKER_AUTH_TOKEN=your-secure-admin-token

# Default AI provider and model
CHAT_PROVIDER=ollama
DEFAULT_MODEL=gpt-oss:20b

# Port (default: 8000)
PORT=8000
```

### Optional Configuration

```bash
# Database (PostgreSQL for production, SQLite for dev)
POSTGRES_DB=nodetool
POSTGRES_USER=nodetool_user
POSTGRES_PASSWORD=your_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Storage (S3-compatible)
ASSET_BUCKET=nodetool-assets
S3_ENDPOINT_URL=https://s3.amazonaws.com
S3_ACCESS_KEY_ID=your_key
S3_SECRET_ACCESS_KEY=your_secret
S3_REGION=us-east-1

# Authentication Provider
AUTH_PROVIDER=supabase  # or: none, local, static
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Vector Database
CHROMA_PATH=/chroma-data
CHROMA_URL=http://chroma:8000  # if using remote ChromaDB

# AI Provider API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
HF_TOKEN=hf_...

# Monitoring
SENTRY_DSN=your_sentry_dsn
```

## Docker Deployment

### Using Pre-built Image

```bash
# Pull the latest image
docker pull ghcr.io/nodetool-ai/nodetool:latest

# Run the worker
docker run -d \
  -p 8000:8000 \
  -e ENV=production \
  -e SECRETS_MASTER_KEY=your-key \
  -e WORKER_AUTH_TOKEN=your-token \
  -e CHAT_PROVIDER=ollama \
  -e DEFAULT_MODEL=gpt-oss:20b \
  ghcr.io/nodetool-ai/nodetool:latest
```

### Building from Source

```bash
# Build the image
docker build -t nodetool-worker .

# Run the worker
docker run -d -p 8000:8000 \
  -e ENV=production \
  -e SECRETS_MASTER_KEY=your-key \
  -e WORKER_AUTH_TOKEN=your-token \
  nodetool-worker
```

## Authentication

All endpoints (except `/health` and `/ping`) require Bearer token authentication.

### Admin Token

The admin token is used to secure all worker endpoints:

1. **Auto-generated**: If `WORKER_AUTH_TOKEN` is not set, a token is auto-generated on first run and saved to `~/.config/nodetool/deployment.yaml`

2. **Environment variable**: Set `WORKER_AUTH_TOKEN` explicitly (recommended for production)

3. **Master key**: For production deployments, also set `SECRETS_MASTER_KEY` for encrypting user secrets

### Making Authenticated Requests

```bash
# Set your token
export WORKER_AUTH_TOKEN=your-token-here

# List models
curl -H "Authorization: Bearer $WORKER_AUTH_TOKEN" \
  http://localhost:8000/v1/models

# Chat completion
curl -H "Authorization: Bearer $WORKER_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -X POST http://localhost:8000/v1/chat/completions \
  -d '{
    "model": "gpt-oss:20b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Admin endpoints
curl -H "Authorization: Bearer $WORKER_AUTH_TOKEN" \
  http://localhost:8000/admin/collections
```

## API Endpoints

### Public (No Auth Required)
- `GET /health` - Health check
- `GET /ping` - Ping with system info

### OpenAI-Compatible (Auth Required)
- `POST /v1/chat/completions` - Chat completions with SSE streaming
- `GET /v1/models` - List available models

### Workflows (Auth Required)
- `GET /workflows` - List workflows
- `POST /workflows/run` - Execute workflow

### Admin (Auth Required)
- `POST /admin/models/huggingface/download` - Download HF model
- `POST /admin/models/ollama/download` - Download Ollama model
- `GET /admin/cache/scan` - Scan cache
- `GET /admin/cache/size` - Get cache size
- `GET /admin/collections` - List collections
- `POST /admin/collections` - Create collection
- `GET /admin/assets` - List assets
- `POST /admin/assets` - Create asset

### Storage (Auth Required)
- `PUT /admin/storage/assets/{path}` - Upload file
- `GET /storage/assets/{path}` - Download file (public read)

## Health Monitoring

```bash
# Basic health check (no auth)
curl http://localhost:8000/health
# Returns: {"status": "healthy", "timestamp": "2024-..."}

# Detailed ping (no auth)
curl http://localhost:8000/ping
# Returns: {"status": "healthy", "timestamp": "2024-..."}
```

## Scaling

For production deployments:

1. **Horizontal scaling**: Run multiple worker instances behind a load balancer
2. **Database**: Use PostgreSQL instead of SQLite
3. **Storage**: Use S3-compatible storage instead of local filesystem
4. **Vector DB**: Use remote ChromaDB instance
5. **Caching**: Enable Memcached for improved performance
6. **Monitoring**: Enable Sentry for error tracking

## Troubleshooting

### Check worker logs
```bash
docker-compose logs -f api
```

### Verify authentication
```bash
# Should fail without token
curl http://localhost:8000/v1/models
# Returns: 401 Unauthorized

# Should succeed with token
curl -H "Authorization: Bearer $WORKER_AUTH_TOKEN" \
  http://localhost:8000/v1/models
# Returns: {"models": [...]}
```

### Check environment variables
```bash
docker-compose exec api env | grep -E "(ENV|WORKER_AUTH|SECRETS_MASTER)"
```

## Security Best Practices

1. **Always set SECRETS_MASTER_KEY in production** - Required for encrypting user secrets
2. **Set WORKER_AUTH_TOKEN explicitly** - Don't rely on auto-generation in production
3. **Use HTTPS** - Configure nginx with proper TLS certificates (update cert.pem and key.pem)
4. **Use PostgreSQL** - Don't use SQLite in production
5. **Use S3 storage** - Don't use local filesystem for assets in production
6. **Enable Sentry** - Monitor errors in production
7. **Regular backups** - Backup database and vector store regularly

## Migration from Proxy-based Deployment

If you were using the old proxy-based deployment:

1. The proxy is no longer needed - the worker handles everything
2. Update docker-compose.yaml to use `python -m nodetool.deploy.worker`
3. Update port from 7777 to 8000 (configurable via PORT environment variable)
4. Set WORKER_AUTH_TOKEN for admin authentication
5. Remove proxy containers and configurations

The worker now includes all functionality previously provided by the proxy plus admin endpoints and authentication.
