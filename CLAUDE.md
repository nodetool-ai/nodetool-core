[← Back to Docs Index](docs/index.md)

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Configuration

NodeTool Core uses environment-specific configuration files to manage different deployment scenarios:

### Environment Files

- **`.env.example`** - Template file with all possible configuration options (committed)
- **`.env.development`** - Development environment defaults (committed)
- **`.env.test`** - Test environment configuration (committed)
- **`.env.production`** - Production environment configuration (committed)
- **`.env.*.local`** - Local overrides with actual secrets (gitignored)

### Loading Order

Configuration is loaded in this order (later sources override earlier ones):

1. Default values from `DEFAULT_ENV` in `environment.py`
1. Base `.env` file (if exists)
1. Environment-specific file (`.env.development`, `.env.test`, or `.env.production`)
1. Local override file (`.env.{environment}.local`)
1. System environment variables
1. YAML settings/secrets files

### Environment Detection

The system determines the environment from:

1. `ENV` environment variable
1. `PYTEST_CURRENT_TEST` (automatically sets test environment)
1. Defaults to "development"

### Security

- Template files (`.env.{env}`) contain no actual secrets
- Actual API keys go in `.env.{env}.local` files (gitignored)
- Production secrets should be set via environment variables
- Never commit files containing API keys or credentials
- Use the `register_setting()` system for adding new configuration options

## Development Environment

### Setup and Installation

**Recommended: Using conda + uv**

```bash
# Create and activate conda environment with system dependencies
conda create -n nodetool python=3.11 pandoc ffmpeg -c conda-forge
conda activate nodetool

# Install Python dependencies with uv (fast and reliable)
uv sync

# Install development dependencies
uv sync --group dev

# Set up environment configuration
cp .env.example .env.development.local
# Edit .env.development.local with your actual API keys
```

**Alternative: Using pip only**

```bash
# Install dependencies with pip
pip install .

# Install development dependencies
pip install -r requirements-dev.txt

# Note: You'll need to install system dependencies separately:
# - pandoc (for document processing)
# - ffmpeg (for media processing)
```

### Common Commands

````bash
# Run all tests
uv run pytest -q

# Run a specific test file
uv run pytest tests/path/to/test_file.py

# Run tests with coverage report
uv run pytest --cov=src

# Lint code
uv run ruff check .
uv run black --check .
uv run mypy .

# Format code
uv run black .

# Add dependencies
uv add package-name

# Add development dependencies
uv add --group dev package-name

# Update dependencies
uv sync

## Project Architecture

NodeTool Core is a Python library for building and running AI workflows using a modular, node-based approach. It consists of several key components:

### Key Components

1. **Workflow System** (`src/nodetool/workflows/`)
   - Represents AI workflows as Directed Acyclic Graphs (DAGs)
   - `Graph`: Contains nodes and edges defining workflow structure
   - `BaseNode`: Basic unit of computation with inputs, outputs, and properties
   - `WorkflowRunner`: Executes graphs by analyzing dependencies and managing execution
   - `ProcessingContext`: Holds runtime information for workflow execution

2. **Agent System** (`src/nodetool/agents/`)
   - Enables LLMs to accomplish complex tasks by breaking them down into subtasks
   - `Agent`: Coordinates planning and execution of tasks
   - `TaskPlanner`: Breaks objectives into structured plans of subtasks
   - `TaskExecutor`: Manages execution of subtasks, handling dependencies
   - `SubTaskContext`: Provides isolated environment for each subtask
   - `Tools`: Specialized utilities for actions like web browsing and file handling

3. **Chat System** (`src/nodetool/chat/`)
   - Handles interactions with AI providers
   - Various provider integrations (OpenAI, Anthropic, Gemini, Ollama)
   - Manages context and conversation flow

4. **Storage System** (`src/nodetool/storage/`)
   - Provides abstractions for data persistence
   - Multiple backend options (memory, file, S3)
   - Caching mechanisms for performance optimization

5. **API Layer** (`src/nodetool/api/`)
   - FastAPI-based server for exposing workflow functionality
   - Handles job management, asset storage, and processing
   - WebSocket support for real-time updates

6. **Models Layer** (`src/nodetool/models/`)
   - Database models and schemas
   - Supports multiple database backends (SQLite, PostgreSQL, Supabase)

### Data Flow

1. Client requests workflow execution via API or direct Python imports
2. WorkflowRunner processes the graph according to node dependencies
3. Nodes execute when dependencies are satisfied and resources are available
4. Results flow through the graph according to defined edges
5. Final outputs are collected from output nodes and returned to the client

### Key Design Patterns

1. **Dependency Injection** - Components receive their dependencies through constructors
2. **Asynchronous Processing** - Heavy use of Pythons asyncio for non-blocking operations
3. **Factory Pattern** - Provider factories create appropriate implementation instances
4. **Strategy Pattern** - Different storage/database backends implement common interfaces
5. **Observer Pattern** - WebSocket updates provide real-time progress tracking

## Code Organization

- `src/nodetool/`: Main package
  - `agents/`: Agent system for complex task execution
  - `api/`: FastAPI server and endpoints
  - `chat/`: LLM provider integrations
  - `common/`: Shared utilities and helpers
  - `dsl/`: Domain-specific language for workflow creation
  - `metadata/`: Type definitions and metadata handling
  - `models/`: Database models and adapters
  - `storage/`: Storage abstractions and implementations
  - `workflows/`: Core workflow system
  - `cli.py`: Command-line interface

- `tests/`: Test suite organized to mirror the src structure
- `examples/`: Example scripts demonstrating library usage

## Debugging Tips

1. Use `pytest -v` for more verbose test output
2. For debugging workflows, enable debug logging:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
````

3. When debugging agents, monitor the workspace directory which contains logs and outputs
1. For API issues, check WebSocket connections and message formats

## Workflow Development Process

1. Define nodes with clear inputs, outputs, and properties
1. Create a graph connecting nodes according to data dependencies
1. Use the WorkflowRunner to execute the graph
1. Review results and debug if necessary
1. Optimize for performance as needed

## Agent Development Process

1. Define clear objectives and available tools
1. Create a new Agent instance with appropriate provider and model
1. Run the agent and monitor its planning and execution
1. Review output and refine as needed

## Comprehensive Environment Variables

### Core Configuration

| Variable      | Description             | Default       | Required |
| ------------- | ----------------------- | ------------- | -------- |
| `ENV`         | Environment name        | `development` | No       |
| `DEBUG`       | Enable debug mode       | `None`        | No       |
| `LOG_LEVEL`   | Logging level           | `INFO`        | No       |
| `REMOTE_AUTH` | Remote authentication   | `0`           | No       |
| `PYTHONPATH`  | Python path for imports | `src`         | No       |

### AI Providers & Language Models

| Variable              | Description                           | Group       | Required      |
| --------------------- | ------------------------------------- | ----------- | ------------- |
| `OPENAI_API_KEY`      | OpenAI API key for GPT models, DALL-E | LLM         | For OpenAI    |
| `ANTHROPIC_API_KEY`   | Anthropic API key for Claude models   | LLM         | For Anthropic |
| `GEMINI_API_KEY`      | Google Gemini API key                 | LLM         | For Gemini    |
| `HF_TOKEN`            | Hugging Face token for gated models   | HuggingFace | For HF        |
| `REPLICATE_API_TOKEN` | Replicate API token                   | Replicate   | For Replicate |
| `ELEVENLABS_API_KEY`  | ElevenLabs text-to-speech             | ElevenLabs  | For TTS       |
| `FAL_API_KEY`         | FAL.ai serverless AI                  | FAL         | For FAL       |
| `AIME_USER`           | Aime service username                 | Aime        | For Aime      |
| `AIME_API_KEY`        | Aime API key                          | Aime        | For Aime      |

### Database Configuration

| Variable            | Description              | Default                               |
| ------------------- | ------------------------ | ------------------------------------- |
| `DB_PATH`           | SQLite database path     | `~/.config/nodetool/nodetool.sqlite3` |
| `POSTGRES_DB`       | PostgreSQL database name | -                                     |
| `POSTGRES_USER`     | PostgreSQL username      | -                                     |
| `POSTGRES_PASSWORD` | PostgreSQL password      | -                                     |
| `POSTGRES_HOST`     | PostgreSQL host          | -                                     |
| `POSTGRES_PORT`     | PostgreSQL port          | -                                     |
| `SUPABASE_URL`      | Supabase project URL     | -                                     |
| `SUPABASE_KEY`      | Supabase service key     | -                                     |

### Storage & Assets

| Variable               | Description                    | Default     |
| ---------------------- | ------------------------------ | ----------- |
| `ASSET_BUCKET`         | S3 bucket for assets           | `images`    |
| `ASSET_TEMP_BUCKET`    | S3 bucket for temporary assets | -           |
| `ASSET_DOMAIN`         | Asset CDN domain               | -           |
| `ASSET_TEMP_DOMAIN`    | Temporary asset domain         | -           |
| `S3_ACCESS_KEY_ID`     | AWS access key ID              | -           |
| `S3_SECRET_ACCESS_KEY` | AWS secret access key          | -           |
| `S3_ENDPOINT_URL`      | S3 endpoint URL                | -           |
| `S3_REGION`            | S3 region                      | -           |
| `AWS_REGION`           | AWS region                     | `us-east-1` |

### Vector Database & Search

| Variable         | Description                   | Default                          |
| ---------------- | ----------------------------- | -------------------------------- |
| `CHROMA_PATH`    | ChromaDB storage path         | `~/.local/share/nodetool/chroma` |
| `CHROMA_URL`     | Remote ChromaDB URL           | -                                |
| `CHROMA_TOKEN`   | ChromaDB authentication token | -                                |
| `OLLAMA_API_URL` | Ollama API endpoint           | `http://127.0.0.1:11434`         |
| `OLLAMA_MODELS`  | Custom Ollama models path     | -                                |

### External Services & Integrations

| Variable                | Description                         | Group      |
| ----------------------- | ----------------------------------- | ---------- |
| `GOOGLE_MAIL_USER`      | Gmail address for email integration | Google     |
| `GOOGLE_APP_PASSWORD`   | Google app password                 | Google     |
| `SERPAPI_API_KEY`       | SerpAPI key for web scraping        | SerpAPI    |
| `DATA_FOR_SEO_LOGIN`    | DataForSEO login                    | DataForSEO |
| `DATA_FOR_SEO_PASSWORD` | DataForSEO password                 | DataForSEO |
| `BROWSER_URL`           | Browser automation endpoint         | Browser    |

### System Tools & Media Processing

| Variable       | Description                       | Default   |
| -------------- | --------------------------------- | --------- |
| `FFMPEG_PATH`  | Path to ffmpeg executable         | `ffmpeg`  |
| `FFPROBE_PATH` | Path to ffprobe executable        | `ffprobe` |
| `FONT_PATH`    | Font directory for text rendering | -         |
| `COMFY_FOLDER` | ComfyUI integration folder        | -         |

### Deployment & Infrastructure

| Variable           | Description                         | Default                 |
| ------------------ | ----------------------------------- | ----------------------- |
| `NODETOOL_API_URL` | NodeTool API base URL               | `http://localhost:8000` |
| `RUNPOD_API_KEY`   | RunPod API key for cloud deployment | -                       |
| `SENTRY_DSN`       | Sentry error tracking DSN           | -                       |
| `MEMCACHE_HOST`    | Memcache server host                | -                       |
| `MEMCACHE_PORT`    | Memcache server port                | -                       |
| `STATIC_FOLDER`    | Static files folder                 | `web/dist`              |
| `PORT`             | Server port                         | `8000`                  |

### Logging & Development

| Variable                     | Description               | Default             |
| ---------------------------- | ------------------------- | ------------------- |
| `NODETOOL_LOG_LEVEL`         | Global log level override | `INFO`              |
| `NODETOOL_LOG_FORMAT`        | Custom log format         | -                   |
| `NODETOOL_LOG_DATEFMT`       | Date format for logs      | `%Y-%m-%d %H:%M:%S` |
| `NO_COLOR`                   | Disable colored output    | -                   |
| `NODETOOL_CACHE_EXPIRY_DAYS` | Cache expiry days         | `7`                 |
| `EDITOR`                     | Default text editor       | `vi`                |
| `DOCKER_USERNAME`            | Docker registry username  | -                   |

### Chat & Deployment Specific

| Variable              | Description           | Default                 |
| --------------------- | --------------------- | ----------------------- |
| `CHAT_PROVIDER`       | Default chat provider | `ollama`                |
| `DEFAULT_MODEL`       | Default model name    | `gpt-oss:20b`           |
| `NODETOOL_TOOLS`      | Available tools list  | -                       |
| `RUNPOD_API_BASE_URL` | RunPod API base URL   | `https://api.runpod.io` |

## Important Configuration Notes

### Adding New Environment Variables

When adding new environment variables, use the `register_setting()` function in `src/nodetool/config/settings.py`:

```python
register_setting(
    package_name="nodetool",
    env_var="YOUR_NEW_VAR",
    group="YourGroup",
    description="Description of what this variable does",
    is_secret=True,  # True for API keys, False for config
)
```

### Environment File Hierarchy

The configuration system loads files in this specific order:

1. `DEFAULT_ENV` dict in `environment.py`
1. `.env` (base file)
1. `.env.{environment}` (development/test/production)
1. `.env.{environment}.local` (your secrets)
1. System environment variables
1. YAML settings/secrets files

### Testing Configuration

- Tests automatically use `ENV=test`
- Test database path is overridden to `/tmp/nodetool_test.db`
- Memory storage is used for assets in test mode

### Production Deployment

- Set `ENV=production`
- Use environment variables for all secrets
- Configure proper database (PostgreSQL/Supabase)
- Set up S3 storage with appropriate buckets
- Enable Sentry for error tracking

## Encrypted Secrets Management

NodeTool Core provides a secure encrypted secrets storage system with per-user encryption and database persistence.

### Overview

Secrets are stored in an encrypted database with the following features:
- **Per-user encryption** with unique derived keys
- **User isolation** - users cannot access each other's secrets
- **Encryption at rest** - all secrets encrypted using AES-256 (Fernet)
- **Secure key management** - master keys stored in system keychain or AWS Secrets Manager

### Architecture

#### Components

1. **Encryption Layer** (`src/nodetool/security/crypto.py`)
   - `SecretCrypto` class for encryption/decryption
   - Uses Fernet (AES-256 CBC with PKCS7 padding)
   - Per-user key derivation: master_key + user_id → derived key (PBKDF2-SHA256, 100k iterations)

2. **Master Key Management** (`src/nodetool/security/master_key.py`)
   - `MasterKeyManager` class with sources (in priority order):
     1. `SECRETS_MASTER_KEY` environment variable
     2. AWS Secrets Manager (if `AWS_SECRETS_MASTER_KEY_NAME` set)
     3. System keychain (macOS Keychain, Windows Credential Manager, Linux Secret Service)
     4. Auto-generated and stored in keychain

3. **Database Model** (`src/nodetool/models/secret.py`)
   - `Secret` model for encrypted secrets
   - Fields: id, user_id, key, encrypted_value, description, created_at, updated_at
   - Unique constraint on (user_id, key)
   - Methods: create, find, upsert, delete_secret, list_for_user, get_decrypted_value

4. **API Endpoints** (`src/nodetool/api/settings.py`)
   - `GET /api/settings/secrets` - List all possible secrets, marked as configured/unconfigured
   - `GET /api/settings/secrets/{key}` - Get specific secret (with optional decryption)
   - `PUT /api/settings/secrets/{key}` - Update or create secret
   - `DELETE /api/settings/secrets/{key}` - Delete secret
   - All endpoints require authentication via `Depends(current_user)`

5. **Helper Functions** (`src/nodetool/security/secret_helper.py`)
   - `get_secret(key, user_id)` - Get secret with fallback to environment
   - `get_secret_required(key, user_id)` - Get or raise exception
   - `get_secret_sync(key)` - Synchronous env-only lookup
   - `has_secret(key, user_id)` - Check if secret exists

6. **AWS Utility** (`src/nodetool/security/aws_secrets_util.py`)
   - CLI tool for managing master keys in AWS Secrets Manager
   - Commands: store, retrieve, generate, delete

### Secret Resolution Priority

When `get_secret(key, user_id)` is called, sources are checked in order:

1. **Environment Variable** (highest priority)
   - `os.environ.get(key)` - System-wide, not user-specific
   - Useful for CI/CD, Docker deployments

2. **Encrypted Database** (recommended)
   - Query `Secret` model for user_id + key
   - Per-user, encrypted at rest
   - Managed via API or helper functions

3. **secrets.yaml** (deprecated, no longer used for new secrets)

### Usage Examples

#### For Application Developers

**Get a secret at runtime:**
```python
from nodetool.security import get_secret

# In an async function with user context
async def my_function(user_id: str):
    api_key = await get_secret("OPENAI_API_KEY", user_id)
    if api_key:
        # Use the API key
        ...
```

**Get a required secret (raises if not found):**
```python
from nodetool.security import get_secret_required

async def my_function(user_id: str):
    api_key = await get_secret_required("OPENAI_API_KEY", user_id)
    # Guaranteed to have a value or exception is raised
```

**For system-wide secrets (env only):**
```python
from nodetool.security import get_secret_sync

# Synchronous, checks only environment variables
api_key = get_secret_sync("SYSTEM_API_KEY", default="default_value")
```

#### For End Users (via API)

**List all possible secrets:**
```bash
curl http://localhost:8000/api/settings/secrets \
  -H "Authorization: Bearer <token>"
```

**Set or update a secret:**
```bash
curl -X PUT http://localhost:8000/api/settings/secrets/OPENAI_API_KEY \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"value": "sk-...", "description": "My OpenAI API key"}'
```

**Get a secret (decrypted):**
```bash
curl "http://localhost:8000/api/settings/secrets/OPENAI_API_KEY?decrypt=true" \
  -H "Authorization: Bearer <token>"
```

**Delete a secret:**
```bash
curl -X DELETE http://localhost:8000/api/settings/secrets/OPENAI_API_KEY \
  -H "Authorization: Bearer <token>"
```

### Security Features

#### Encryption
- **Algorithm**: Fernet (AES-128 CBC + HMAC-SHA256)
- **Key Derivation**: PBKDF2-SHA256 with 100,000 iterations
- **Per-user salt**: Each user's secrets encrypted with unique derived key
- **Master key**: Stored in system keychain or AWS Secrets Manager

#### Authentication
- **Development**: Defaults to user_id="1" (no auth)
- **Production**: JWT authentication via Supabase
- **User isolation**: Database queries filtered by authenticated user

#### Access Control
- Users can only access their own secrets
- API endpoints require authentication (production)
- Read/write operations are user-scoped

### Production Deployment

#### Using AWS Secrets Manager

1. **Generate master key:**
```bash
python -m nodetool.security.aws_secrets_util generate \
  --secret-name nodetool-master-key \
  --region us-east-1
```

2. **Configure environment:**
```bash
export AWS_SECRETS_MASTER_KEY_NAME=nodetool-master-key
export AWS_REGION=us-east-1
```

3. **Verify:**
```bash
python -m nodetool.security.aws_secrets_util retrieve \
  --secret-name nodetool-master-key
```

#### Using Environment Variable

```bash
# Generate key
export SECRETS_MASTER_KEY=$(python -c "from nodetool.security.crypto import SecretCrypto; print(SecretCrypto.generate_master_key())")

# Backup key securely!
echo $SECRETS_MASTER_KEY > /secure/location/master_key.txt
```

### Testing

```bash
# Test crypto utilities
pytest tests/security/test_crypto.py -v

# Test master key management
pytest tests/security/test_master_key.py -v

# Test Secret model
pytest tests/models/test_secret.py -v

# Test helper functions
pytest tests/security/test_secret_helper.py -v

# Test API endpoints
pytest tests/api/test_settings_api.py -v
```

### Dependencies

Secrets management requires:
```toml
dependencies = [
    "cryptography>=43.0.0",
    "keyring>=25.5.0",
]
```
