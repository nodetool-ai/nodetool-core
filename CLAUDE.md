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
2. Base `.env` file (if exists)
3. Environment-specific file (`.env.development`, `.env.test`, or `.env.production`)
4. Local override file (`.env.{environment}.local`)
5. System environment variables
6. YAML settings/secrets files

### Environment Detection

The system determines the environment from:

1. `ENV` environment variable 
2. `PYTEST_CURRENT_TEST` (automatically sets test environment)
3. Defaults to "development"

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

```bash
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
   ```
3. When debugging agents, monitor the workspace directory which contains logs and outputs
4. For API issues, check WebSocket connections and message formats

## Workflow Development Process

1. Define nodes with clear inputs, outputs, and properties
2. Create a graph connecting nodes according to data dependencies
3. Use the WorkflowRunner to execute the graph
4. Review results and debug if necessary
5. Optimize for performance as needed

## Agent Development Process

1. Define clear objectives and available tools
2. Create a new Agent instance with appropriate provider and model
3. Run the agent and monitor its planning and execution
4. Review output and refine as needed

## Comprehensive Environment Variables

### Core Configuration
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENV` | Environment name | `development` | No |
| `DEBUG` | Enable debug mode | `None` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `REMOTE_AUTH` | Remote authentication | `0` | No |
| `PYTHONPATH` | Python path for imports | `src` | No |

### AI Providers & Language Models
| Variable | Description | Group | Required |
|----------|-------------|-------|----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models, DALL-E | LLM | For OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude models | LLM | For Anthropic |
| `GEMINI_API_KEY` | Google Gemini API key | LLM | For Gemini |
| `HF_TOKEN` | Hugging Face token for gated models | HuggingFace | For HF |
| `REPLICATE_API_TOKEN` | Replicate API token | Replicate | For Replicate |
| `ELEVENLABS_API_KEY` | ElevenLabs text-to-speech | ElevenLabs | For TTS |
| `FAL_API_KEY` | FAL.ai serverless AI | FAL | For FAL |
| `AIME_USER` | Aime service username | Aime | For Aime |
| `AIME_API_KEY` | Aime API key | Aime | For Aime |

### Database Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `DB_PATH` | SQLite database path | `~/.config/nodetool/nodetool.sqlite3` |
| `POSTGRES_DB` | PostgreSQL database name | - |
| `POSTGRES_USER` | PostgreSQL username | - |
| `POSTGRES_PASSWORD` | PostgreSQL password | - |
| `POSTGRES_HOST` | PostgreSQL host | - |
| `POSTGRES_PORT` | PostgreSQL port | - |
| `SUPABASE_URL` | Supabase project URL | - |
| `SUPABASE_KEY` | Supabase service key | - |

### Storage & Assets
| Variable | Description | Default |
|----------|-------------|---------|
| `ASSET_BUCKET` | S3 bucket for assets | `images` |
| `ASSET_TEMP_BUCKET` | S3 bucket for temporary assets | - |
| `ASSET_DOMAIN` | Asset CDN domain | - |
| `ASSET_TEMP_DOMAIN` | Temporary asset domain | - |
| `S3_ACCESS_KEY_ID` | AWS access key ID | - |
| `S3_SECRET_ACCESS_KEY` | AWS secret access key | - |
| `S3_ENDPOINT_URL` | S3 endpoint URL | - |
| `S3_REGION` | S3 region | - |
| `AWS_REGION` | AWS region | `us-east-1` |

### Vector Database & Search
| Variable | Description | Default |
|----------|-------------|---------|
| `CHROMA_PATH` | ChromaDB storage path | `~/.local/share/nodetool/chroma` |
| `CHROMA_URL` | Remote ChromaDB URL | - |
| `CHROMA_TOKEN` | ChromaDB authentication token | - |
| `OLLAMA_API_URL` | Ollama API endpoint | `http://127.0.0.1:11434` |
| `OLLAMA_MODELS` | Custom Ollama models path | - |

### External Services & Integrations
| Variable | Description | Group |
|----------|-------------|-------|
| `GOOGLE_MAIL_USER` | Gmail address for email integration | Google |
| `GOOGLE_APP_PASSWORD` | Google app password | Google |
| `SERPAPI_API_KEY` | SerpAPI key for web scraping | SerpAPI |
| `DATA_FOR_SEO_LOGIN` | DataForSEO login | DataForSEO |
| `DATA_FOR_SEO_PASSWORD` | DataForSEO password | DataForSEO |
| `BROWSER_URL` | Browser automation endpoint | Browser |

### System Tools & Media Processing
| Variable | Description | Default |
|----------|-------------|---------|
| `FFMPEG_PATH` | Path to ffmpeg executable | `ffmpeg` |
| `FFPROBE_PATH` | Path to ffprobe executable | `ffprobe` |
| `FONT_PATH` | Font directory for text rendering | - |
| `COMFY_FOLDER` | ComfyUI integration folder | - |

### Deployment & Infrastructure
| Variable | Description | Default |
|----------|-------------|---------|
| `NODETOOL_API_URL` | NodeTool API base URL | `http://localhost:8000` |
| `RUNPOD_API_KEY` | RunPod API key for cloud deployment | - |
| `SENTRY_DSN` | Sentry error tracking DSN | - |
| `MEMCACHE_HOST` | Memcache server host | - |
| `MEMCACHE_PORT` | Memcache server port | - |
| `STATIC_FOLDER` | Static files folder | `web/dist` |
| `PORT` | Server port | `8000` |

### Logging & Development
| Variable | Description | Default |
|----------|-------------|---------|
| `NODETOOL_LOG_LEVEL` | Global log level override | `INFO` |
| `NODETOOL_LOG_FORMAT` | Custom log format | - |
| `NODETOOL_LOG_DATEFMT` | Date format for logs | `%Y-%m-%d %H:%M:%S` |
| `NO_COLOR` | Disable colored output | - |
| `NODETOOL_CACHE_EXPIRY_DAYS` | Cache expiry days | `7` |
| `EDITOR` | Default text editor | `vi` |
| `DOCKER_USERNAME` | Docker registry username | - |

### Chat & Deployment Specific
| Variable | Description | Default |
|----------|-------------|---------|
| `CHAT_PROVIDER` | Default chat provider | `ollama` |
| `DEFAULT_MODEL` | Default model name | `gpt-oss:20b` |
| `NODETOOL_TOOLS` | Available tools list | - |
| `RUNPOD_API_BASE_URL` | RunPod API base URL | `https://api.runpod.io` |

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
2. `.env` (base file)
3. `.env.{environment}` (development/test/production)
4. `.env.{environment}.local` (your secrets)
5. System environment variables
6. YAML settings/secrets files

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
