[← Back to Docs Index](docs/index.md)

# Repository Guidelines

## Project Structure & Module Organization

- Source: `src/nodetool/` (e.g., `agents/`, `api/`, `chat/`, `common/`, `dsl/`, `workflows/`).
- Tests: `tests/` mirrors the source layout (e.g., `tests/agents`, `tests/api`).
- Docs and examples: `docs/`, `examples/`.
- Packaging: Hatch project (`pyproject.toml`), console entry `nodetool`.

## Build, Test, and Development Commands

### ⚠️ Python Environment (IMPORTANT)

**Local Development:** Use the conda `nodetool` environment. Do not use system Python.

```bash
# Option 1: Activate the environment first
conda activate nodetool
python -m pytest tests/...

# Option 2: Use conda run (preferred for scripts/agents)
conda run -n nodetool python -m pytest tests/...
```

**GitHub CI / Copilot Agent:** Uses standard Python 3.11 with pip. Dependencies are pre-installed via `.github/workflows/copilot-setup-steps.yml`. Run commands directly:

```bash
pytest -v
pip install -e .
```

### Commands

- Install dependencies: `pip install . && pip install -r requirements-dev.txt`
- Run tests: `pytest -q` (quick) or `pytest -v` (verbose)
- Run specific tests: `pytest tests/path/to/test_file.py`
- Test with coverage: `pytest --cov=src`
- Lint and format: `ruff check .`, `black .`, `mypy .`

## Coding Style & Naming Conventions

- Language: Python 3.11, type hints required for new/changed code.
- Formatting: Black style; keep imports and whitespace tidy; prefer f‑strings.
- Linting: Ruff for quick rules; Flake8/Mypy/Pylint configs exist for CI parity.
- Names: `snake_case` for files/functions, `PascalCase` for classes, `SCREAMING_SNAKE_CASE` for constants.
- Modules: keep public APIs under `src/nodetool/...` with small, focused modules.

## Testing Guidelines

- Framework: pytest; locate tests under `tests/` with structure mirroring `src/`.
- Naming: files `test_*.py`, functions `test_*`, classes `Test*` (no `__init__`).
- Running: `pytest -q` for quick checks; add fixtures in `tests/conftest.py`.
- Scope: include unit tests for logic and lightweight integration tests for I/O and async paths.
- Environment: Tests automatically use `ENV=test` with in-memory storage and `/tmp/nodetool_test.db`.
- Debugging: Use `pytest -v` for verbose output, enable debug logging for workflows.

## Commit & Pull Request Guidelines

- Commits: follow Conventional Commits (`feat:`, `fix:`, `refactor:`, etc.); keep messages imperative and scoped.
- PRs: include a clear description, linked issues, and screenshots/logs if UI/CLI behavior changes.
- Checks: ensure `pytest`, `ruff`, `black`, and `mypy` pass locally; update docs/examples when APIs change.

## Security & Configuration Tips

- Do not commit secrets; use environment-specific `.env` files with `.local` overrides for actual secrets.
- The system uses a layered configuration approach: defaults → base `.env` → environment-specific → local overrides →
  environment variables → YAML settings.
- Prefer async I/O where supported (many subsystems are async) and avoid blocking calls in hot paths.

## Environment Configuration

### Environment Files Structure

- `.env.example` - Template with all configuration options (committed)
- `.env.development` - Development defaults (committed, no secrets)
- `.env.test` - Test environment configuration (committed)
- `.env.production` - Production template (committed, no secrets)
- `.env.*.local` - Local overrides with actual API keys (gitignored)

### Key Environment Variables by Category

#### Core Configuration

- `ENV` - Environment name (`development`, `test`, `production`)
- `DEBUG` - Enable debug mode
- `LOG_LEVEL` - Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `REMOTE_AUTH` - Enable remote authentication (`0`/`1`)

#### AI Providers & APIs

- `OPENAI_API_KEY` - OpenAI API key for GPT models, DALL-E
- `ANTHROPIC_API_KEY` - Anthropic API key for Claude models
- `GEMINI_API_KEY` - Google Gemini API key
- `HF_TOKEN` - Hugging Face token for gated models
- `REPLICATE_API_TOKEN` - Replicate API token
- `ELEVENLABS_API_KEY` - ElevenLabs text-to-speech API key
- `FAL_API_KEY` - FAL.ai serverless AI infrastructure key
- `AIME_USER` / `AIME_API_KEY` - Aime service credentials

#### Database & Storage

- `DB_PATH` - SQLite database path (default: `~/.config/nodetool/nodetool.sqlite3`)
- `POSTGRES_*` - PostgreSQL connection parameters
- `SUPABASE_URL` / `SUPABASE_KEY` - Supabase configuration
- `ASSET_BUCKET` / `ASSET_TEMP_BUCKET` - S3 storage buckets
- `S3_*` - S3 configuration (access keys, endpoint, region)

#### Vector Database & AI Services

- `CHROMA_PATH` - ChromaDB storage path (default: `~/.local/share/nodetool/chroma`)
- `CHROMA_URL` / `CHROMA_TOKEN` - Remote ChromaDB configuration
- `OLLAMA_API_URL` - Ollama API endpoint (default: `http://127.0.0.1:11434`)

#### External Integrations

- `GOOGLE_MAIL_USER` / `GOOGLE_APP_PASSWORD` - Gmail integration
- `SERPAPI_API_KEY` - SerpAPI for web scraping
- `DATA_FOR_SEO_*` - DataForSEO credentials
- `BROWSER_URL` - Browser automation endpoint

#### System & Media Processing

- `FFMPEG_PATH` / `FFPROBE_PATH` - Media processing tools
- `FONT_PATH` - Font directory for text rendering
- `COMFY_FOLDER` - ComfyUI integration folder

#### Deployment & Monitoring

- `NODETOOL_API_URL` - NodeTool API base URL
- `RUNPOD_API_KEY` - RunPod cloud deployment
- `SENTRY_DSN` - Error tracking
- `MEMCACHE_HOST` / `MEMCACHE_PORT` - Caching

### Setup Example

```bash
# Copy template and add your secrets
cp .env.example .env.development.local

# Edit with your API keys
vim .env.development.local
```

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

## Architecture Overview

### Core Components

- **Workflow System** (`src/nodetool/workflows/`) - DAG-based workflow execution
- **Agent System** (`src/nodetool/agents/`) - LLM task planning and execution
- **Chat System** (`src/nodetool/chat/`) - AI provider integrations
- **Storage System** (`src/nodetool/storage/`) - Multi-backend data persistence
- **API Layer** (`src/nodetool/api/`) - FastAPI server with WebSocket support
- **Models Layer** (`src/nodetool/models/`) - Database adapters and schemas

### Key Design Patterns

- **Dependency Injection** - Components receive dependencies through constructors
- **Asynchronous Processing** - Heavy use of asyncio for non-blocking operations
- **Factory Pattern** - Provider factories create appropriate implementations
- **Strategy Pattern** - Different backends implement common interfaces
- **Observer Pattern** - WebSocket updates for real-time progress tracking

## Development Workflows

### Workflow Development

1. Define nodes with clear inputs, outputs, and properties
1. Create graphs connecting nodes by data dependencies
1. Use WorkflowRunner to execute graphs
1. Monitor execution via WebSocket updates
1. Debug with verbose logging (`logging.basicConfig(level=logging.DEBUG)`)

### Agent Development

1. Define objectives and available tools
1. Create Agent instances with appropriate providers
1. Monitor planning and execution in workspace directory
1. Review outputs and refine tool usage
1. Test with different models and configurations
