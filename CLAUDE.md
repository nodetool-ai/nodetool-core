# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## What This Repository Is

nodetool-core is a **Python library and node runner** for the NodeTool platform. The TypeScript server handles HTTP API, WebSocket, database, auth, agents, chat, storage, deploy, and workflow orchestration. Python remains for two roles:

1. **Node runner subprocess** — TS spawns `python -m nodetool.worker`, connects via WebSocket+MessagePack for `discover`/`execute`/`cancel`/`provider.*`
2. **Node system and type definitions** — `BaseNode`, `ProcessingContext`, metadata types, used by all Python node packages

Cloud/API providers (OpenAI, Anthropic, Gemini, Ollama, etc.) are implemented in the TS server. Python only has local-compute providers (HuggingFace local, MLX) registered via external packages.

## Code Organization

```
src/nodetool/
├── config/          # Environment, logging, settings
├── dsl/             # DSL for graph construction (codegen, handles)
├── integrations/    # HuggingFace models, vector stores
├── io/              # URI utilities, media fetch
├── media/           # Audio, image, video processing helpers
├── metadata/        # Type definitions, node metadata, tool_types
├── ml/              # Model management, TTS/ASR/image model lists
├── models/          # Database models (Asset, Job, Secret, etc.)
├── nodes/           # Built-in node implementations
├── packages/        # Package registry and scanning
├── providers/       # Provider base classes, registry, fake provider
├── runtime/         # ResourceScope, DB connection pools
├── security/        # Crypto, master key, secret helper, auth providers
├── storage/         # Abstract storage, memory/file/S3 backends
├── types/           # API graph types, prediction types
├── utils/           # Misc utilities
├── worker/          # Worker subprocess (WebSocket server, executor)
└── workflows/       # Node execution core (see below)
```

### workflows/ — Node Execution Core

These files support node execution. There is no workflow runner or orchestration — that's in TS.

- `base_node.py` — `BaseNode` class, all nodes inherit from this
- `processing_context.py` — `ProcessingContext` for node execution (media helpers, secrets, asset storage)
- `types.py` — `Chunk`, `NodeProgress`, `NodeUpdate`, etc.
- `graph.py` — Graph representation (nodes + edges)
- `inbox.py`, `channel.py` — Message passing between nodes
- `memory_utils.py` — GPU/CPU memory tracking, garbage collection
- `processing_offload.py` — Thread offloading for CPU-bound work
- `torch_support.py` — PyTorch device management
- `property.py` — Node property descriptors
- `recommended_models.py` — Model recommendations per node
- `asset_storage.py` — Asset ref utilities (content type, auto-save)
- `io.py` — Node input/output helpers

### What Was Removed

The following were moved to the TypeScript server and deleted from Python:

- `api/`, `chat/`, `agents/`, `messaging/`, `tools/`, `deploy/`, `proxy/`, `system/`, `ui/`, `html/`, `gateway/`, `indexing/`, `migrations/`, `code_runners/`, `observability/`
- `cli.py`, `cli_migrations.py`
- Workflow orchestration: `workflow_runner.py`, `actor.py`, `job_execution.py`, `run_workflow.py`, `checkpoint_manager.py`, `state_manager.py`, etc.
- Cloud provider implementations: `openai_provider.py`, `anthropic_provider.py`, `gemini_provider.py`, `ollama_provider.py`, etc.

## Development Setup

```bash
conda create -n nodetool python=3.11 pandoc ffmpeg -c conda-forge
conda activate nodetool
uv sync
uv sync --group dev
```

## Common Commands

```bash
# Run all tests
uv run pytest -q

# Run specific test file
uv run pytest tests/path/to/test_file.py

# Lint
uv run ruff check .

# Format
uv run ruff format .
```

## Key Patterns

### Python Node Development

```python
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

class MyNode(BaseNode):
    """
    Brief description
    tags, keywords, for, search
    """
    input_field: str = ""

    async def process(self, context: ProcessingContext) -> str:
        return self.input_field.upper()
```

### ProcessingContext Methods Available to Nodes

Media conversion: `image_to_pil`, `image_from_pil`, `image_from_bytes`, `image_from_tensor`, `audio_from_numpy`, `audio_to_numpy`, `video_from_frames`, `video_from_numpy`, `text_from_str`, `asset_to_io`, `asset_to_bytes`, `dataframe_to_pandas`, `dataframe_from_pandas`

Secrets: `get_secret`, `get_secret_required`

Communication: `post_message`, `has_messages`, `pop_message_async`

Properties: `device` (torch device), `is_cancelled`, `user_id`, `workflow_id`

Storage: `create_asset`, `download_asset`, `asset_storage_url`

### Provider Infrastructure

`providers/base.py` has `BaseProvider`, `register_provider`, `get_registered_provider`. External packages (nodetool-mlx, nodetool-huggingface) register local-compute providers. The worker's `provider_handler.py` exposes them to TS via WebSocket.

### Tool Type

`metadata/tool_types.py` has the `Tool` class used by provider function-calling interfaces. Relocated from the deleted `agents/tools/base.py`.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENV` | Environment name | `development` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `HF_TOKEN` | HuggingFace token | - |
| `DB_PATH` | SQLite database path | `~/.config/nodetool/nodetool.sqlite3` |
| `FFMPEG_PATH` | Path to ffmpeg | `ffmpeg` |
| `SECRETS_MASTER_KEY` | Master key for encrypted secrets | auto-generated |

## Testing

Tests are in `tests/` mirroring `src/` structure. Key test directories:

- `tests/worker/` — Worker subprocess tests
- `tests/workflows/` — Node execution, processing context, graph tests
- `tests/models/` — Database model tests
- `tests/security/` — Crypto, secret helper tests
- `tests/providers/` — Cost calculator, provider infrastructure tests

Note: `tests/io/test_media_fetch.py` has a pre-existing mock recursion timeout — skip with `--ignore=tests/io/test_media_fetch.py` if needed.
