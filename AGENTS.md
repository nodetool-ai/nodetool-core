# nodetool-core — Agent Guide

Guidance for coding agents and contributors working in this repository.
`CLAUDE.md` only includes this file.

## What This Repository Is

nodetool-core is a **Python library and node runner** for the NodeTool platform. The TypeScript server handles HTTP API, WebSocket, database, auth, agents, chat, storage, deploy, and workflow orchestration. Python remains for two roles:

1. **Node runner subprocess** — TS spawns `python -m nodetool.worker --stdio` and exchanges MessagePack messages over stdin/stdout (or connects over WebSocket to a remote worker named by `NODETOOL_WORKER_URL`) for `discover`/`execute`/`cancel`/`provider.*`/`models.*`/`comfy.*` (the `comfy.*` messages proxy a co-located ComfyUI server — see `docs/comfy-proxy.md`)
2. **Node system and type definitions** — `BaseNode`, `ProcessingContext`, metadata types, used by all Python node packages

Cloud/API providers (OpenAI, Anthropic, Gemini, Ollama, etc.) are implemented in the TS server. Python only has local-compute providers (HuggingFace local, MLX) registered via external packages.

## Code Organization

```
src/nodetool/
├── config/            # Environment, logging, settings
├── integrations/      # HuggingFace models
├── io/                # URI utilities, media fetch
├── media/             # Audio, image, video processing helpers
├── metadata/          # Type definitions, node metadata, tool_types
├── ml/                # Model management
├── package_metadata/  # Package metadata JSON (nodetool-core.json)
├── package_tools/     # Package registry scanning (nodetool-pkg CLI)
├── providers/         # Provider base classes, registry
├── runtime/           # ResourceScope, DB connection pools
├── security/          # Secret helper (secrets come from the environment)
├── storage/           # Abstract storage, memory/file/S3 backends
├── types/             # API graph types, prediction types
├── utils/             # Misc utilities
├── worker/            # Worker subprocess (stdio and WebSocket servers, executor)
└── workflows/         # Node execution core (see below)
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
- `asset_storage.py` — Asset ref utilities (content type, auto-save)
- `io.py` — Node input/output helpers

### What Was Removed

The following were moved to the TypeScript server and deleted from Python:

- `api/`, `chat/`, `agents/`, `messaging/`, `tools/`, `deploy/`, `proxy/`, `system/`, `ui/`, `html/`, `gateway/`, `indexing/`, `migrations/`, `code_runners/`, `observability/`
- `cli.py`, `cli_migrations.py`
- Workflow orchestration: `workflow_runner.py`, `actor.py`, `job_execution.py`, `run_workflow.py`, `checkpoint_manager.py`, `state_manager.py`, etc.
- Cloud provider implementations: `openai_provider.py`, `anthropic_provider.py`, `gemini_provider.py`, `ollama_provider.py`, etc.
- Vector stores: `integrations/vectorstores/chroma/` — the TS `packages/vectorstore/` carries `chroma-client.ts` and `embedding.ts`
- Secret encryption at rest: `security/crypto.py`, `security/master_key.py` — the worker now
  reads secrets from the environment only (`security/secret_helper.py`)

## Development Setup

Install into the conda env named `nodetool`. The TypeScript server's Python
bridge (`packages/runtime/src/python-stdio-bridge.ts` in the nodetool repo)
uses `NODETOOL_PYTHON` when set. Otherwise it uses the active `CONDA_PREFIX`
when that env is named `nodetool`, then looks for `envs/nodetool` under
`~/miniconda3` or `~/anaconda3` and the desktop app's managed env. A project
`.venv` is never found automatically.

```bash
conda create -n nodetool python=3.11 pandoc ffmpeg -c conda-forge
conda activate nodetool
uv pip install -e ".[dev]"   # uv pip targets the active conda env
```

Node packs go into the same env, for example
`uv pip install -e ../nodetool-huggingface`. To run the worker from another
interpreter, start the server with `NODETOOL_PYTHON=/path/to/python`.

`uv sync`, `uv run` and the `make` targets use a separate project `.venv`
pinned by `uv.lock`. CI runs `uv sync --locked --all-extras --dev`, so run
`uv lock` and commit `uv.lock` whenever `pyproject.toml` dependencies change.
Do not point `uv sync` at the conda env (`UV_PROJECT_ENVIRONMENT`): it removes
packages that are not in the lock, including installed node packs.

## Common Commands

Run these in the activated `nodetool` env:

```bash
pytest -q                              # all tests
pytest tests/path/to/test_file.py      # one file
ruff check .                           # lint
ruff format .                          # format
nodetool-pkg scan --write              # regenerate src/nodetool/package_metadata/nodetool-core.json
```

CI checks that `nodetool-pkg scan --write` leaves the metadata file unchanged.
Run it after changing the version or `[project]` metadata.

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

`providers/base.py` has `BaseProvider`, `register_provider`, `get_registered_provider`. External packages (nodetool-mlx, nodetool-huggingface) register local-compute providers. `LOCAL_PROVIDER_MODULES` lists their modules. `import_provider_module` skips a pack that is not installed and logs any other import failure with the module name. The worker's `provider_handler.py` exposes the providers to TS over the worker connection.

### Node Discovery

`worker/node_loader.py` loads the union of two sources: entry points in group
`nodetool.namespaces` (each value is a comma-separated list of namespace names,
such as `huggingface`) and the subdirectories of every `nodetool/nodes` path on
`sys.path`.

### Tool Type

`metadata/tool_types.py` has the `Tool` class used by provider function-calling interfaces. Relocated from the deleted `agents/tools/base.py`.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENV` | Environment name | `development` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `HF_TOKEN` | HuggingFace token | - |
| `DB_PATH` | SQLite database path | `~/.local/share/nodetool/nodetool.sqlite3` on Linux and macOS, `%APPDATA%\nodetool\nodetool.sqlite3` on Windows |
| `NODETOOL_WORKER_HOST` / `NODETOOL_WORKER_PORT` | WebSocket worker bind address | `127.0.0.1` / `0` (any free port) |
| `NODETOOL_WORKER_TOKEN` | Bearer token required by the WebSocket worker | unset (no auth) |
| `NODETOOL_TORCH_DEVICE` | Force the torch device: `cpu`, `mps`, `cuda` or `cuda:<index>`. An unavailable device logs a warning and falls back to automatic selection | automatic: MPS, then CUDA, then CPU |
| `PYTORCH_ENABLE_MPS_FALLBACK` | Run ops without an MPS kernel on the CPU. Set by `nodetool.worker` when unset | `1` |
| `COMFYUI_URL` | ComfyUI server proxied by the worker's `comfy.*` messages | `http://127.0.0.1:8188` |
| `COMFY_MODELS_DIR` | Model tree for `comfy.models.*` (RunPod network volume) | `/workspace/models` |

## Testing

Tests are in `tests/` mirroring `src/` structure. Key test directories:

- `tests/worker/` — Worker subprocess tests
- `tests/workflows/` — Node execution, processing context, graph tests
- `tests/security/` — Secret helper tests
- `tests/integrations/` — HuggingFace model detection, safetensors
- `tests/storage/` — Storage backend tests

## Commits and Pull Requests

Use Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`,
`chore:`) with imperative, scoped messages. Before committing, run
`ruff check .` and `pytest -q`, and `make typecheck` when you change types.
