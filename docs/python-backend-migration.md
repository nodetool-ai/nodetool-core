# Long-term: Slim nodetool-core to Python Node Runtime

This document describes the phased plan to slim `nodetool-core` down from a full-stack Python backend to a focused Python node execution runtime. The TS backend (`nodetool/packages/`) already replaces most of nodetool-core's functionality.

## What STAYS (the "keep set")

These modules are needed for Python node execution permanently:

| Module | Why |
|--------|-----|
| `workflows/base_node.py` | BaseNode class, NODE_BY_TYPE registry |
| `workflows/processing_context.py` | ProcessingContext (nodes call `ctx.*`) |
| `metadata/types.py` | AssetRef, ImageRef, AudioRef, VideoRef, Model3DRef |
| `metadata/node_metadata.py` | NodeMetadata, PackageModel |
| `runtime/resources.py` | ResourceScope |
| `packages/registry.py` → extract `discovery.py` | `discover_node_packages()` only |
| `worker/` | The merged worker |
| `media/` | ffmpeg, PIL, audio processing used by nodes |
| `config/logging_config.py` | Logging setup |

## What GOES (replaced by TS backend)

| Module | TS Replacement |
|--------|---------------|
| `api/` (FastAPI server) | `@nodetool/websocket` |
| `chat/` | `@nodetool/chat` |
| `agents/` | `@nodetool/agents` |
| `providers/` | `@nodetool/runtime/providers` |
| `models/` | `@nodetool/models` |
| `storage/` | `@nodetool/storage` |
| `security/` | `@nodetool/auth` + `@nodetool/security` |
| `dsl/` | TS codegen |
| `cli.py` | `@nodetool/cli` |
| `io/`, `html/`, `gateway/`, `deploy/`, `messaging/` | Various TS packages |
| `workflows/workflow_runner.py`, `actor.py` | `@nodetool/kernel` |

## Python-only Node Packages

Only these 4 packages need the Python worker long-term. The rest (Replicate, FAL, ElevenLabs, Comfy) are API-based and belong in TS.

| Package | Nodes | Key ctx methods used |
|---------|-------|---------------------|
| **nodetool-huggingface** | 31 | `image_from_pil/bytes/numpy/tensor`, `image_to_pil/numpy/tensor`, `audio_from/to_numpy`, `asset_to_bytes/io`, `video_from_frames/numpy`, `dataframe_to_pandas`, `device`, `model` (ModelManager), `get_secret`, `post_message`, `is_huggingface_model_cached` |
| **nodetool-mlx** | 5 | `image_from/to_pil`, `audio_from/to_numpy`, `audio_to_audio_segment`, `post_message` |
| **nodetool-apple** | 15 | `asset_to_bytes`, `create_asset`, `image_from_bytes`, `text_from_str` |
| **nodetool-whispercpp** | 1 | `audio_to_audio_segment` |

## BaseProcessingContext — Required ABC Surface

Based on actual usage across the 4 Python-only packages:

- **Secrets:** `get_secret()`, `get_secret_required()`
- **Images:** `image_from_pil()`, `image_from_bytes()`, `image_from_numpy()`, `image_from_tensor()`, `image_to_pil()`, `image_to_numpy()`, `image_to_tensor()`
- **Audio:** `audio_from_numpy()`, `audio_to_numpy()`, `audio_to_audio_segment()`
- **Video:** `video_from_frames()`, `video_from_numpy()`
- **Assets:** `asset_to_bytes()`, `asset_to_io()`, `create_asset()`, `text_from_str()`
- **Data:** `dataframe_to_pandas()`
- **ML:** `device` property, `model` property (ModelManager), `is_huggingface_model_cached()`
- **Lifecycle:** `post_message()`, `is_cancelled`

## Removal Strategy (phased)

### Phase 1 — Extract BaseProcessingContext (prerequisite)

The blocker: `ProcessingContext` (3600 lines) imports from `models/`, `storage/`, `security/`, `config/`. Python nodes only use a thin slice (~30 methods listed above).

Create `workflows/base_context.py` — an ABC with only those methods. Make `WorkerContext` inherit from `BaseProcessingContext` instead of `ProcessingContext`. This breaks the dependency chain.

Note: `model` (ModelManager) and `is_huggingface_model_cached()` are the trickiest — they pull in HF integration code. May need to inject ModelManager as an optional capability rather than baking it into the ABC.

### Phase 2 — Remove leaf modules

No other code depends on these:
- `api/`, `chat/`, `agents/`, `dsl/`, `cli.py`, `html/`, `gateway/`, `deploy/`

### Phase 3 — Remove mid-level modules

After Phase 2, nothing references these:
- `providers/`, `integrations/`, `messaging/`, `indexing/`, `tools/`

### Phase 4 — Remove core modules

After ProcessingContext decoupling:
- `models/`, `storage/`, `security/`, `runtime/db_*.py`
- Simplify `config/` to just logging + env loading

### Phase 5 — Slim dependencies in pyproject.toml

**Remove:** `fastapi`, `uvicorn`, `supabase`, `boto3`, `openai`, `anthropic`, `ollama`, `chromadb`, `langchain-*`, `llama-index-*`, `docker`, `paramiko`, `keyring`, `cryptography`, etc.

**Keep:** `pydantic`, `websockets`, `msgpack`, `numpy`, `pillow`, `aiohttp`, `httpx`

## Risks

1. **ProcessingContext entanglement** — The biggest risk. Must audit which `ctx.*` methods all Python node packages (huggingface, mlx, base, fal, etc.) actually call before defining the ABC.
2. **ResourceScope** — Currently tries to acquire DB connections. Worker doesn't need this. May need a `WorkerResourceScope` shim.
3. **registry.py imports** — `discover_node_packages()` is clean but the module imports `Environment`, `httpx`, `click`, etc. Extract to standalone `discovery.py` before removing those deps.
4. **Atomicity** — TS bridge module path change (`nodetool_worker` → `nodetool.worker`) must ship with the merge. Add a compatibility `nodetool_worker` shim during transition if needed.
