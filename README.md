# NodeTool Core

<h3>Python Node System & Worker</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue.svg" alt="Python Version Badge">
  <img src="https://img.shields.io/badge/License-AGPL%20v3-blue.svg" alt="License Badge">
</p>

NodeTool Core is the Python library that provides the node system and worker subprocess for [NodeTool](https://github.com/nodetool-ai/nodetool). The TypeScript server handles HTTP API, workflow orchestration, agents, and chat. Python handles node execution and local-compute providers (HuggingFace, MLX).

______________________________________________________________________

## What's Here

- **Node system** — `BaseNode`, `ProcessingContext`, type metadata
- **Worker subprocess** — `python -m nodetool.worker` communicates with the TS server via WebSocket+MessagePack
- **Provider infrastructure** — Base classes and registry for local-compute providers
- **Media processing** — Image, audio, video conversion utilities
- **DSL** — Graph construction and code generation helpers
- **Models** — Database models (Asset, Job, Secret, etc.)
- **Storage** — Abstract storage backends (memory, file, S3)

## Quick Start

```bash
# Install
conda create -n nodetool python=3.11 pandoc ffmpeg -c conda-forge
conda activate nodetool
uv sync

# Run tests
uv run pytest -q

# Start worker (normally spawned by TS server)
python -m nodetool.worker
```

## Writing Nodes

```python
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

class MyNode(BaseNode):
    """
    Brief description
    tags, keywords, for, search
    """
    input_text: str = ""

    async def process(self, context: ProcessingContext) -> str:
        return self.input_text.upper()
```

Nodes use `ProcessingContext` for media conversion (`image_to_pil`, `audio_from_numpy`, etc.), secrets (`get_secret`), asset storage, and progress reporting.

## Architecture

```
TS Server (Fastify)
    ├── HTTP API, WebSocket, Auth
    ├── Workflow orchestration (DAG scheduling)
    ├── Cloud providers (OpenAI, Anthropic, Gemini, ...)
    └── Spawns Python worker subprocess
            ↕ WebSocket + MessagePack
Python Worker (this repo)
    ├── Node discovery & execution
    ├── Local providers (HuggingFace, MLX)
    └── Media processing (ffmpeg, PIL, numpy)
```

## Worker Protocol

The worker speaks a msgpack message protocol to the TS server (bridge protocol **v3**, `nodetool.worker.BRIDGE_PROTOCOL_VERSION`). Two transports carry the same messages:

- **WebSocket** (default): `python -m nodetool.worker --host 0.0.0.0 --port 7777`. Each message is one binary msgpack frame. On startup the worker prints `NODETOOL_WORKER_PORT=<port>` to stdout (the only thing on stdout). If `NODETOOL_WORKER_TOKEN` is set, the opening handshake must carry `Authorization: Bearer <token>` (constant-time compare, rejected with 401 before any frame); unset means open, for local/dev use.
- **stdio** (`--stdio`): same msgpack payloads with 4-byte big-endian length-prefixed framing over stdin/stdout, for parent processes that spawn the worker directly.

Frames are capped at 256 MiB by default (`NODETOOL_BRIDGE_MAX_FRAME_SIZE`). Unknown msgpack extension types decode to `None` rather than erroring. Binary data (images, audio, model files) travels as native msgpack `bin` values — there is no base64.

### Message envelope

Every message is a map: `{"type": string, "request_id": string, "data": object}`. Requests carry a unique `request_id`; every frame the worker sends back echoes it, so concurrent requests multiplex safely over one connection. The worker replies with frames of type:

| Frame type | Meaning |
|------------|---------|
| `result` | Terminal success — one per request |
| `error` | Terminal failure — `data: {error, traceback}` |
| `progress` | Streamed progress updates (zero or more, before the terminal frame) |
| `chunk` | Streamed payload items (streaming execution, token streams, TTS audio) |
| `comfy.event` | ComfyUI execution events streamed during `comfy.execute` — `data: {event, prompt_id, ...}` |

`cancel` (`{type: "cancel", request_id: <id-of-in-flight-request>}`) requests cooperative cancellation of a running request; it produces no reply of its own — the cancelled request emits its own terminal frame.

### Core messages

| Request | Data | Response |
|---------|------|----------|
| `discover` | – | `discover` frame: `{protocol_version, nodes: [metadata], load_errors}` |
| `worker.status` | – | `{protocol_version, node_count, provider_count, namespaces, load_errors, transport, max_frame_size, comfy: {enabled, url}}` |
| `execute` | `{node_type, fields, secrets, blobs}` | `progress`\* then `result {outputs, blobs}` |
| `execute.stream` | same as `execute` | each yielded item as `chunk {outputs, blobs}`; empty `result` terminates the stream |
| `cancel` | – (uses `request_id`) | none |

For `execute`, `fields` are the node's property values; `blobs` maps field names to binary inputs (written to temp files and resolved as asset refs); `secrets` are forwarded to the node's context. Binary outputs come back in the result's `blobs` map.

### Provider messages (`provider.*`)

Local-compute providers (HuggingFace local, MLX) registered in this Python environment:

| Request | Data | Response |
|---------|------|----------|
| `provider.list` | – | `{providers: [{id, capabilities, required_secrets}]}` |
| `provider.models` | `{provider, model_type}` | `{models}` |
| `provider.generate` | `{provider, model, messages, tools?, max_tokens?, temperature?, top_p?, response_format?}` | `{message}` |
| `provider.stream` | same as `provider.generate` | `chunk {type: "chunk", content, done}` / `chunk {type: "tool_call", id, name, args}`, then `result {done: true}` |
| `provider.text_to_image` | `{provider, params}` | `result {blobs: {image}}` |
| `provider.image_to_image` | `{provider, image, params}` | `result {blobs: {image}}` |
| `provider.text_to_video` | `{provider, model, prompt, ...}` | `result {blobs: {video}}` |
| `provider.text_to_audio` | `{provider, model, prompt, ...}` | `result {blobs: {audio}}` |
| `provider.tts` | `{provider, model, text, voice?, speed?}` | `chunk {blobs: {audio}}`\* then `result {done: true}` |
| `provider.asr` | `{provider, model, audio, language?, ...}` | `{text, ...}` |
| `provider.embedding` | `{provider, model, text, dimensions?}` | `{embeddings}` |

`provider.stream` and `provider.tts` honor `cancel`.

### Model cache messages (`models.*`)

HuggingFace cache management on the worker's `HF_HOME`:

| Request | Data | Response |
|---------|------|----------|
| `models.list_cached` | – | `{models: UnifiedModel[]}` |
| `models.download` | `{repo_id, path?, allow_patterns?, ignore_patterns?}` | `progress {status: start/progress/completed/cancelled/error, downloaded_bytes, total_bytes, ...}`\* then `result {repo_id, status}` |
| `models.delete` | `{repo_id}` | `{deleted}` |

Tokens are resolved on the worker; client-supplied tokens are ignored. Downloads honor `cancel`.

### ComfyUI proxy messages (`comfy.*`)

When a ComfyUI server is co-located (`COMFYUI_URL`, advertised via `worker.status.comfy`), the worker proxies it — see [docs/comfy-proxy.md](docs/comfy-proxy.md) for frame-level shapes:

| Request | Data | Response |
|---------|------|----------|
| `comfy.execute` | `{workflow, blobs?, previews?, include_temp?, timeout?}` | `comfy.event` frames (`queued`/`executing`/`progress`/`node_output`/`preview`/…), then `result {prompt_id, status, outputs, blobs}` |
| `comfy.queue` | – | `{queue_running, queue_pending}` |
| `comfy.interrupt` | – | `{interrupted}` |
| `comfy.cancel` | `{prompt_id}` | `{cancelled, prompt_id}` |
| `comfy.upload` | `{data, filename?, kind?, subfolder?, overwrite?, original_ref?}` | ComfyUI upload response `{name, subfolder, type}` |
| `comfy.view` | `{filename, subfolder?, type?}` | `{filename, content_type, data}` |
| `comfy.object_info` | `{node_class?}` | `{object_info}` |
| `comfy.system_stats` | – | VRAM/RAM/device info |
| `comfy.free` | `{unload_models?, free_memory?}` | `{freed}` |
| `comfy.status` | – | `{enabled, url, reachable, system_stats?, queue_remaining?}` |
| `comfy.models.download` | `{folder, filename?, force?, source: {type: huggingface/url, ...}}` | `progress`\* then `result {status, path, size_bytes}` |
| `comfy.models.list` | – | `{models_dir, models: [{folder, filename, size_bytes}]}` |
| `comfy.models.delete` | `{folder, filename}` | `{deleted}` |

Input blobs are uploaded to ComfyUI and spliced into the workflow wherever a `"blob:<key>"` placeholder appears; file outputs are fetched from ComfyUI and returned as result blobs. `comfy.execute` and `comfy.models.download` honor `cancel`.

### Protocol version history

| Version | Changes |
|---------|---------|
| 1 | Initial protocol: `discover`/`execute`/`cancel` + `provider.*`, msgpack framing |
| 2 | `models.*` HuggingFace cache management |
| 3 | `comfy.*` ComfyUI proxy, `comfy.event` frame type, `comfy` capability block in `worker.status` |

The TS bridge declares the minimum version it can speak; a worker reporting a lower `protocol_version` in `discover`/`worker.status` is rejected.

## External Node Packages

- **nodetool-huggingface** — HuggingFace model integrations + local provider
- **nodetool-mlx** — Apple Silicon optimized nodes + MLX provider
- **nodetool-replicate** — Replicate API integrations
- **nodetool-fal** — FAL AI service integrations
- **nodetool-elevenlabs** — ElevenLabs audio generation
- **nodetool-apple** — Apple platform integrations

## Development

```bash
# Setup
conda activate nodetool
uv sync --group dev

# Run tests
uv run pytest -q

# Lint
uv run ruff check .
```

## License

[AGPL License](LICENSE)

## Learn More

- [NodeTool Website](https://nodetool.ai)
- [Discord Community](https://discord.gg/nodetool)
