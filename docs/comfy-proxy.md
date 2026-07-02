# ComfyUI Proxy (comfy.* worker protocol)

The worker can front a co-located ComfyUI server and expose it through the
NodeTool bridge protocol (protocol version 3+). The TS server talks msgpack to
the worker on `:7777`; the worker talks HTTP + WebSocket to ComfyUI on
loopback. ComfyUI is never exposed directly.

```
TS server ──ws/msgpack──▶ nodetool worker ──http/ws──▶ ComfyUI (127.0.0.1:8188)
                              │
                              └─▶ /workspace (RunPod network volume: models, input, output)
```

Handler: `src/nodetool/worker/comfy_handler.py`. Image: `Dockerfile.comfy` →
`ghcr.io/nodetool-ai/nodetool-worker-comfy` (built by
`.github/workflows/docker-comfy.yml`).

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `COMFYUI_URL` | Base URL of the ComfyUI server to proxy | `http://127.0.0.1:8188` |
| `COMFY_MODELS_DIR` | Model tree for `comfy.models.*` | `/workspace/models` |
| `NODETOOL_COMFY_ENABLED` | Advertise the comfy capability in `worker.status` | implied by `COMFYUI_URL` |
| `COMFY_ARGS` | Extra ComfyUI CLI flags (image entrypoint only) | – |
| `COMFY_STARTUP_TIMEOUT` | Seconds to wait for ComfyUI boot (image entrypoint only) | `300` |

`worker.status` includes a `comfy: {enabled, url}` block so the TS server can
route ComfyUI jobs only to workers that proxy one.

## comfy.execute

Runs a workflow (ComfyUI **API-format** prompt JSON) with managed inputs and
outputs. One request = one ComfyUI `client_id` + one events WebSocket, so
concurrent prompts never leak events across bridge requests; routing stays
keyed on the protocol `request_id`.

Request `data`:

```jsonc
{
  "workflow": { "1": { "class_type": "LoadImage", "inputs": { "image": "blob:photo" } }, ... },
  "blobs": { "photo": "<bytes>" },   // binary inputs, msgpack bin
  "previews": false,                  // forward binary preview frames
  "include_temp": false,              // also fetch type=temp outputs (previews)
  "timeout": 600                      // optional wall-clock seconds
}
```

Input management: each blob is uploaded via `POST /upload/image` under a
unique `nodetool_*` filename (extension sniffed from the bytes), and every
string value `"blob:<key>"` in the workflow is replaced with the uploaded
filename before `POST /prompt`.

Execution events stream back as dedicated `comfy.event` frames
(`{"type": "comfy.event", "request_id": ..., "data": {"event": ..., "prompt_id": ..., ...}}`) —
a separate frame type from `progress`, whose generic `{progress, total, message}`
shape doesn't fit ComfyUI's lifecycle. In submission order:

| `event` | Fields | Source |
|---------|--------|--------|
| `queued` | `prompt_id`, `queue_position` | `POST /prompt` response |
| `queue` | `queue_remaining` | ws `status` event |
| `started` / `cached` | `nodes` (cached) | ws `execution_start` / `execution_cached` |
| `executing` | `node` | ws `executing` |
| `progress` | `node`, `value`, `max` | ws `progress` |
| `node_output` | `node`, `outputs` (filename metadata) | ws `executed` |
| `preview` | `format` (`jpeg`/`png`), `image` (bytes) | ws binary frames (opt-in) |
| `completed` / `cancelled` | `prompt_id` | terminal |

Result frame `data`:

```jsonc
{
  "prompt_id": "...",
  "status": "completed",            // or "cancelled"
  "outputs": {
    "9": { "images": [ { "filename": "ComfyUI_0001.png", "subfolder": "", "type": "output",
                          "content_type": "image/png", "blob": "9/images/0/ComfyUI_0001.png" } ] }
  },
  "blobs": { "9/images/0/ComfyUI_0001.png": "<bytes>" }
}
```

Output management: after the events socket reports completion, the handler
reads `GET /history/{prompt_id}` (with retries — history writes lag the
socket) and fetches every file output via `GET /view`, returning the bytes as
result blobs. Non-file outputs (e.g. text) pass through as-is. `type: "temp"`
entries (preview nodes) keep their metadata but are only fetched with
`include_temp`. If the socket drops mid-run the handler falls back to polling
`/history` + `/queue`.

Cancellation uses the standard protocol `cancel` message: the handler deletes
the prompt from the pending queue (`POST /queue {"delete": [id]}`) and only
calls `POST /interrupt` if `GET /queue` shows *our* prompt running —
interrupting blindly would kill another client's job.

Errors (workflow validation from `POST /prompt`, `execution_error` events,
timeouts) surface as protocol `error` frames.

## Other comfy.* messages

| Message | Data | Result |
|---------|------|--------|
| `comfy.status` | – | `{enabled, url, reachable, system_stats, queue_remaining}` |
| `comfy.queue` | – | `{queue_running, queue_pending}` (`GET /queue`) |
| `comfy.interrupt` | – | `{interrupted}` (`POST /interrupt`) |
| `comfy.cancel` | `prompt_id` | `{cancelled}` — queue-delete + guarded interrupt |
| `comfy.upload` | `data` (bytes), `filename?`, `kind` (`image`/`mask`), `subfolder?`, `overwrite?`, `original_ref?` | ComfyUI upload response `{name, subfolder, type}` |
| `comfy.view` | `filename`, `subfolder?`, `type?` | `{filename, content_type, data}` |
| `comfy.object_info` | `node_class?` | `{object_info}` — node catalog |
| `comfy.system_stats` | – | VRAM/RAM/device info |
| `comfy.free` | `unload_models?`, `free_memory?` | `{freed}` — unload models without a cold start |

## Model volume management (comfy.models.*)

Models live under `COMFY_MODELS_DIR` (`/workspace/models` — the RunPod network
volume), which the bundled `extra_model_paths.yaml` points ComfyUI at, so
downloads survive pod restarts.

`comfy.models.download`:

```jsonc
{
  "folder": "checkpoints",           // ComfyUI folder, or a nodetool type name
                                      // like "comfy.checkpoint_file" (mapped via
                                      // comfy_model_to_folder)
  "filename": "sd_xl_base_1.0.safetensors",  // optional; defaults to source basename
  "force": false,                     // re-download even if present
  "source": { "type": "huggingface", "repo_id": "org/repo", "path": "file.safetensors", "revision": "main" }
  // or       { "type": "url", "url": "https://..." }
}
```

Emits `start`/`progress`/`completed` (or `cancelled`/`error`) progress frames
with `downloaded_bytes`/`total_bytes`, mirroring `models.download`. Downloads
stream to a `.part` file and rename atomically; an existing file
short-circuits with `status: "exists"` (no network). HuggingFace auth uses the
worker-side token — client-supplied tokens are ignored. Paths are validated
against traversal out of the models dir. Cancellable via the protocol `cancel`
message.

`comfy.models.list` → `{models_dir, models: [{folder, filename, size_bytes}]}`.
`comfy.models.delete` `{folder, filename}` → `{deleted}`.

## The nodetool-worker-comfy image

`Dockerfile.comfy` bundles ComfyUI (pinned via `COMFYUI_VERSION` build-arg)
and nodetool-core (from PyPI, `NODETOOL_VERSION` build-arg).
`docker/comfy/start.sh` boots ComfyUI on loopback, waits for its HTTP API,
then starts the worker on `0.0.0.0:7777`; if either process dies the container
exits so the orchestrator restarts it. Only `:7777` is exposed.

The GitHub workflow `docker-comfy.yml` builds and pushes
`ghcr.io/nodetool-ai/nodetool-worker-comfy:<version>` on every release tag
(waiting for the matching nodetool-core on PyPI first) and supports manual
dispatch to rebuild with a different ComfyUI version.
