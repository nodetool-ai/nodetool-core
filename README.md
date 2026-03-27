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
