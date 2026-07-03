# Nodetool Providers

Provider infrastructure for AI service integrations.

Cloud/API providers (OpenAI, Anthropic, Gemini, Ollama, etc.) are in the TypeScript server. This package contains the base classes, registry, and utilities used by local-compute providers (HuggingFace Local, MLX) registered via external Python packages.

## Files

- `__init__.py` — Package exports
- `base.py` — `BaseProvider`, `register_provider`, `get_registered_provider`
- `types.py` — Shared types (`ImageBytes`, `TextToImageParams`, `ImageToImageParams`)
