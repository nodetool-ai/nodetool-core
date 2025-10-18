import json
from typing import AsyncGenerator
from nodetool.config.environment import Environment
from nodetool.metadata.types import LlamaModel
from nodetool.types.model import UnifiedModel
from ollama import AsyncClient

# Simple module-level cache
_cached_ollama_models = None


def get_ollama_client() -> AsyncClient:
    api_url = Environment.get("OLLAMA_API_URL")
    assert api_url, "OLLAMA_API_URL not set"

    return AsyncClient(api_url)


async def get_ollama_models() -> list[LlamaModel]:
    global _cached_ollama_models

    if Environment.is_production() and _cached_ollama_models is not None:
        return _cached_ollama_models

    try:
        ollama = get_ollama_client()
        models = await ollama.list()
        result = [
            LlamaModel(
                name=model.model or "",
                repo_id=model.model or "",
                modified_at=model.modified_at.isoformat() if model.modified_at else "",
                size=model.size or 0,
                digest=model.digest or "",
                details=model.details.model_dump() if model.details else {},
            )
            for model in models.models
        ]

        if Environment.is_production():
            _cached_ollama_models = result
        return result
    except Exception as e:
        print(f"Error getting ollama models: {e}")
        return []


async def get_ollama_models_unified() -> list[UnifiedModel]:
    models = await get_ollama_models()
    return [
        UnifiedModel(
            id=model.name,
            type="llama_model",
            name=model.name,
            repo_id=model.name,
            path=None,
            cache_path=None,
            allow_patterns=None,
            ignore_patterns=None,
            description=None,
            readme=None,
            size_on_disk=model.size,
            downloaded=True,
            pipeline_tag=None,
            tags=None,
            has_model_index=False,
            downloads=0,
            likes=0,
            trending_score=0,
        )
        for model in models
    ]


async def get_ollama_model_info(model_name: str) -> dict | None:
    ollama = get_ollama_client()
    try:
        res = await ollama.show(model_name)
    except Exception:
        return None
    return res.model_dump()


async def stream_ollama_model_pull(model_name: str) -> AsyncGenerator[str, None]:
    try:
        ollama = get_ollama_client()
        res = await ollama.pull(model_name, stream=True)
        async for chunk in res:
            yield json.dumps(chunk.model_dump()) + "\n"
    except Exception as e:
        # Surface a clear, user-friendly error when Ollama is not reachable
        api_url = Environment.get("OLLAMA_API_URL")
        error_payload = {
            "status": "error",
            "message": (
                f"Cannot connect to Ollama at {api_url!s}. "
                "Make sure Ollama is running. Try: 'ollama serve' or set OLLAMA_API_URL."
            ),
            "error": str(e),
            "model": model_name,
        }
        yield json.dumps(error_payload) + "\n"


async def delete_ollama_model(model_name: str) -> bool:
    """Delete an Ollama model by name.

    Returns True when deletion succeeds or the model does not exist,
    False on error.
    """
    ollama = get_ollama_client()
    try:
        await ollama.delete(model_name)
        return True
    except Exception as e:
        print(f"Error deleting ollama model '{model_name}': {e}")
        return False
