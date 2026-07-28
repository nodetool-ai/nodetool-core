"""
Provider bridge handler for the worker server.

Loads Python-only providers (HuggingFace Local, MLX) and handles
provider operations dispatched from the TS backend. The handler is
transport-agnostic — it only needs a transport exposing ``send_msg`` —
so the same code path serves both the WebSocket and stdio workers.
"""

import asyncio
import hashlib
import os
import sys
import traceback
from collections import OrderedDict
from functools import lru_cache
from typing import Any

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

# Cached provider instances, keyed by (provider_id, secrets hash) so a
# different / rotated secret set never reuses another caller's instance.
# Bounded LRU: each instance can pin loaded model weights, so an unbounded
# cache would leak a full model copy per rotated token / per user secrets set.
PROVIDER_CACHE_MAX_SIZE = max(1, int(os.environ.get("NODETOOL_PROVIDER_CACHE_SIZE", "4")))
_provider_cache: "OrderedDict[tuple[str, str], Any]" = OrderedDict()
_providers_imported = False


def _hash_secrets(secrets: dict[str, str]) -> str:
    """Stable hash of a secrets dict for cache keying (never logs values)."""
    h = hashlib.sha256()
    for key, value in sorted((secrets or {}).items()):
        h.update(key.encode("utf-8"))
        h.update(b"\x00")
        h.update(str(value).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _release_provider(instance: Any) -> None:
    """Best-effort release of an evicted provider's resources (model weights)."""
    for hook_name in ("close", "unload", "unload_model"):
        hook = getattr(instance, hook_name, None)
        if not callable(hook):
            continue
        try:
            result = hook()
            if asyncio.iscoroutine(result):
                try:
                    asyncio.get_running_loop().create_task(result)
                except RuntimeError:
                    result.close()
        except Exception as e:  # pragma: no cover — release is best-effort
            print(f"Warning: failed to release provider via {hook_name}(): {e}", file=sys.stderr)
        return


def _ensure_providers_imported() -> None:
    """Import provider modules so they register via @register_provider."""
    global _providers_imported
    if _providers_imported:
        return
    _providers_imported = True

    # Try importing local-only providers
    for module_name in [
        "nodetool.mlx.mlx_provider",
        "nodetool.huggingface.huggingface_local_provider",
    ]:
        try:
            __import__(module_name)
            print(f"Loaded provider module: {module_name}", file=sys.stderr)
        except ImportError:
            pass
        except Exception as e:
            print(f"Warning: failed to import {module_name}: {e}", file=sys.stderr)


def _get_provider(provider_id: str, secrets: dict[str, str]) -> Any:
    """Get or create a cached provider instance."""
    from nodetool.metadata.types import Provider
    from nodetool.providers.base import get_registered_provider

    _ensure_providers_imported()

    cache_key = (provider_id, _hash_secrets(secrets))
    cached = _provider_cache.get(cache_key)
    if cached is not None:
        _provider_cache.move_to_end(cache_key)
        return cached

    provider_enum = Provider(provider_id)
    cls, kwargs = get_registered_provider(provider_enum)
    instance = cls(secrets=secrets, **kwargs)
    _provider_cache[cache_key] = instance
    _provider_cache.move_to_end(cache_key)
    while len(_provider_cache) > PROVIDER_CACHE_MAX_SIZE:
        _, evicted = _provider_cache.popitem(last=False)
        _release_provider(evicted)
    return instance


def get_available_providers() -> list[dict[str, Any]]:
    """List providers that are available in this Python environment."""
    from nodetool.providers.base import _PROVIDER_REGISTRY

    _ensure_providers_imported()

    # Only expose local/Python-only providers through the bridge
    local_providers = {"huggingface", "mlx"}
    result = []
    for provider_enum in _PROVIDER_REGISTRY:
        pid = str(provider_enum.value)
        if pid in local_providers:
            cls, _ = _PROVIDER_REGISTRY[provider_enum]
            # Detect capabilities from implemented methods
            capabilities = []
            base_methods = {
                "generate_message",
                "generate_messages",
                "text_to_image",
                "image_to_image",
                "text_to_speech",
                "text_to_audio",
                "automatic_speech_recognition",
                "text_to_video",
                "image_to_video",
                "generate_embedding",
            }
            for method_name in base_methods:
                method = getattr(cls, method_name, None)
                if method is not None:
                    # Check if it's overridden from BaseProvider
                    from nodetool.providers.base import BaseProvider

                    base_method = getattr(BaseProvider, method_name, None)
                    if method is not base_method:
                        capabilities.append(method_name)

            result.append(
                {
                    "id": pid,
                    "capabilities": capabilities,
                    "required_secrets": (cls.required_secrets() if hasattr(cls, "required_secrets") else []),
                }
            )
    return result


# ── Handler implementations ─────────────────────────────────────────────


@lru_cache(maxsize=1)
def _content_part_types() -> dict[str, tuple[type[Any], str]]:
    """Map wire content-part ``type`` values to (model class, payload field).

    The wire protocol names the image part ``"image"`` while
    :class:`MessageImageContent` declares ``Literal["image_url"]``, so the
    discriminator can never be forwarded verbatim — only the payload field is.
    """
    from nodetool.metadata.types import (
        MessageAudioContent,
        MessageImageContent,
        MessageTextContent,
        MessageVideoContent,
    )

    return {
        "text": (MessageTextContent, "text"),
        "image": (MessageImageContent, "image"),
        "image_url": (MessageImageContent, "image"),
        "audio": (MessageAudioContent, "audio"),
        "video": (MessageVideoContent, "video"),
    }


def _deserialize_messages(raw_messages: list[dict[str, Any]]) -> list[Any]:
    """Convert wire-format messages to Python Message objects."""
    from nodetool.metadata.types import Message

    messages: list[Any] = []
    for m in raw_messages:
        content: Any = m.get("content")
        # content can be string, list of content parts, or None
        if isinstance(content, list):
            parts: list[Any] = []
            for part in content:
                # Note: the discriminator is never passed explicitly — the wire
                # names ("image") differ from the model's own Literal defaults
                # ("image_url"), and passing the wire name raises a pydantic
                # ValidationError.
                wire_type = part.get("type")
                factory = _content_part_types().get(wire_type)
                if factory is None:
                    log.warning("Dropping message content part with unknown type %r", wire_type)
                    continue
                model_cls, field = factory
                if field not in part:
                    log.warning("Dropping %r content part with no %r field", wire_type, field)
                    continue
                parts.append(model_cls(**{field: part[field]}))
            content = parts

        msg = Message(
            role=m["role"],
            content=content,
        )
        if m.get("tool_calls"):
            from nodetool.metadata.types import ToolCall

            msg.tool_calls = [ToolCall(id=tc["id"], name=tc["name"], args=tc.get("args", {})) for tc in m["tool_calls"]]
        if m.get("tool_call_id"):
            msg.tool_call_id = m["tool_call_id"]
        messages.append(msg)
    return messages


def _serialize_message(msg: Any) -> dict:
    """Convert a Python Message to wire format."""
    result: dict[str, Any] = {"role": msg.role}

    if isinstance(msg.content, str):
        result["content"] = msg.content
    elif isinstance(msg.content, list):
        result["content"] = [_serialize_content_part(p) for p in msg.content]
    elif msg.content is not None:
        result["content"] = str(msg.content)

    if msg.tool_calls:
        result["tool_calls"] = [{"id": tc.id, "name": tc.name, "args": tc.args} for tc in msg.tool_calls]
    return result


def _serialize_content_part(part: Any) -> dict[str, Any]:
    """Serialize a MessageContent part to the wire format.

    Mirrors :func:`_content_part_types` so a round-trip through the bridge is
    lossless — including video parts, which previously degraded to ``str(part)``
    text because they were not recognised here.
    """
    for wire_type, (_model_cls, field) in _content_part_types().items():
        # "image_url" is only accepted on the way in; emit the canonical "image".
        if wire_type == "image_url":
            continue
        value = getattr(part, field, None)
        if value is not None:
            return {"type": wire_type, field: value}
    return {"type": "text", "text": str(part)}


async def _handle_models(data: dict) -> dict:
    """Handle provider.models — return available models for a provider."""
    provider = _get_provider(data["provider"], data.get("secrets", {}))
    model_type = data.get("model_type", "language")

    getter_map = {
        "language": "get_available_language_models",
        "image": "get_available_image_models",
        "tts": "get_available_tts_models",
        "asr": "get_available_asr_models",
        "video": "get_available_video_models",
        "embedding": "get_available_embedding_models",
        "3d": "get_available_3d_models",
    }

    getter_name = getter_map.get(model_type)
    if not getter_name:
        return {"models": []}

    getter = getattr(provider, getter_name, None)
    if getter is None:
        return {"models": []}

    models = await getter()
    return {"models": [m.model_dump() if hasattr(m, "model_dump") else m.__dict__ for m in models]}


async def _handle_generate(data: dict) -> dict:
    """Handle provider.generate — single message generation."""
    provider = _get_provider(data["provider"], data.get("secrets", {}))
    messages = _deserialize_messages(data["messages"])
    model = data["model"]

    kwargs: dict[str, Any] = {}
    for key in ("max_tokens", "temperature", "top_p", "response_format"):
        if key in data:
            kwargs[key] = data[key]

    tools = data.get("tools")
    if tools:
        kwargs["tools"] = tools

    result_msg = await provider.generate_message(
        messages=messages,
        model=model,
        **kwargs,
    )
    return {"message": _serialize_message(result_msg)}


async def _handle_text_to_image(data: dict) -> dict:
    """Handle provider.text_to_image."""
    provider = _get_provider(data["provider"], data.get("secrets", {}))
    params = data.get("params", {})

    image_bytes = await provider.text_to_image(params)
    return {"blobs": {"image": image_bytes}}


async def _handle_image_to_image(data: dict) -> dict:
    """Handle provider.image_to_image."""
    provider = _get_provider(data["provider"], data.get("secrets", {}))
    image_data = data.get("image", b"")
    params = data.get("params", {})

    result_bytes = await provider.image_to_image(image_data, params)
    return {"blobs": {"image": result_bytes}}


async def _extract_media_bytes(ctx: Any, ref: Any) -> bytes:
    """Pull encoded bytes out of a worker context after media generation.

    Audio/image are captured into the WorkerContext's output blobs; file-based
    refs (e.g. video) are resolved directly via the context.
    """
    get_blobs = getattr(ctx, "get_output_blobs", None)
    blobs = get_blobs() if callable(get_blobs) else {}
    if blobs:
        return next(iter(blobs.values()))
    return await ctx.asset_to_bytes(ref)


async def _handle_text_to_video(data: dict) -> dict:
    """Handle provider.text_to_video."""
    from nodetool.worker.context_stub import WorkerContext

    provider = _get_provider(data["provider"], data.get("secrets", {}))
    ctx = WorkerContext(secrets=data.get("secrets", {}))
    kwargs: dict[str, Any] = {
        "prompt": data["prompt"],
        "model": data["model"],
        "context": ctx,
    }
    for key in (
        "negative_prompt",
        "num_frames",
        "guidance_scale",
        "num_inference_steps",
        "height",
        "width",
        "fps",
        "seed",
        "max_sequence_length",
    ):
        if key in data:
            kwargs[key] = data[key]

    video_ref = await provider.text_to_video(**kwargs)
    return {"blobs": {"video": await _extract_media_bytes(ctx, video_ref)}}


async def _handle_text_to_audio(data: dict) -> dict:
    """Handle provider.text_to_audio."""
    from nodetool.worker.context_stub import WorkerContext

    provider = _get_provider(data["provider"], data.get("secrets", {}))
    ctx = WorkerContext(secrets=data.get("secrets", {}))
    kwargs: dict[str, Any] = {
        "prompt": data["prompt"],
        "model": data["model"],
        "context": ctx,
    }
    for key in (
        "lyrics",
        "audio_duration",
        "guidance_scale",
        "num_inference_steps",
        "seed",
    ):
        if key in data:
            kwargs[key] = data[key]

    audio_ref = await provider.text_to_audio(**kwargs)
    return {"blobs": {"audio": await _extract_media_bytes(ctx, audio_ref)}}


async def _handle_asr(data: dict) -> dict:
    """Handle provider.asr — automatic speech recognition."""
    provider = _get_provider(data["provider"], data.get("secrets", {}))

    kwargs: dict[str, Any] = {
        "audio": data.get("audio", b""),
        "model": data["model"],
    }
    for key in ("language", "prompt", "temperature", "word_timestamps"):
        if key in data:
            kwargs[key] = data[key]

    result = await provider.automatic_speech_recognition(**kwargs)
    # Provider may return str (legacy) or dict with text + chunks
    if isinstance(result, str):
        return {"text": result}
    return result


async def _handle_embedding(data: dict) -> dict:
    """Handle provider.embedding — generate embeddings."""
    provider = _get_provider(data["provider"], data.get("secrets", {}))

    result = await provider.generate_embedding(
        text=data["text"],
        model=data["model"],
        dimensions=data.get("dimensions"),
    )
    return {"embeddings": result}


# ── Provider message dispatch ─────────────────────────────────────────────


async def handle_provider_message(
    msg_type: str,
    request_id: str | None,
    data: dict[str, Any],
    transport: Any,  # WorkerTransport (exposes async send_msg)
    cancel_flags: dict[str, asyncio.Event],
) -> None:
    """Handle a provider.* message via any transport exposing ``send_msg``."""

    async def send_result(rid: str | None, d: dict) -> None:
        await transport.send_msg({"type": "result", "request_id": rid, "data": d})

    async def send_error(rid: str | None, error: str, tb: str | None = None) -> None:
        await transport.send_msg(
            {
                "type": "error",
                "request_id": rid,
                "data": {"error": error, "traceback": tb},
            }
        )

    async def send_chunk(rid: str | None, d: dict) -> None:
        await transport.send_msg({"type": "chunk", "request_id": rid, "data": d})

    try:
        if msg_type == "provider.list":
            providers = get_available_providers()
            await send_result(request_id, {"providers": providers})

        elif msg_type == "provider.models":
            result = await _handle_models(data)
            await send_result(request_id, result)

        elif msg_type == "provider.generate":
            result = await _handle_generate(data)
            await send_result(request_id, result)

        elif msg_type == "provider.stream":
            cancel_event = asyncio.Event()
            if request_id:
                cancel_flags[request_id] = cancel_event
            try:
                provider = _get_provider(data["provider"], data.get("secrets", {}))
                messages = _deserialize_messages(data["messages"])
                model = data["model"]
                kwargs: dict[str, Any] = {}
                for key in ("max_tokens", "temperature", "top_p", "response_format"):
                    if key in data:
                        kwargs[key] = data[key]
                tools = data.get("tools")
                if tools:
                    kwargs["tools"] = tools

                from nodetool.metadata.types import ToolCall

                async for item in provider.generate_messages(messages=messages, model=model, **kwargs):
                    if cancel_event.is_set():
                        break
                    if isinstance(item, ToolCall):
                        await send_chunk(
                            request_id,
                            {
                                "type": "tool_call",
                                "id": item.id,
                                "name": item.name,
                                "args": item.args,
                            },
                        )
                    else:
                        await send_chunk(
                            request_id,
                            {
                                "type": "chunk",
                                "content": getattr(item, "content", str(item)),
                                "done": getattr(item, "done", False),
                            },
                        )
                await send_result(request_id, {"done": True})
            finally:
                if request_id:
                    cancel_flags.pop(request_id, None)

        elif msg_type == "provider.text_to_image":
            result = await _handle_text_to_image(data)
            await send_result(request_id, result)

        elif msg_type == "provider.image_to_image":
            result = await _handle_image_to_image(data)
            await send_result(request_id, result)

        elif msg_type == "provider.text_to_video":
            result = await _handle_text_to_video(data)
            await send_result(request_id, result)

        elif msg_type == "provider.text_to_audio":
            result = await _handle_text_to_audio(data)
            await send_result(request_id, result)

        elif msg_type == "provider.tts":
            cancel_event = asyncio.Event()
            if request_id:
                cancel_flags[request_id] = cancel_event
            try:
                provider = _get_provider(data["provider"], data.get("secrets", {}))
                kwargs_tts: dict[str, Any] = {
                    "text": data["text"],
                    "model": data["model"],
                }
                for key in ("voice", "speed"):
                    if key in data:
                        kwargs_tts[key] = data[key]
                async for audio_chunk in provider.text_to_speech(**kwargs_tts):
                    if cancel_event.is_set():
                        break
                    await send_chunk(request_id, {"blobs": {"audio": audio_chunk.tobytes()}})
                await send_result(request_id, {"done": True})
            finally:
                if request_id:
                    cancel_flags.pop(request_id, None)

        elif msg_type == "provider.asr":
            result = await _handle_asr(data)
            await send_result(request_id, result)

        elif msg_type == "provider.embedding":
            result = await _handle_embedding(data)
            await send_result(request_id, result)

        else:
            await send_error(request_id, f"Unknown provider message type: {msg_type}")

    except Exception as e:
        await send_error(request_id, str(e), traceback.format_exc())
