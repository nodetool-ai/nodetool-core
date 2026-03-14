"""
Provider bridge handler for the worker server.

Loads Python-only providers (HuggingFace Local, MLX) and handles
provider operations dispatched from the TS backend via the WebSocket protocol.
"""

import asyncio
import sys
import traceback
from typing import Any, AsyncIterator

from websockets.asyncio.server import ServerConnection
import msgpack


# Cached provider instances
_provider_cache: dict[str, Any] = {}
_providers_imported = False


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

    if provider_id in _provider_cache:
        return _provider_cache[provider_id]

    provider_enum = Provider(provider_id)
    cls, kwargs = get_registered_provider(provider_enum)
    instance = cls(secrets=secrets, **kwargs)
    _provider_cache[provider_id] = instance
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
                "generate_message", "generate_messages", "text_to_image",
                "image_to_image", "text_to_speech", "automatic_speech_recognition",
                "text_to_video", "image_to_video", "generate_embedding",
            }
            for method_name in base_methods:
                method = getattr(cls, method_name, None)
                if method is not None:
                    # Check if it's overridden from BaseProvider
                    from nodetool.providers.base import BaseProvider
                    base_method = getattr(BaseProvider, method_name, None)
                    if method is not base_method:
                        capabilities.append(method_name)

            result.append({
                "id": pid,
                "capabilities": capabilities,
                "required_secrets": cls.required_secrets() if hasattr(cls, "required_secrets") else [],
            })
    return result


async def handle_provider_message(
    msg_type: str,
    request_id: str | None,
    data: dict[str, Any],
    websocket: ServerConnection,
    cancel_flags: dict[str, asyncio.Event],
) -> None:
    """Handle a provider.* message type."""
    try:
        if msg_type == "provider.list":
            providers = get_available_providers()
            await _send_result(websocket, request_id, {"providers": providers})

        elif msg_type == "provider.models":
            result = await _handle_models(data)
            await _send_result(websocket, request_id, result)

        elif msg_type == "provider.generate":
            result = await _handle_generate(data)
            await _send_result(websocket, request_id, result)

        elif msg_type == "provider.stream":
            cancel_event = asyncio.Event()
            if request_id:
                cancel_flags[request_id] = cancel_event
            try:
                await _handle_stream(data, request_id, websocket, cancel_event)
            finally:
                if request_id:
                    cancel_flags.pop(request_id, None)

        elif msg_type == "provider.text_to_image":
            result = await _handle_text_to_image(data)
            await _send_result(websocket, request_id, result)

        elif msg_type == "provider.image_to_image":
            result = await _handle_image_to_image(data)
            await _send_result(websocket, request_id, result)

        elif msg_type == "provider.tts":
            cancel_event = asyncio.Event()
            if request_id:
                cancel_flags[request_id] = cancel_event
            try:
                await _handle_tts(data, request_id, websocket, cancel_event)
            finally:
                if request_id:
                    cancel_flags.pop(request_id, None)

        elif msg_type == "provider.asr":
            result = await _handle_asr(data)
            await _send_result(websocket, request_id, result)

        elif msg_type == "provider.embedding":
            result = await _handle_embedding(data)
            await _send_result(websocket, request_id, result)

        else:
            await _send_error(websocket, request_id, f"Unknown provider message type: {msg_type}")

    except Exception as e:
        await _send_error(websocket, request_id, str(e), traceback.format_exc())


# ── Message helpers ──────────────────────────────────────────────────────


async def _send_result(ws: ServerConnection, request_id: str | None, data: dict) -> None:
    await ws.send(msgpack.packb({
        "type": "result",
        "request_id": request_id,
        "data": data,
    }))


async def _send_error(ws: ServerConnection, request_id: str | None, error: str, tb: str | None = None) -> None:
    await ws.send(msgpack.packb({
        "type": "error",
        "request_id": request_id,
        "data": {"error": error, "traceback": tb},
    }))


async def _send_chunk(ws: ServerConnection, request_id: str | None, data: dict) -> None:
    await ws.send(msgpack.packb({
        "type": "chunk",
        "request_id": request_id,
        "data": data,
    }))


# ── Handler implementations ─────────────────────────────────────────────


def _deserialize_messages(raw_messages: list[dict]) -> list[Any]:
    """Convert wire-format messages to Python Message objects."""
    from nodetool.metadata.types import Message

    messages = []
    for m in raw_messages:
        content = m.get("content")
        # content can be string, list of content parts, or None
        if isinstance(content, list):
            from nodetool.metadata.types import (
                MessageTextContent,
                MessageImageContent,
                MessageAudioContent,
            )
            parts = []
            for part in content:
                if part.get("type") == "text":
                    parts.append(MessageTextContent(type="text", text=part["text"]))
                elif part.get("type") == "image":
                    parts.append(MessageImageContent(type="image", image=part["image"]))
                elif part.get("type") == "audio":
                    parts.append(MessageAudioContent(type="audio", audio=part["audio"]))
            content = parts

        msg = Message(
            role=m["role"],
            content=content,
        )
        if m.get("tool_calls"):
            from nodetool.metadata.types import ToolCall
            msg.tool_calls = [
                ToolCall(id=tc["id"], name=tc["name"], args=tc.get("args", {}))
                for tc in m["tool_calls"]
            ]
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
        result["tool_calls"] = [
            {"id": tc.id, "name": tc.name, "args": tc.args}
            for tc in msg.tool_calls
        ]
    return result


def _serialize_content_part(part: Any) -> dict:
    """Serialize a MessageContent part."""
    if hasattr(part, "text"):
        return {"type": "text", "text": part.text}
    if hasattr(part, "image"):
        return {"type": "image", "image": part.image}
    if hasattr(part, "audio"):
        return {"type": "audio", "audio": part.audio}
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
    return {
        "models": [
            m.model_dump() if hasattr(m, "model_dump") else m.__dict__
            for m in models
        ]
    }


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


async def _handle_stream(
    data: dict,
    request_id: str | None,
    websocket: ServerConnection,
    cancel_event: asyncio.Event,
) -> None:
    """Handle provider.stream — streaming message generation."""
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

    async for item in provider.generate_messages(
        messages=messages,
        model=model,
        **kwargs,
    ):
        if cancel_event.is_set():
            break

        if isinstance(item, ToolCall):
            await _send_chunk(websocket, request_id, {
                "type": "tool_call",
                "id": item.id,
                "name": item.name,
                "args": item.args,
            })
        else:
            # Chunk object
            chunk_data: dict[str, Any] = {
                "type": "chunk",
                "content": getattr(item, "content", str(item)),
                "done": getattr(item, "done", False),
            }
            await _send_chunk(websocket, request_id, chunk_data)

    # Signal stream end
    await _send_result(websocket, request_id, {"done": True})


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


async def _handle_tts(
    data: dict,
    request_id: str | None,
    websocket: ServerConnection,
    cancel_event: asyncio.Event,
) -> None:
    """Handle provider.tts — streaming text-to-speech."""
    provider = _get_provider(data["provider"], data.get("secrets", {}))

    kwargs: dict[str, Any] = {
        "text": data["text"],
        "model": data["model"],
    }
    for key in ("voice", "speed"):
        if key in data:
            kwargs[key] = data[key]

    async for audio_chunk in provider.text_to_speech(**kwargs):
        if cancel_event.is_set():
            break
        # audio_chunk is numpy int16 array
        await websocket.send(msgpack.packb({
            "type": "chunk",
            "request_id": request_id,
            "data": {"blobs": {"audio": audio_chunk.tobytes()}},
        }))

    await _send_result(websocket, request_id, {"done": True})


async def _handle_asr(data: dict) -> dict:
    """Handle provider.asr — automatic speech recognition."""
    provider = _get_provider(data["provider"], data.get("secrets", {}))

    kwargs: dict[str, Any] = {
        "audio": data.get("audio", b""),
        "model": data["model"],
    }
    for key in ("language", "prompt", "temperature"):
        if key in data:
            kwargs[key] = data[key]

    text = await provider.automatic_speech_recognition(**kwargs)
    return {"text": text}


async def _handle_embedding(data: dict) -> dict:
    """Handle provider.embedding — generate embeddings."""
    provider = _get_provider(data["provider"], data.get("secrets", {}))

    result = await provider.generate_embedding(
        text=data["text"],
        model=data["model"],
        dimensions=data.get("dimensions"),
    )
    return {"embeddings": result}
