"""
Provider bridge handler for the worker server.

Loads Python-only providers (HuggingFace Local, MLX) and handles
provider operations dispatched from the TS backend. The handler is
transport-agnostic — it only needs a transport exposing ``send_msg`` —
so the same code path serves both the WebSocket and stdio workers.
"""

import asyncio
import hashlib
import json
import os
import re
import shlex
import signal
import sys
import tempfile
import traceback
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
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

_ADAPTER_PROVIDER_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_ADAPTER_COMMAND_PREFIX = "NODETOOL_PROVIDER_ADAPTER_COMMAND_"
_ADAPTER_STDERR_LIMIT = 32 * 1024
_ADAPTER_LINE_LIMIT = 1024 * 1024
_REFERENCE_INPUT_LIMIT = 192 * 1024 * 1024


def _reference_media_suffix(data: bytes, kind: str) -> str | None:
    """Identify reference media formats before exposing them to an adapter."""
    if kind == "image":
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if data.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return ".webp"
        if data.startswith((b"GIF87a", b"GIF89a")):
            return ".gif"
        return None
    if kind == "video":
        if data.startswith(b"\x1a\x45\xdf\xa3"):
            return ".webm"
        if len(data) >= 12 and data[4:8] == b"ftyp" and data[8:12] in {
            b"isom", b"iso2", b"mp41", b"mp42", b"avc1", b"mp4v", b"M4V "
        }:
            return ".mp4"
        return None
    return None


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


def _adapter_suffix(provider_id: str) -> str:
    return provider_id.upper().replace("-", "_")


def _adapter_command_env(provider_id: str) -> str:
    return f"{_ADAPTER_COMMAND_PREFIX}{_adapter_suffix(provider_id)}"


def _adapter_provider_ids() -> list[str]:
    """Return command-backed providers configured by the worker image."""
    providers: list[str] = []
    for name, value in os.environ.items():
        if not name.startswith(_ADAPTER_COMMAND_PREFIX) or not value.strip():
            continue
        provider_id = name[len(_ADAPTER_COMMAND_PREFIX) :].lower()
        if _ADAPTER_PROVIDER_RE.fullmatch(provider_id):
            providers.append(provider_id)
    return sorted(set(providers))


def _adapter_capabilities(provider_id: str) -> list[str]:
    raw = os.environ.get(
        f"NODETOOL_PROVIDER_ADAPTER_CAPABILITIES_{_adapter_suffix(provider_id)}", ""
    )
    return sorted({item.strip() for item in raw.split(",") if item.strip()})


def _adapter_display_name(provider_id: str) -> str:
    return os.environ.get(
        f"NODETOOL_PROVIDER_ADAPTER_DISPLAY_NAME_{_adapter_suffix(provider_id)}",
        provider_id,
    ).strip() or provider_id


async def _terminate_adapter(process: asyncio.subprocess.Process) -> None:
    """Terminate an adapter and its subprocess tree."""
    if process.returncode is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except TimeoutError:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        await process.wait()


async def _read_adapter_stderr(stream: asyncio.StreamReader | None) -> str:
    if stream is None:
        return ""
    kept = bytearray()
    while chunk := await stream.read(4096):
        kept.extend(chunk)
        if len(kept) > _ADAPTER_STDERR_LIMIT:
            del kept[: len(kept) - _ADAPTER_STDERR_LIMIT]
    return kept.decode("utf-8", "replace").strip()


async def _run_provider_adapter(
    provider_id: str,
    payload: dict[str, Any],
    request_id: str | None,
    cancel_flags: dict[str, asyncio.Event],
    send_progress: Any,
) -> dict[str, Any]:
    """Run one image-owned provider operation over a JSON-lines subprocess API.

    The authenticated caller selects only an advertised provider and operation;
    the executable itself always comes from worker environment configuration.
    Adapters may emit any number of ``progress`` records followed by exactly one
    ``result`` record. Binary media crosses the boundary through temporary input
    files and an adapter-owned output path, keeping JSON small and WanGP in its
    dependency-isolated interpreter.
    """
    command_text = os.environ.get(_adapter_command_env(provider_id), "").strip()
    command = shlex.split(command_text)
    if not command:
        raise ValueError(f"Provider adapter is unavailable: {provider_id}")

    cancel_event = cancel_flags.get(request_id) if request_id else None
    if cancel_event is None:
        cancel_event = asyncio.Event()
        if request_id:
            cancel_flags[request_id] = cancel_event
    if cancel_event.is_set():
        raise RuntimeError("Provider operation cancelled")
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=os.name == "posix",
    )
    stderr_task = asyncio.create_task(_read_adapter_stderr(process.stderr))
    result: dict[str, Any] | None = None
    try:
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("Provider adapter pipes are unavailable")
        process.stdin.write(json.dumps(payload).encode("utf-8") + b"\n")
        await process.stdin.drain()
        process.stdin.close()

        while True:
            line_task = asyncio.create_task(process.stdout.readline())
            cancel_task = asyncio.create_task(cancel_event.wait())
            done, pending = await asyncio.wait(
                {line_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if cancel_task in done and cancel_task.result():
                await _terminate_adapter(process)
                raise RuntimeError("Provider operation cancelled")

            line = line_task.result()
            if not line:
                break
            if len(line) > _ADAPTER_LINE_LIMIT:
                raise ValueError("Provider adapter emitted an oversized line")
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("Provider adapter emitted invalid JSON") from exc
            if not isinstance(event, dict):
                raise ValueError("Provider adapter event must be a JSON object")
            event_type = event.get("type")
            event_data = event.get("data", {})
            if not isinstance(event_data, dict):
                raise ValueError("Provider adapter event data must be a JSON object")
            if event_type == "progress":
                await send_progress(request_id, event_data)
            elif event_type == "result":
                if result is not None:
                    raise ValueError("Provider adapter emitted multiple results")
                result = event_data
            elif event_type == "error":
                raise RuntimeError(str(event_data.get("error") or "Provider adapter failed"))
            else:
                raise ValueError(f"Provider adapter emitted unknown event type: {event_type!r}")

        returncode = await process.wait()
        stderr = await stderr_task
        if returncode != 0:
            detail = stderr.splitlines()[-1] if stderr else f"exit status {returncode}"
            raise RuntimeError(f"Provider adapter failed: {detail}")
        if result is None:
            raise RuntimeError("Provider adapter exited without a result")
        return result
    except BaseException:
        await _terminate_adapter(process)
        if not stderr_task.done():
            stderr_task.cancel()
        await asyncio.gather(stderr_task, return_exceptions=True)
        raise
    finally:
        if request_id:
            cancel_flags.pop(request_id, None)


def _tts_kwargs(data: dict[str, Any]) -> dict[str, Any]:
    """Build the public TTS provider arguments from a bridge request."""
    result: dict[str, Any] = {
        "text": data["text"],
        "model": data["model"],
    }
    for key in (
        "voice",
        "speed",
        "reference_audio",
        "reference_text",
        "language",
        "instructions",
    ):
        if key in data:
            result[key] = data[key]
    return result


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
                    # Every provider exposed by this handler executes inside
                    # the Python worker. The TypeScript server assigns whether
                    # that worker is local or remote from the active transport.
                    "access": "in_process",
                    "display_name": "Hugging Face Local" if pid == "huggingface" else "MLX",
                }
            )
    for provider_id in _adapter_provider_ids():
        result.append(
            {
                "id": provider_id,
                "capabilities": _adapter_capabilities(provider_id),
                "required_secrets": [],
                "access": "in_process",
                "display_name": _adapter_display_name(provider_id),
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


async def _handle_models(
    data: dict,
    request_id: str | None = None,
    cancel_flags: dict[str, asyncio.Event] | None = None,
    send_progress: Any = None,
) -> dict:
    """Handle provider.models — return available models for a provider."""
    provider_id = data["provider"]
    if provider_id in _adapter_provider_ids():
        if cancel_flags is None or send_progress is None:
            raise RuntimeError("Provider adapter context is unavailable")
        result = await _run_provider_adapter(
            provider_id,
            {
                "operation": "models",
                "provider": provider_id,
                "model_type": data.get("model_type", "language"),
            },
            request_id,
            cancel_flags,
            send_progress,
        )
        models = result.get("models", [])
        if not isinstance(models, list):
            raise ValueError("Provider adapter models result must contain a list")
        return {"models": models}

    provider = _get_provider(data["provider"], data.get("secrets", {}))
    model_type = data.get("model_type", "language")

    getter_map = {
        "language": "get_available_language_models",
        "image": "get_available_image_models",
        "tts": "get_available_tts_models",
        "music": "get_available_audio_models",
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


def _encoded_media_suffix(data: bytes, kind: str) -> str:
    """Choose a useful extension for adapter-owned temporary input files."""
    if kind == "image":
        if data.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return ".webp"
        return ".png"
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        return ".wav"
    if data.startswith(b"fLaC"):
        return ".flac"
    if data.startswith(b"OggS"):
        return ".ogg"
    return ".mp3"


async def _stage_adapter_input(
    temp_dir: str, data: object, kind: str
) -> str:
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise ValueError(f"Provider adapter requires non-empty {kind} bytes")
    encoded = bytes(data)
    path = Path(temp_dir) / f"input-{kind}{_encoded_media_suffix(encoded, kind)}"
    await asyncio.to_thread(path.write_bytes, encoded)
    return str(path)


async def _handle_adapter_media(
    operation: str,
    data: dict[str, Any],
    request_id: str | None,
    cancel_flags: dict[str, asyncio.Event],
    send_progress: Any,
) -> dict[str, Any]:
    """Run a command-backed media operation and return its encoded file."""
    provider_id = str(data.get("provider") or "")
    if provider_id not in _adapter_provider_ids():
        raise ValueError(f"Provider adapter is unavailable: {provider_id}")
    adapter_operation = operation.removeprefix("provider.")
    capability = (
        "text_to_speech_encoded"
        if adapter_operation == "tts_encoded"
        else adapter_operation
    )
    if capability not in _adapter_capabilities(provider_id):
        raise ValueError(f"Provider {provider_id} does not support {capability}")

    cancel_event = asyncio.Event()
    if request_id:
        cancel_flags[request_id] = cancel_event
    try:
        with tempfile.TemporaryDirectory(prefix="nodetool-provider-") as temp_dir:
            payload: dict[str, Any] = {
                "operation": adapter_operation,
                "provider": provider_id,
                "params": dict(data.get("params", {})),
            }
            if adapter_operation in {"image_to_image", "image_to_video"}:
                payload["image_path"] = await _stage_adapter_input(
                    temp_dir, data.get("image", b""), "image"
                )
            elif adapter_operation == "reference_to_video":
                image_values = data.get("reference_images", [])
                video_values = data.get("reference_videos", [])
                if not isinstance(image_values, list) or not isinstance(video_values, list):
                    raise ValueError("provider.reference_to_video reference inputs must be arrays")
                if not image_values and not video_values:
                    raise ValueError("reference_to_video requires at least one reference image or video")
                values: list[tuple[str, bytes]] = []
                for kind, entries in (("image", image_values), ("video", video_values)):
                    for item in entries:
                        if not isinstance(item, (bytes, bytearray)) or not item:
                            raise ValueError(f"provider.reference_to_video requires non-empty {kind} bytes")
                        values.append((kind, bytes(item)))
                total = sum(len(encoded) for _, encoded in values)
                if total > _REFERENCE_INPUT_LIMIT:
                    raise ValueError(
                        f"provider.reference_to_video input is {total} bytes; "
                        f"maximum is {_REFERENCE_INPUT_LIMIT} bytes"
                    )
                image_paths: list[str] = []
                video_paths: list[str] = []
                for index, (kind, encoded) in enumerate(values):
                    if cancel_event.is_set():
                        raise RuntimeError("Provider operation cancelled")
                    suffix = _reference_media_suffix(encoded, kind)
                    if suffix is None:
                        raise ValueError(
                            f"provider.reference_to_video received unsupported "
                            f"{kind} media format"
                        )
                    path = Path(temp_dir) / (
                        f"reference-{kind}-{index}{suffix}"
                    )
                    await asyncio.to_thread(path.write_bytes, encoded)
                    if kind == "image":
                        image_paths.append(str(path))
                    else:
                        video_paths.append(str(path))
                payload["reference_image_paths"] = image_paths
                payload["reference_video_paths"] = video_paths
            if adapter_operation == "tts_encoded":
                params = payload["params"]
                reference_audio = params.pop("referenceAudio", params.pop("reference_audio", None))
                if reference_audio is not None:
                    payload["reference_audio_path"] = await _stage_adapter_input(temp_dir, reference_audio, "audio")
            if cancel_event.is_set():
                raise RuntimeError("Provider operation cancelled")
            result = await _run_provider_adapter(provider_id, payload, request_id, cancel_flags, send_progress)
            output_path = result.get("path")
            if not isinstance(output_path, str) or not output_path:
                raise ValueError("Provider adapter result must contain an output path")
            path = Path(output_path)
            if not path.is_file():
                raise FileNotFoundError(f"Provider adapter output does not exist: {path}")
            output_key = {
                "text_to_image": "image",
                "image_to_image": "image",
                "text_to_video": "video",
                "image_to_video": "video",
                "reference_to_video": "video",
                "text_to_audio": "audio",
                "tts_encoded": "audio",
            }[adapter_operation]
            return {"blobs": {output_key: await asyncio.to_thread(path.read_bytes)}}
    finally:
        if request_id and cancel_flags.get(request_id) is cancel_event:
            cancel_flags.pop(request_id, None)


def _text_to_audio_kwargs(data: dict[str, Any], context: Any) -> dict[str, Any]:
    """Normalize legacy top-level and bridge-style nested music arguments."""
    params = data.get("params")
    source = params if isinstance(params, dict) else data
    kwargs: dict[str, Any] = {
        "prompt": source["prompt"],
        "model": source["model"],
        "context": context,
    }
    optional_fields = {
        "lyrics": ("lyrics",),
        "audio_duration": ("audio_duration", "durationSeconds"),
        "guidance_scale": ("guidance_scale", "guidanceScale"),
        "num_inference_steps": ("num_inference_steps", "numInferenceSteps"),
        "seed": ("seed",),
    }
    for target, aliases in optional_fields.items():
        for alias in aliases:
            if alias in source:
                kwargs[target] = source[alias]
                break
    return kwargs


async def _handle_text_to_audio(data: dict) -> dict:
    """Handle provider.text_to_audio."""
    from nodetool.worker.context_stub import WorkerContext

    provider = _get_provider(data["provider"], data.get("secrets", {}))
    ctx = WorkerContext(secrets=data.get("secrets", {}))
    kwargs = _text_to_audio_kwargs(data, ctx)

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
        blobs = d.get("blobs")
        if data.get("blob_transfer") == "chunked-v1" and isinstance(blobs, dict):
            chunk_size = 4 * 1024 * 1024
            for name, blob in blobs.items():
                if not isinstance(name, str) or not isinstance(blob, bytes):
                    raise TypeError("Provider result blobs must map string names to bytes")
                await transport.send_msg(
                    {
                        "type": "blob.start",
                        "request_id": rid,
                        "data": {"name": name, "size": len(blob)},
                    }
                )
                digest = hashlib.sha256()
                for offset in range(0, len(blob), chunk_size):
                    chunk = blob[offset : offset + chunk_size]
                    digest.update(chunk)
                    await transport.send_msg(
                        {
                            "type": "blob.chunk",
                            "request_id": rid,
                            "data": {"name": name, "offset": offset, "bytes": chunk},
                        }
                    )
                await transport.send_msg(
                    {
                        "type": "blob.end",
                        "request_id": rid,
                        "data": {
                            "name": name,
                            "size": len(blob),
                            "sha256": digest.hexdigest(),
                        },
                    }
                )
            d = {**d, "blobs": {}}
        await transport.send_msg({"type": "result", "request_id": rid, "data": d})

    async def send_error(rid: str | None, error: str, tb: str | None = None) -> None:
        # Omitted rather than null — the JS side's frame schema types
        # `traceback` as an optional string, and a null fails validation.
        data: dict[str, Any] = {"error": error}
        if tb:
            data["traceback"] = tb
        await transport.send_msg({"type": "error", "request_id": rid, "data": data})

    async def send_chunk(rid: str | None, d: dict) -> None:
        await transport.send_msg({"type": "chunk", "request_id": rid, "data": d})

    async def send_progress(rid: str | None, d: dict) -> None:
        await transport.send_msg({"type": "progress", "request_id": rid, "data": d})

    try:
        if msg_type == "provider.list":
            providers = get_available_providers()
            await send_result(request_id, {"providers": providers})

        elif msg_type == "provider.models":
            result = await _handle_models(
                data, request_id, cancel_flags, send_progress
            )
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
            if data.get("provider") in _adapter_provider_ids():
                result = await _handle_adapter_media(
                    msg_type, data, request_id, cancel_flags, send_progress
                )
            else:
                result = await _handle_text_to_image(data)
            await send_result(request_id, result)

        elif msg_type == "provider.image_to_image":
            if data.get("provider") in _adapter_provider_ids():
                result = await _handle_adapter_media(
                    msg_type, data, request_id, cancel_flags, send_progress
                )
            else:
                result = await _handle_image_to_image(data)
            await send_result(request_id, result)

        elif msg_type == "provider.text_to_video":
            if data.get("provider") in _adapter_provider_ids():
                result = await _handle_adapter_media(
                    msg_type, data, request_id, cancel_flags, send_progress
                )
            else:
                result = await _handle_text_to_video(data)
            await send_result(request_id, result)

        elif msg_type in ("provider.image_to_video", "provider.reference_to_video"):
            result = await _handle_adapter_media(
                msg_type, data, request_id, cancel_flags, send_progress
            )
            await send_result(request_id, result)

        elif msg_type == "provider.text_to_audio":
            if data.get("provider") in _adapter_provider_ids():
                result = await _handle_adapter_media(
                    msg_type, data, request_id, cancel_flags, send_progress
                )
            else:
                result = await _handle_text_to_audio(data)
            await send_result(request_id, result)

        elif msg_type == "provider.tts_encoded":
            result = await _handle_adapter_media(
                msg_type, data, request_id, cancel_flags, send_progress
            )
            await send_result(request_id, result)

        elif msg_type == "provider.tts":
            cancel_event = asyncio.Event()
            if request_id:
                cancel_flags[request_id] = cancel_event
            try:
                provider = _get_provider(data["provider"], data.get("secrets", {}))
                kwargs_tts = _tts_kwargs(data)
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
