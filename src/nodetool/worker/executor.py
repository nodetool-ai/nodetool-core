"""
Instantiate a Python BaseNode, execute it, and collect final outputs.
"""
import asyncio
import os
import shutil
import tempfile
from collections.abc import AsyncGenerator
from types import UnionType
from typing import Any, Awaitable, Callable, Union, get_args, get_origin

from nodetool.metadata.types import (
    AssetRef,
    AudioRef,
    ImageRef,
    Model3DRef,
    TypeToName,
    VideoRef,
)
from nodetool.runtime.resources import ResourceScope
from nodetool.worker.context_stub import WorkerContext
from nodetool.workflows.base_node import NODE_BY_TYPE, BaseNode

# Asset ref types that should be extracted as blobs
ASSET_REF_TYPES = (ImageRef, AudioRef, VideoRef, Model3DRef, AssetRef)
REF_TYPE_BY_CLASS_NAME = {
    "ImageRef": "image",
    "AudioRef": "audio",
    "VideoRef": "video",
    "Model3DRef": "model_3d",
    "AssetRef": "asset",
}


def _get_asset_ref_type(annotation: Any) -> str:
    """Infer the asset type literal expected by BaseNode.assign_property()."""
    if annotation is None:
        return "asset"

    origin = get_origin(annotation)
    if origin in (list, tuple, set):
        args = get_args(annotation)
        return _get_asset_ref_type(args[0] if args else None)
    if origin in (UnionType, Union):
        for arg in get_args(annotation):
            if arg is type(None):
                continue
            ref_type = _get_asset_ref_type(arg)
            if ref_type != "asset":
                return ref_type
        return "asset"

    type_name = getattr(annotation, "__name__", "")
    return REF_TYPE_BY_CLASS_NAME.get(type_name, "asset")


async def _emit_pending_progress(
    ctx: WorkerContext,
    emit_progress: Callable[[dict[str, Any]], Awaitable[None]] | None,
) -> None:
    """Forward any NodeProgress messages the node queued during execution."""
    if emit_progress is None:
        ctx.drain_progress()
        return

    for msg in ctx.drain_progress():
        progress = getattr(msg, "progress", None)
        total = getattr(msg, "total", None)
        if progress is None:
            current = getattr(msg, "current", None)
            if current is not None:
                progress = current
        if progress is None:
            progress = 0
        if total is None:
            total = 100
        await emit_progress({
            "progress": progress,
            "total": total,
            "message": getattr(msg, "message", None),
        })


async def _prepare_node(
    node_class: type[BaseNode],
    fields: dict[str, Any],
    input_blobs: dict[str, bytes | list[bytes]],
    temp_dir: str,
    ctx: WorkerContext,
    emit_progress: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> BaseNode:
    """Instantiate a node, assign fields/blobs, and run its preprocessing lifecycle.

    Shared by both the unary (`execute_node`) and streaming (`execute_node_stream`)
    entry points so they resolve inputs identically.
    """
    # Write input blobs to temp files for URI resolution
    input_ref_uris: dict[str, str | list[str]] = {}
    for name, data in input_blobs.items():
        if isinstance(data, list):
            uris: list[str] = []
            for index, item in enumerate(data):
                path = os.path.join(temp_dir, f"input_{name}_{index}")
                with open(path, "wb") as f:
                    f.write(item)
                uris.append(f"file://{path}")
            input_ref_uris[name] = uris
        else:
            path = os.path.join(temp_dir, f"input_{name}")
            with open(path, "wb") as f:
                f.write(data)
            input_ref_uris[name] = f"file://{path}"

    # Instantiate node
    node = node_class()

    # Set fields — convert blob references for asset fields
    resolved_fields = dict(fields)
    for field_name, field_info in node.__class__.model_fields.items():
        if field_name in input_blobs:
            uri = input_ref_uris.get(field_name, f"blob://{field_name}")
            ref_type = _get_asset_ref_type(field_info.annotation)
            if isinstance(uri, list):
                resolved_fields[field_name] = [
                    {"uri": item, "type": ref_type} for item in uri
                ]
            else:
                resolved_fields[field_name] = {
                    "uri": uri,
                    "type": ref_type,
                }

    for key, value in resolved_fields.items():
        error = node.assign_property(key, value)
        if error:
            raise ValueError(error)

    # Lifecycle: pre_process -> preload_model -> move_to_device
    await node.pre_process(ctx)
    await _emit_pending_progress(ctx, emit_progress)
    await node.preload_model(ctx)
    await _emit_pending_progress(ctx, emit_progress)
    await node.move_to_device(ctx.device)
    await _emit_pending_progress(ctx, emit_progress)
    return node


async def execute_node(
    node_type: str,
    fields: dict[str, Any],
    secrets: dict[str, str],
    input_blobs: dict[str, bytes | list[bytes]],
    cancel_event: asyncio.Event | None = None,
    emit_progress: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    emit_chunk: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """Execute a single Python node and return outputs + blobs.

    Streaming nodes (``is_streaming_output() == True``) emit each yielded item
    via ``emit_chunk`` when provided; the return value still carries the final
    aggregated outputs so callers that only need the last value can ignore the
    chunks.
    """
    node_class = NODE_BY_TYPE.get(node_type)
    if node_class is None:
        raise ValueError(f"Unknown node type: {node_type}")

    async with ResourceScope():
        ctx = WorkerContext(
            secrets=secrets,
            cancel_event=cancel_event,
        )

        temp_dir = tempfile.mkdtemp(prefix="nodetool_worker_")
        try:
            node = await _prepare_node(
                node_class, fields, input_blobs, temp_dir, ctx, emit_progress
            )
            if node.is_streaming_output():
                if emit_chunk is not None:
                    result = await _stream_streaming_outputs(
                        node, ctx, emit_progress, emit_chunk
                    )
                else:
                    result = await _collect_streaming_outputs(node, ctx, emit_progress)
                outputs, blobs = _extract_named_outputs(result, ctx)
            else:
                result = await node.process(ctx)
                await _emit_pending_progress(ctx, emit_progress)
                outputs, blobs = _extract_outputs(result, ctx)
            return {"outputs": outputs, "blobs": blobs}
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


async def execute_node_stream(
    node_type: str,
    fields: dict[str, Any],
    secrets: dict[str, str],
    input_blobs: dict[str, bytes | list[bytes]],
    cancel_event: asyncio.Event | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Execute a streaming node, yielding {"outputs", "blobs"} for each emitted item.

    Drives the node's ``gen_process`` and serializes every yielded mapping the same
    way a unary streaming collect would, but emits each one incrementally so the
    caller can forward it as a `chunk` frame over the stdio bridge.
    """
    node_class = NODE_BY_TYPE.get(node_type)
    if node_class is None:
        raise ValueError(f"Unknown node type: {node_type}")

    async with ResourceScope():
        ctx = WorkerContext(
            secrets=secrets,
            cancel_event=cancel_event,
        )

        temp_dir = tempfile.mkdtemp(prefix="nodetool_worker_")
        try:
            node = await _prepare_node(node_class, fields, input_blobs, temp_dir, ctx)
            async for item in node.gen_process(ctx):
                if not isinstance(item, dict):
                    raise TypeError(
                        "Streaming worker nodes must yield dictionaries mapping output names to values."
                    )
                outputs, blobs = _extract_named_outputs(item, ctx)
                yield {"outputs": outputs, "blobs": blobs}
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _extract_outputs(
    result: Any,
    ctx: WorkerContext,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Split a node's return value into scalar outputs and binary blobs.

    Single-output nodes always get their result wrapped as {"output": value}.
    Only nodes returning a dict with AssetRef blob values need special handling.
    """
    output_blobs = ctx.get_output_blobs()

    if isinstance(result, ASSET_REF_TYPES) and result.uri and result.uri.startswith("blob://"):
        blob_key = result.uri[len("blob://"):]
        return {}, {"output": output_blobs.get(blob_key, b"")}

    # Check if result is a dict with blob values that need extraction
    if isinstance(result, dict):
        has_blobs = any(
            isinstance(v, ASSET_REF_TYPES) and v.uri and v.uri.startswith("blob://")
            for v in result.values()
        )
        if has_blobs:
            # Multi-output with blobs: each key is a separate output slot
            outputs = {}
            blobs = {}
            for key, value in result.items():
                if isinstance(value, ASSET_REF_TYPES) and value.uri and value.uri.startswith("blob://"):
                    blob_key = value.uri[len("blob://"):]
                    if blob_key in output_blobs:
                        blobs[key] = output_blobs[blob_key]
                else:
                    outputs[key] = _serialize_value(value)
            return outputs, blobs

    # Multi-output dict: if result is a dict with named output keys
    # (not a plain data dict), split into separate output handles.
    # A dict with an "output" key is a single-output wrapper — don't split.
    if isinstance(result, dict) and len(result) > 1 and "output" not in result:
        outputs = {key: _serialize_value(value) for key, value in result.items()}
        return outputs, output_blobs

    # Default: single output slot named "output"
    return {"output": _serialize_value(result)}, output_blobs


async def _collect_streaming_outputs(
    node: BaseNode,
    ctx: WorkerContext,
    emit_progress: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """Collect the final value emitted for each slot from a streaming node."""
    outputs: dict[str, Any] = {}
    async for item in node.gen_process(ctx):
        await _emit_pending_progress(ctx, emit_progress)
        if not isinstance(item, dict):
            raise TypeError(
                "Streaming worker nodes must yield dictionaries mapping output names to values."
            )
        for slot_name, value in item.items():
            if not isinstance(slot_name, str):
                raise TypeError(
                    "Streaming worker nodes must use string keys for output names."
                )
            if value is not None:
                outputs[slot_name] = value
    return outputs


async def _stream_streaming_outputs(
    node: BaseNode,
    ctx: WorkerContext,
    emit_progress: Callable[[dict[str, Any]], Awaitable[None]] | None,
    emit_chunk: Callable[[dict[str, Any]], Awaitable[None]],
) -> dict[str, Any]:
    """Emit each streaming item while still collecting final outputs."""
    outputs: dict[str, Any] = {}
    async for item in node.gen_process(ctx):
        await _emit_pending_progress(ctx, emit_progress)
        if not isinstance(item, dict):
            raise TypeError(
                "Streaming worker nodes must yield dictionaries mapping output names to values."
            )

        partial: dict[str, Any] = {}
        for slot_name, value in item.items():
            if not isinstance(slot_name, str):
                raise TypeError(
                    "Streaming worker nodes must use string keys for output names."
                )
            if value is not None:
                outputs[slot_name] = value
                partial[slot_name] = value

        if partial:
            chunk_outputs, chunk_blobs = _extract_named_outputs(partial, ctx)
            await emit_chunk({"outputs": chunk_outputs, "blobs": chunk_blobs})

    return outputs


def _extract_named_outputs(
    result: dict[str, Any],
    ctx: WorkerContext,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Serialize a named-output mapping and extract blob-backed asset refs."""
    output_blobs = ctx.get_output_blobs()
    outputs: dict[str, Any] = {}
    blobs: dict[str, bytes] = {}

    for key, value in result.items():
        if (
            isinstance(value, ASSET_REF_TYPES)
            and value.uri
            and value.uri.startswith("blob://")
        ):
            blob_key = value.uri[len("blob://"):]
            if blob_key in output_blobs:
                blobs[key] = output_blobs[blob_key]
            continue

        outputs[key] = _serialize_value(value)

    return outputs, blobs


def _serialize_value(value: Any) -> Any:
    """Convert a value to JSON/msgpack-safe form."""
    if isinstance(value, ASSET_REF_TYPES):
        return {
            "uri": value.uri,
            "type": TypeToName.get(type(value), type(value).__name__),
        }
    from enum import Enum
    if isinstance(value, Enum):
        return value.value
    from pydantic import BaseModel
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    return value
