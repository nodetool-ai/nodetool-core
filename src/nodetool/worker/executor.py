"""
Instantiate a Python BaseNode, execute it, and collect final outputs.
"""
import asyncio
import os
import tempfile
from types import UnionType
from typing import Any, Union, get_args, get_origin

from nodetool.metadata.types import AssetRef, AudioRef, ImageRef, Model3DRef, VideoRef
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


async def execute_node(
    node_type: str,
    fields: dict[str, Any],
    secrets: dict[str, str],
    input_blobs: dict[str, bytes | list[bytes]],
    cancel_event: asyncio.Event | None = None,
) -> dict[str, Any]:
    """Execute a single Python node and return outputs + blobs."""
    node_class = NODE_BY_TYPE.get(node_type)
    if node_class is None:
        raise ValueError(f"Unknown node type: {node_type}")

    async with ResourceScope():
        ctx = WorkerContext(
            secrets=secrets,
            cancel_event=cancel_event,
        )

        # Write input blobs to temp files for URI resolution
        temp_dir = tempfile.mkdtemp(prefix="nodetool_worker_")
        input_ref_uris: dict[str, str | list[str]] = {}
        try:
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

            # Lifecycle: pre_process -> preload_model -> move_to_device -> execute
            await node.pre_process(ctx)
            await node.preload_model(ctx)
            await node.move_to_device(ctx.device)
            if node.is_streaming_output():
                result = await _collect_streaming_outputs(node, ctx)
                outputs, blobs = _extract_named_outputs(result, ctx)
            else:
                result = await node.process(ctx)
                outputs, blobs = _extract_outputs(result, ctx)
            return {"outputs": outputs, "blobs": blobs}
        finally:
            import shutil
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


async def _collect_streaming_outputs(node: BaseNode, ctx: WorkerContext) -> dict[str, Any]:
    """Collect the final value emitted for each slot from a streaming node."""
    outputs: dict[str, Any] = {}
    async for item in node.gen_process(ctx):
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


def _extract_named_outputs(
    result: dict[str, Any],
    ctx: WorkerContext,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Serialize a named-output mapping without wrapping it in ``output``."""
    outputs = {key: _serialize_value(value) for key, value in result.items()}
    return outputs, ctx.get_output_blobs()


def _serialize_value(value: Any) -> Any:
    """Convert a value to JSON/msgpack-safe form."""
    if isinstance(value, ASSET_REF_TYPES):
        return {"uri": value.uri, "type": type(value).__name__}
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
