"""
Instantiate a Python BaseNode, set fields, call process(), collect results.
"""
import asyncio
import os
import tempfile
from typing import Any

from nodetool.workflows.base_node import NODE_BY_TYPE
from nodetool.metadata.types import AssetRef, ImageRef, AudioRef, VideoRef, Model3DRef
from nodetool.runtime.resources import ResourceScope

from nodetool.worker.context_stub import WorkerContext

# Asset ref types that should be extracted as blobs
ASSET_REF_TYPES = (ImageRef, AudioRef, VideoRef, Model3DRef, AssetRef)


async def execute_node(
    node_type: str,
    fields: dict[str, Any],
    secrets: dict[str, str],
    input_blobs: dict[str, bytes],
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
        input_ref_uris: dict[str, str] = {}
        blob_ref_map: dict[str, str] = {}
        try:
            for name, data in input_blobs.items():
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
                    annotation = field_info.annotation
                    type_name = getattr(annotation, "__name__", "")
                    uri = input_ref_uris.get(field_name, f"blob://{field_name}")
                    if type_name == "ImageRef":
                        ref = ImageRef(uri=uri)
                    elif type_name == "AudioRef":
                        ref = AudioRef(uri=uri)
                    elif type_name == "VideoRef":
                        ref = VideoRef(uri=uri)
                    elif type_name == "Model3DRef":
                        ref = Model3DRef(uri=uri)
                    else:
                        ref = AssetRef(uri=uri)
                    blob_ref_map[uri] = field_name
                    resolved_fields[field_name] = ref

            for key, value in resolved_fields.items():
                if hasattr(node, key):
                    # Use Pydantic validation for complex fields (e.g. HFTextClassification)
                    field_info = node.__class__.model_fields.get(key)
                    if field_info and isinstance(value, dict):
                        from pydantic import BaseModel
                        annotation = field_info.annotation
                        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                            value = annotation.model_validate(value)
                    setattr(node, key, value)

            # Lifecycle: pre_process -> preload_model -> move_to_device -> process
            await node.pre_process(ctx)
            await node.preload_model(ctx)
            await node.move_to_device(ctx.device)
            result = await node.process(ctx)

            # Extract outputs
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

    # Default: single output slot named "output"
    return {"output": _serialize_value(result)}, output_blobs


def _serialize_value(value: Any) -> Any:
    """Convert a value to JSON-safe form."""
    if isinstance(value, ASSET_REF_TYPES):
        return {"uri": value.uri, "type": type(value).__name__}
    from enum import Enum
    if isinstance(value, Enum):
        return value.value
    from pydantic import BaseModel
    if isinstance(value, BaseModel):
        return value.model_dump()
    return value
