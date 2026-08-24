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

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    AssetRef,
    AudioRef,
    ImageRef,
    Model3DRef,
    TypeToName,
    VideoRef,
)
from nodetool.ml.core.model_manager import ModelManager
from nodetool.runtime.resources import ResourceScope
from nodetool.worker.context_stub import WorkerContext
from nodetool.worker.job_registry import JobRegistry
from nodetool.workflows.base_node import NODE_BY_TYPE, BaseNode

log = get_logger(__name__)

# Asset ref types that should be extracted as blobs
ASSET_REF_TYPES = (ImageRef, AudioRef, VideoRef, Model3DRef, AssetRef)
REF_TYPE_BY_CLASS_NAME = {
    "ImageRef": "image",
    "AudioRef": "audio",
    "VideoRef": "video",
    "Model3DRef": "model_3d",
    "AssetRef": "asset",
}

# How often the background pump flushes queued NodeProgress messages while a
# node's lifecycle / process methods are running. 50ms keeps progress feeling
# real-time without burning CPU on the queue check.
_PROGRESS_POLL_INTERVAL = 0.05


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
    emit_update: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> None:
    """Forward messages the node queued during execution.

    NodeProgress goes to ``emit_progress`` in the flat shape the bridge's
    ``progress`` frame expects. Preview/log/binary updates go to
    ``emit_update`` as their serialized model dump (discriminated by their
    ``type`` field) so live previews and logs can reach the UI mid-render.
    """
    from nodetool.workflows.types import NodeProgress

    if emit_progress is None and emit_update is None:
        ctx.drain_messages()
        return

    for msg in ctx.drain_messages():
        if isinstance(msg, NodeProgress):
            if emit_progress is None:
                continue
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
            await emit_progress(
                {
                    "progress": progress,
                    "total": total,
                    "message": getattr(msg, "message", None),
                }
            )
        elif emit_update is not None:
            await emit_update(_serialize_value(msg))


def _start_progress_pump(
    ctx: WorkerContext,
    emit_progress: Callable[[dict[str, Any]], Awaitable[None]] | None,
    emit_update: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> tuple[asyncio.Task, asyncio.Event]:
    """Spawn a background task that flushes queued progress in near-real-time.

    Runs concurrently with the node's lifecycle and process methods so progress
    posted mid-execution doesn't have to wait for a synchronous boundary. The
    pump also performs a final drain after the stop signal, so progress queued
    just before completion (or before an exception propagates) isn't lost.

    Caller must invoke ``_stop_progress_pump`` in a ``finally`` block.
    """
    stop = asyncio.Event()

    async def pump() -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=_PROGRESS_POLL_INTERVAL)
            except TimeoutError:
                pass
            try:
                await _emit_pending_progress(ctx, emit_progress, emit_update)
            except Exception:
                # Don't let a transient transport error kill the pump or mask
                # the node's own error — just drop this batch.
                pass
        # Final drain after stop so anything queued during shutdown still ships.
        try:
            await _emit_pending_progress(ctx, emit_progress, emit_update)
        except Exception:
            pass

    task = asyncio.create_task(pump())
    return task, stop


async def _stop_progress_pump(
    handle: tuple[asyncio.Task, asyncio.Event] | None,
) -> None:
    if handle is None:
        return
    task, stop = handle
    stop.set()
    try:
        await task
    except BaseException:
        # Pump errors must not mask the caller's error path.
        pass


def read_run_identity(data: dict[str, Any]) -> dict[str, Any]:
    """Extract the v4 run-identity kwargs from an ``execute`` payload.

    Every key is optional and absent keys must reproduce the pre-v4 behaviour
    exactly: the JS side omits a field it cannot name rather than sending
    ``null``, so ``data.get("node_id") is None`` is the normal old-client path,
    not an error. Wrong-typed values are dropped for the same reason — a peer
    that sends a number where a string belongs should degrade to "no identity",
    not fail an otherwise valid execution.

    Returns kwargs suitable for ``execute_node(**identity)``; keys the payload
    did not carry are simply ``None``.
    """

    def _str(key: str) -> str | None:
        value = data.get(key)
        return value if isinstance(value, str) and value else None

    raw_vram = data.get("requires_vram_gb")
    # bool is an int subclass; a stray True must not become 1.0 GiB.
    vram = (
        float(raw_vram)
        if isinstance(raw_vram, (int, float)) and not isinstance(raw_vram, bool) and raw_vram > 0
        else None
    )

    return {
        "node_id": _str("node_id"),
        "job_id": _str("job_id"),
        "workflow_id": _str("workflow_id"),
        "user_id": _str("user_id"),
        "requires_vram_gb": vram,
    }


async def _prepare_node(
    node_class: type[BaseNode],
    fields: dict[str, Any],
    input_blobs: dict[str, bytes | list[bytes]],
    temp_dir: str,
    ctx: WorkerContext,
    node_id: str | None = None,
) -> BaseNode:
    """Instantiate a node, assign fields/blobs, and run its preprocessing lifecycle.

    Shared by both the unary (`execute_node`) and streaming (`execute_node_stream`)
    entry points so they resolve inputs identically. Progress messages queued by
    the lifecycle methods are flushed by the background pump started in
    ``execute_node`` — no inline drain is needed here.
    """
    # Write input blobs into the workspace (`temp_dir`) and address them
    # workspace-relative. `ProcessingContext.resolve_workspace_path` re-roots
    # every absolute path at the workspace to block traversal, so a
    # `file:///private/var/.../input_audio` URI resolves to
    # `<workspace>/private/var/.../input_audio` and the node never finds its
    # own input. `file:///input_audio` resolves back to the file just written.
    input_ref_uris: dict[str, str | list[str]] = {}
    for name, data in input_blobs.items():
        if isinstance(data, list):
            uris: list[str] = []
            for index, item in enumerate(data):
                filename = f"input_{name}_{index}"
                with open(os.path.join(temp_dir, filename), "wb") as f:
                    f.write(item)
                uris.append(f"file:///{filename}")
            input_ref_uris[name] = uris
        else:
            filename = f"input_{name}"
            with open(os.path.join(temp_dir, filename), "wb") as f:
                f.write(data)
            input_ref_uris[name] = f"file:///{filename}"

    # Instantiate node. ``node_id`` is the graph id the JS side sends on
    # ``execute`` (bridge protocol v4); before v4 every node was built with no
    # id, so ``self._id`` was "" for all of them and a node calling
    # ``set_model(self._id, …)`` registered into a single shared bucket. Absent
    # (a pre-v4 client) it stays "", exactly as before.
    node = node_class(id=node_id) if node_id else node_class()

    # Set fields — convert blob references for asset fields
    resolved_fields = dict(fields)
    for field_name, field_info in node.__class__.model_fields.items():
        if field_name in input_blobs:
            uri = input_ref_uris.get(field_name, f"blob://{field_name}")
            ref_type = _get_asset_ref_type(field_info.annotation)
            if isinstance(uri, list):
                resolved_fields[field_name] = [{"uri": item, "type": ref_type} for item in uri]
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
    await node.preload_model(ctx)
    await node.move_to_device(ctx.device)
    return node


async def execute_node(
    node_type: str,
    fields: dict[str, Any],
    secrets: dict[str, str],
    input_blobs: dict[str, bytes | list[bytes]],
    cancel_event: asyncio.Event | None = None,
    emit_progress: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    emit_chunk: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    emit_update: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    node_id: str | None = None,
    job_id: str | None = None,
    workflow_id: str | None = None,
    user_id: str | None = None,
    requires_vram_gb: float | None = None,
) -> dict[str, Any]:
    """Execute a single Python node and return outputs + blobs.

    For streaming nodes (``is_streaming_output() == True``), each item emitted
    by ``gen_process`` is forwarded via ``emit_chunk`` when provided; the
    return value still carries the aggregated final outputs so callers that
    only need the last value can ignore the chunks.

    Progress posted to ``ctx.message_queue`` is forwarded to ``emit_progress``
    in near-real-time by a background pump (every 50ms), and one final drain
    runs after the node finishes — including on exception paths.

    The five run-identity arguments arrive from the bridge's ``execute`` frame
    (protocol v4) and are all optional — every one of them absent must
    reproduce the pre-v4 behaviour exactly, because that is the shape a client
    that cannot name them still sends:

    - ``node_id`` becomes the node's ``_id``, which is what turns
      ``ModelManager._models_by_node`` from a single ``""`` bucket into a real
      map and gives ``release_nodes()`` something to release.
    - ``job_id`` pairs the execution with the ``job.start`` / ``job.end``
      boundary via :class:`~nodetool.worker.job_registry.JobRegistry`.
    - ``workflow_id`` / ``user_id`` populate the ``WorkerContext``.
    - ``requires_vram_gb`` gives the pre-execution reclaim pass a real number
      to target instead of only a percentage threshold.
    """
    node_class = NODE_BY_TYPE.get(node_type)
    if node_class is None:
        raise ValueError(f"Unknown node type: {node_type}")

    # ``execution_scope`` is synchronous, so it nests inside the ``async with``
    # rather than joining it. It pins every model this execution touches for as
    # long as the node is running, which is what lets the reclaim pass below —
    # and any reclaim triggered by a concurrent execution — stay safe.
    async with ResourceScope():
        with ModelManager.execution_scope():
            ctx = WorkerContext(
                secrets=secrets,
                cancel_event=cancel_event,
                workflow_id=workflow_id,
                user_id=user_id,
                job_id=job_id,
            )
            # Attribute this node to its run so job.end has something to
            # retire. A no-op when either id is absent (the pre-v4 path).
            JobRegistry.note_execution(job_id, node_id)
            temp_dir: str | None = None
            pump_handle: tuple[asyncio.Task, asyncio.Event] | None = None
            node: BaseNode | None = None
            try:
                # Both inside the try so a failure here (e.g. mkdtemp on a
                # read-only FS) still runs the cleanup in `finally`.
                temp_dir = tempfile.mkdtemp(prefix="nodetool_worker_")
                # The temp dir is this run's workspace, not just scratch space:
                # `_prepare_node` writes every input blob into it and hands the
                # node a `file://` URI, and resolving one goes through
                # `ProcessingContext.resolve_workspace_path`, which refuses
                # every path when no workspace is assigned. Without this a node
                # sees its media input as "No workspace is assigned" instead of
                # the bytes the caller sent. Assigned here rather than passed to
                # the constructor so the mkdtemp call stays inside the `try`.
                ctx.workspace_dir = temp_dir
                pump_handle = _start_progress_pump(ctx, emit_progress, emit_update)
                # Reclaim before the node loads anything, rather than only after
                # an OOM has already been raised. This scope has touched no
                # models yet, so nothing of ours can be evicted, and models
                # pinned by concurrently executing nodes are protected by their
                # own scopes. The call is threshold- and cooldown-gated, so it
                # is a no-op unless VRAM is actually under pressure.
                #
                # ``requires_vram_gb`` (protocol v4) is what the worker itself
                # reported for this node type at ``discover``, echoed back by
                # the JS side. With it the pass targets the amount the node is
                # about to load and reclaims once, correctly; without it it
                # falls back to the percentage threshold, which can only
                # trickle because it has no idea what is coming.
                ModelManager.free_vram_if_needed(
                    reason=f"Preparing to execute node {node_type}",
                    required_free_gb=requires_vram_gb,
                )
                node = await _prepare_node(
                    node_class, fields, input_blobs, temp_dir, ctx, node_id=node_id
                )
                ctx.raise_if_cancelled()
                if node.is_streaming_output():
                    if emit_chunk is not None:
                        result = await _stream_streaming_outputs(node, ctx, emit_chunk)
                    else:
                        result = await _collect_streaming_outputs(node, ctx)
                    outputs, blobs = _extract_named_outputs(result, ctx)
                else:
                    result = await node.process(ctx)
                    outputs, blobs = _extract_outputs(result, ctx)
                return {"outputs": outputs, "blobs": blobs}
            finally:
                if node is not None:
                    # The node's teardown hook — the counterpart to
                    # pre_process/preload_model/move_to_device. Runs on the
                    # error path too, and its own failure must never mask the
                    # node's result or error. Ordered before the pump stops so
                    # progress posted during teardown still ships.
                    try:
                        await node.finalize(ctx)
                    except Exception:
                        log.exception("Error finalizing node %s", node_type)
                await _stop_progress_pump(pump_handle)
                if temp_dir is not None:
                    shutil.rmtree(temp_dir, ignore_errors=True)


async def execute_node_stream(
    node_type: str,
    fields: dict[str, Any],
    secrets: dict[str, str],
    input_blobs: dict[str, bytes | list[bytes]],
    cancel_event: asyncio.Event | None = None,
    node_id: str | None = None,
    job_id: str | None = None,
    workflow_id: str | None = None,
    user_id: str | None = None,
    requires_vram_gb: float | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Execute a streaming node, yielding each emitted ``{"outputs", "blobs"}``.

    Thin adapter around ``execute_node`` so the two entry points can never
    diverge in their chunk semantics: this generator yields exactly what
    ``execute_node`` would have passed to ``emit_chunk`` — including how it
    handles the v4 run identity, which is forwarded verbatim.
    """
    queue: asyncio.Queue = asyncio.Queue()
    sentinel: object = object()

    async def emit_chunk(chunk: dict[str, Any]) -> None:
        await queue.put(chunk)

    async def runner() -> None:
        try:
            await execute_node(
                node_type=node_type,
                fields=fields,
                secrets=secrets,
                input_blobs=input_blobs,
                cancel_event=cancel_event,
                emit_chunk=emit_chunk,
                node_id=node_id,
                job_id=job_id,
                workflow_id=workflow_id,
                user_id=user_id,
                requires_vram_gb=requires_vram_gb,
            )
        finally:
            queue.put_nowait(sentinel)

    task = asyncio.create_task(runner())
    try:
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            yield item
        # Propagate any error from execute_node back to the consumer.
        await task
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except BaseException:
                pass


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
        blob_key = result.uri[len("blob://") :]
        return {}, {"output": output_blobs.get(blob_key, b"")}

    # Check if result is a dict with blob values that need extraction
    if isinstance(result, dict):
        has_blobs = any(
            isinstance(v, ASSET_REF_TYPES) and v.uri and v.uri.startswith("blob://") for v in result.values()
        )
        if has_blobs:
            # Multi-output with blobs: each key is a separate output slot
            outputs = {}
            blobs = {}
            for key, value in result.items():
                if isinstance(value, ASSET_REF_TYPES) and value.uri and value.uri.startswith("blob://"):
                    blob_key = value.uri[len("blob://") :]
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
) -> dict[str, Any]:
    """Collect the final value emitted for each slot from a streaming node."""
    outputs: dict[str, Any] = {}
    async for item in node.gen_process(ctx):
        # Cooperative cancellation between chunks: a cancel request must not
        # wait for the whole stream to finish before taking effect.
        ctx.raise_if_cancelled()
        if not isinstance(item, dict):
            raise TypeError("Streaming worker nodes must yield dictionaries mapping output names to values.")
        for slot_name, value in item.items():
            if not isinstance(slot_name, str):
                raise TypeError("Streaming worker nodes must use string keys for output names.")
            if value is not None:
                outputs[slot_name] = value
    return outputs


async def _stream_streaming_outputs(
    node: BaseNode,
    ctx: WorkerContext,
    emit_chunk: Callable[[dict[str, Any]], Awaitable[None]],
) -> dict[str, Any]:
    """Emit each gen_process item via emit_chunk while collecting final outputs.

    Chunks carry the raw yielded mapping (including ``None`` placeholders for
    not-yet-final slots), matching the semantics of ``execute_node_stream``.
    Only non-``None`` values feed the aggregated final result.
    """
    outputs: dict[str, Any] = {}
    async for item in node.gen_process(ctx):
        ctx.raise_if_cancelled()
        if not isinstance(item, dict):
            raise TypeError("Streaming worker nodes must yield dictionaries mapping output names to values.")
        for slot_name, value in item.items():
            if not isinstance(slot_name, str):
                raise TypeError("Streaming worker nodes must use string keys for output names.")
            if value is not None:
                outputs[slot_name] = value

        # Drain the blobs captured for this chunk: once emitted they are on the
        # wire, so holding them would grow the context for the whole request.
        chunk_outputs, chunk_blobs = _extract_named_outputs(item, ctx, drain=True)
        await emit_chunk({"outputs": chunk_outputs, "blobs": chunk_blobs})

    return outputs


def _extract_named_outputs(
    result: dict[str, Any],
    ctx: WorkerContext,
    drain: bool = False,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Serialize a named-output mapping and extract blob-backed asset refs.

    With ``drain=True`` the context's captured blobs are consumed, releasing
    their memory. Callers that need the full set at the end (the non-streaming
    and collect-only paths) must leave it ``False``.
    """
    output_blobs = ctx.take_output_blobs() if drain else ctx.get_output_blobs()
    outputs: dict[str, Any] = {}
    blobs: dict[str, bytes] = {}

    for key, value in result.items():
        if isinstance(value, ASSET_REF_TYPES) and value.uri and value.uri.startswith("blob://"):
            blob_key = value.uri[len("blob://") :]
            if blob_key in output_blobs:
                blobs[key] = output_blobs[blob_key]
            continue

        outputs[key] = _serialize_value(value)

    return outputs, blobs


def msgpack_default(obj: Any) -> Any:
    """Fallback encoder for ``msgpack.packb``.

    ``_serialize_value`` preserves ``datetime``/``Decimal``/``UUID`` and similar
    types as native Python objects (python-mode ``model_dump``), but msgpack
    cannot pack them. Transports call ``packb(msg, default=msgpack_default,
    datetime=True)`` so ``datetime`` becomes a msgpack timestamp ext type and
    this hook converts the rest to msgpack-native equivalents.
    """
    import decimal
    import uuid

    if isinstance(obj, decimal.Decimal):
        # Encode as string, not float, so the TS side can parse it losslessly
        # (float() would silently drop precision for cost/token accounting).
        return str(obj)
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    # date/time/datetime and any object exposing isoformat().
    isoformat = getattr(obj, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    # numpy scalars (np.float32, np.int64, ...) reach the packer inside
    # dataframe rows. `.item()` gives the Python primitive; it raises for a
    # multi-element array, which has no primitive form and stays unsupported.
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return item()
        except (ValueError, TypeError):
            pass
    raise TypeError(f"Object of type {type(obj).__name__} is not msgpack-serializable")


def _is_binary_payload(payload: Any) -> bool:
    """True when an AssetRef's ``data`` holds raw bytes rather than a payload."""
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return True
    return isinstance(payload, list) and any(
        isinstance(item, (bytes, bytearray, memoryview)) for item in payload
    )


def _strip_binary_payloads(obj: Any) -> Any:
    """Drop every raw-bytes ``data`` field from a dumped ref, at any depth.

    Nesting is real: ``Model3DRef.texture_files`` is a list of ``ImageRef`` and
    ``material_file`` is one too, so a top-level check would still inline
    texture bytes into the frame.
    """
    if isinstance(obj, dict):
        out = {k: _strip_binary_payloads(v) for k, v in obj.items()}
        if _is_binary_payload(out.get("data")):
            out["data"] = None
        return out
    if isinstance(obj, list):
        return [_strip_binary_payloads(item) for item in obj]
    return obj


def _serialize_asset_ref(value: AssetRef) -> dict[str, Any]:
    """Serialize an asset ref, keeping every field except raw bytes.

    Bytes never travel inline. A ref whose payload is real bytes carries a
    ``blob://`` uri, and ``_extract_outputs`` moves those bytes into the frame's
    separate blobs map; copying them into ``data`` as well would duplicate a
    megabyte-scale payload in every frame.

    Nothing else may be dropped. This used to reduce every ref to ``uri`` and
    ``type``, which is right only for a ref whose payload is in a blob.
    ``DataframeRef`` overrides ``data`` with ``list[list[Any]]`` and holds its
    rows there with no blob and an empty uri, so the flattening discarded the
    whole table and reported success. ``Model3DRef`` (format, material_file,
    texture_files) and ``VideoRef`` (duration, format) lost metadata the same
    way.
    """
    dumped = _strip_binary_payloads(value.model_dump())
    # Keep the wire's type name authoritative: the field's own literal and
    # TypeToName agree for every shipped ref, but an unregistered subclass would
    # otherwise announce a name the TS side does not know.
    dumped["type"] = TypeToName.get(type(value), type(value).__name__)
    return dumped


def _serialize_value(value: Any) -> Any:
    """Convert a value to JSON/msgpack-safe form."""
    if isinstance(value, ASSET_REF_TYPES):
        return _serialize_asset_ref(value)
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
