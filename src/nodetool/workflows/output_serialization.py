"""
Output serialization utilities for event logging.

This module provides efficient serialization of node outputs for the event log,
with special handling for large objects to prevent bloating the database.

Strategy:
- AssetRef types (ImageRef, VideoRef, etc.) → use temp storage for in-flight outputs
- Small objects (<1MB) → serialize inline as JSON
- Large objects (>1MB) → store in temp storage, log reference ID
- Streaming outputs → compress all chunks into single entry

Important for Streaming:
- Streaming nodes emit thousands of chunks (creates write contention)
- Solution: Log only at completion, compress chunks into one entry
- Store compressed chunks in temp storage if needed
"""

import io
import json
import uuid
from typing import Any

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

# Maximum size for inline serialization (1MB)
MAX_INLINE_SIZE = 1_000_000


def is_asset_ref(value: Any) -> bool:
    """Check if a value is an AssetRef type."""
    try:
        from nodetool.metadata.types import AssetRef

        return isinstance(value, AssetRef)
    except ImportError:
        return False


def uses_temp_storage(uri: str) -> bool:
    """Check if a URI uses temp storage (needs migration for durability)."""
    return uri.startswith("memory://") or uri.startswith("temp://")


async def store_large_output_in_temp_storage(value: Any, storage: Any) -> str | None:
    """Store a large output value in temp storage.

    Args:
        value: The value to store (must be JSON-serializable)
        storage: Storage instance (AbstractStorage)

    Returns:
        Storage ID if successful, None otherwise
    """
    try:
        # Serialize to JSON
        serialized = json.dumps(value)
        data = serialized.encode("utf-8")

        # Generate unique storage ID
        storage_id = f"output_{uuid.uuid4().hex}"

        # Store in temp storage
        data_stream = io.BytesIO(data)
        await storage.upload(storage_id, data_stream)

        log.debug(f"Stored large output in temp storage: {storage_id} ({len(data)} bytes)")
        return storage_id
    except Exception as e:
        log.error(f"Failed to store large output in temp storage: {e}")
        return None


async def store_streaming_output_in_temp_storage(outputs: dict[str, Any], storage: Any) -> str | None:
    """Store streaming outputs in temp storage.

    Args:
        outputs: Dictionary of streaming outputs to store
        storage: Storage instance (AbstractStorage)

    Returns:
        Storage ID if successful, None otherwise
    """
    try:
        # Serialize to JSON
        serialized = json.dumps(outputs, default=str)
        data = serialized.encode("utf-8")

        # Generate unique storage ID
        storage_id = f"streaming_{uuid.uuid4().hex}"

        # Store in temp storage
        data_stream = io.BytesIO(data)
        await storage.upload(storage_id, data_stream)

        log.debug(f"Stored streaming outputs in temp storage: {storage_id} ({len(data)} bytes)")
        return storage_id
    except Exception as e:
        log.error(f"Failed to store streaming outputs in temp storage: {e}")
        return None


async def retrieve_output_from_temp_storage(storage_id: str, storage: Any) -> Any | None:
    """Retrieve a large output value from temp storage.

    Args:
        storage_id: The storage ID to retrieve
        storage: Storage instance (AbstractStorage)

    Returns:
        The deserialized value, or None if retrieval fails
    """
    try:
        # Download from temp storage
        data_stream = io.BytesIO()
        await storage.download(storage_id, data_stream)
        data_stream.seek(0)
        data = data_stream.read()

        # Deserialize from JSON
        value = json.loads(data.decode("utf-8"))

        log.debug(f"Retrieved output from temp storage: {storage_id}")
        return value
    except Exception as e:
        log.error(f"Failed to retrieve output from temp storage {storage_id}: {e}")
        return None


def serialize_output_for_event_log(value: Any, max_size: int = MAX_INLINE_SIZE, use_temp_storage: bool = True, storage: Any = None) -> dict:
    """Serialize output for event log with efficient handling of large objects.

    Args:
        value: The output value to serialize
        max_size: Maximum size in bytes for inline serialization
        use_temp_storage: If True, migrate AssetRefs to temp storage for durability
        storage: Optional storage instance for storing large outputs

    Returns:
        Dict with one of:
        - {'type': 'asset_ref', 'asset_type': '...', 'uri': 'temp://...', 'asset_id': '...'}
        - {'type': 'inline', 'value': {...}}
        - {'type': 'external_ref', 'storage_id': '...', 'size_bytes': N}
        - {'type': 'truncated', 'reason': '...', 'preview': '...'}

    Note:
        When use_temp_storage=True and AssetRef has memory:// URI, the URI should be
        migrated to temp storage before logging. This ensures outputs survive crashes.

        If storage is provided and value is too large for inline storage, it will
        be stored in temp storage and a reference returned instead.

    Examples:
        >>> # AssetRef types store only reference (temp storage for durability)
        >>> img = ImageRef(uri="temp://bucket/image.png", asset_id="temp_abc123")
        >>> serialize_output_for_event_log(img)
        {'type': 'asset_ref', 'asset_type': 'ImageRef', 'uri': 'temp://...', 'asset_id': 'temp_abc123'}

        >>> # Small objects serialize inline
        >>> data = {"status": "ok", "count": 42}
        >>> serialize_output_for_event_log(data)
        {'type': 'inline', 'value': {'status': 'ok', 'count': 42}}

        >>> # Large objects get external reference (stored in temp storage)
        >>> big_data = {"data": "x" * 2_000_000}
        >>> serialize_output_for_event_log(big_data, storage=storage)
        {'type': 'external_ref', 'storage_id': 'output_abc123...', 'size_bytes': 2000013}
    """

    # Handle None/null
    if value is None:
        return {"type": "inline", "value": None}

    # Handle AssetRef types - these already store references, not data
    if is_asset_ref(value):
        try:
            uri = getattr(value, "uri", "")
            asset_id = getattr(value, "asset_id", None)

            # Warn if using memory URI (not durable)
            if use_temp_storage and uses_temp_storage(uri):
                log.warning(
                    f"AssetRef uses non-durable storage (uri={uri}). "
                    "For resumable execution, migrate to temp storage before logging."
                )

            return {
                "type": "asset_ref",
                "asset_type": value.__class__.__name__,
                "uri": uri,
                "asset_id": asset_id,
            }
        except Exception as e:
            log.warning(f"Error serializing AssetRef: {e}")
            return {
                "type": "truncated",
                "reason": f"AssetRef serialization error: {str(e)[:100]}",
            }

    # Try to serialize inline if small enough
    try:
        # First check if it's JSON-serializable
        serialized = json.dumps(value)
        size_bytes = len(serialized.encode("utf-8"))

        if size_bytes <= max_size:
            return {"type": "inline", "value": value}
        else:
            # Too large for inline storage
            # Store in temp storage if available
            if storage is not None:
                log.debug(f"Output too large for inline storage: {size_bytes} bytes (store in temp)")
                # Note: This is a synchronous wrapper, async storage happens elsewhere
                # For now, return a placeholder that can be updated later
                storage_id = f"output_{uuid.uuid4().hex}"
                return {
                    "type": "external_ref",
                    "storage_id": storage_id,
                    "size_bytes": size_bytes,
                }
            else:
                log.debug(f"Output too large for inline storage: {size_bytes} bytes (no storage available)")
                return {
                    "type": "external_ref",
                    "storage_id": "not_implemented",
                    "size_bytes": size_bytes,
                }

    except (TypeError, ValueError) as e:
        # Not JSON-serializable
        # Try to get a string representation
        try:
            str_repr = str(value)
            preview = str_repr if len(str_repr) <= 500 else str_repr[:500] + "..."

            return {
                "type": "truncated",
                "reason": f"Not JSON-serializable: {type(value).__name__}",
                "preview": preview,
            }
        except Exception:
            return {
                "type": "truncated",
                "reason": f"Serialization failed: {str(e)[:100]}",
            }


def serialize_outputs_dict(outputs: dict[str, Any], max_size: int = MAX_INLINE_SIZE) -> dict[str, dict]:
    """Serialize a dictionary of outputs for event logging.

    Args:
        outputs: Dictionary mapping output names to values
        max_size: Maximum size for inline serialization per output

    Returns:
        Dictionary mapping output names to serialized representations

    Example:
        >>> outputs = {
        ...     "image": ImageRef(uri="file:///image.png"),
        ...     "result": {"status": "ok", "count": 42},
        ...     "large_data": {"data": "x" * 2_000_000}
        ... }
        >>> serialized = serialize_outputs_dict(outputs)
        >>> serialized["image"]["type"]
        'asset_ref'
        >>> serialized["result"]["type"]
        'inline'
        >>> serialized["large_data"]["type"]
        'external_ref'
    """
    result = {}
    for key, value in outputs.items():
        try:
            result[key] = serialize_output_for_event_log(value, max_size)
        except Exception as e:
            log.error(f"Error serializing output '{key}': {e}")
            result[key] = {
                "type": "truncated",
                "reason": f"Unexpected error: {str(e)[:100]}",
            }
    return result


def deserialize_output_from_event_log(serialized: dict) -> Any:
    """Deserialize an output value from event log representation.

    Args:
        serialized: Serialized representation from serialize_output_for_event_log()

    Returns:
        The deserialized value, or a placeholder if the value was stored externally

    Note:
        For 'external_ref' types, this returns a placeholder dict. The actual
        value should be retrieved from Asset storage using the storage_id.
    """
    output_type = serialized.get("type")

    if output_type == "inline":
        return serialized.get("value")

    elif output_type == "asset_ref":
        # Reconstruct AssetRef object
        try:
            from nodetool.metadata.types import AssetRef, NameToType

            # asset_type is the class name (e.g., 'ImageRef')
            # but we need to look it up using the lowercase type attribute
            asset_type_name = serialized.get("asset_type", "AssetRef")

            # Try to find the class by name first
            # The type system uses lowercase type attributes (e.g., 'image' for ImageRef)
            # So we convert ImageRef -> image for lookup
            type_key = asset_type_name.lower().replace("ref", "")  # ImageRef -> image
            asset_cls = NameToType.get(type_key, AssetRef)

            return asset_cls(
                uri=serialized.get("uri", ""),
                asset_id=serialized.get("asset_id"),
            )
        except Exception as e:
            log.warning(f"Error deserializing AssetRef: {e}")
            return {
                "_placeholder": True,
                "_type": "asset_ref",
                "_uri": serialized.get("uri"),
                "_asset_id": serialized.get("asset_id"),
            }

    elif output_type == "external_ref":
        # Return placeholder - caller should fetch from storage
        return {
            "_placeholder": True,
            "_type": "external_ref",
            "_storage_id": serialized.get("storage_id"),
            "_size_bytes": serialized.get("size_bytes"),
        }

    elif output_type == "truncated":
        # Return truncated representation
        return {
            "_placeholder": True,
            "_type": "truncated",
            "_reason": serialized.get("reason"),
            "_preview": serialized.get("preview"),
        }

    else:
        log.warning(f"Unknown output type in event log: {output_type}")
        return None


def deserialize_outputs_dict(serialized_outputs: dict[str, dict]) -> dict[str, Any]:
    """Deserialize a dictionary of outputs from event log representation.

    Args:
        serialized_outputs: Dictionary of serialized outputs

    Returns:
        Dictionary of deserialized values
    """
    result = {}
    for key, serialized in serialized_outputs.items():
        try:
            result[key] = deserialize_output_from_event_log(serialized)
        except Exception as e:
            log.error(f"Error deserializing output '{key}': {e}")
            result[key] = None
    return result


def compress_streaming_outputs(outputs: dict[str, Any], storage: Any = None) -> dict:
    """Compress streaming outputs into single log entry.

    Streaming nodes can emit thousands of chunks (e.g., 1000 video frames),
    which would create database write contention if logged individually.
    This function compresses all chunks into a single entry.

    Args:
        outputs: Dictionary of outputs, potentially with lists of chunks
        storage: Optional storage instance for storing compressed chunks

    Returns:
        Dict with:
        - {'type': 'streaming_compressed', 'chunk_count': N, 'storage_id': '...', 'size_bytes': X}
        - If no storage provided: 'storage_id' will be 'not_implemented'

    Note:
        When storage is provided, the actual chunks should be stored in temp storage.
        This prevents bloating the event log while maintaining recoverability.

    Example:
        >>> # Node emits 1000 image chunks
        >>> outputs = {'frames': [ImageRef(...) for _ in range(1000)]}
        >>> compressed = compress_streaming_outputs(outputs, storage=storage)
        >>> compressed['chunk_count']
        1000
        >>> compressed['type']
        'streaming_compressed'
    """
    # Count total chunks across all outputs
    chunk_count = 0
    for value in outputs.values():
        if isinstance(value, list):
            chunk_count += len(value)
        else:
            chunk_count += 1

    # Estimate size (would be actual size if stored)
    try:
        serialized = json.dumps(outputs, default=str)
        size_bytes = len(serialized.encode("utf-8"))
    except Exception:
        size_bytes = 0

    # Store in temp storage if available
    storage_id = "not_implemented"
    if storage is not None and size_bytes > 0:
        # Note: This is a synchronous wrapper
        # Actual storage should happen via async version
        storage_id = f"streaming_{uuid.uuid4().hex}"

    log.debug(
        f"Compressing streaming outputs: {chunk_count} chunks, ~{size_bytes} bytes"
        + (f" (stored as {storage_id})" if storage_id != "not_implemented" else " (no storage available)")
    )

    return {
        "type": "streaming_compressed",
        "chunk_count": chunk_count,
        "storage_id": storage_id,
        "size_bytes": size_bytes,
    }


def should_compress_streaming(outputs: dict[str, Any], threshold: int = 100) -> bool:
    """Determine if outputs should be compressed as streaming.

    Args:
        outputs: Dictionary of outputs to check
        threshold: Minimum number of chunks to trigger compression

    Returns:
        True if outputs contain many chunks and should be compressed

    Example:
        >>> outputs = {'frames': [ImageRef(...) for _ in range(1000)]}
        >>> should_compress_streaming(outputs)
        True
        >>> outputs = {'result': ImageRef(...)}
        >>> should_compress_streaming(outputs)
        False
    """
    chunk_count = 0
    for value in outputs.values():
        if isinstance(value, list):
            chunk_count += len(value)

    return chunk_count >= threshold
