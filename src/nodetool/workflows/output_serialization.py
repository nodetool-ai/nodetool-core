"""
Output serialization utilities for event logging.

This module provides efficient serialization of node outputs for the event log,
with special handling for large objects to prevent bloating the database.

Strategy:
- AssetRef types (ImageRef, VideoRef, etc.) → store only reference metadata
- Small objects (<1MB) → serialize inline as JSON
- Large objects (>1MB) → store separately in Asset storage, log reference ID
"""

import json
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


def serialize_output_for_event_log(value: Any, max_size: int = MAX_INLINE_SIZE) -> dict:
    """Serialize output for event log with efficient handling of large objects.
    
    Args:
        value: The output value to serialize
        max_size: Maximum size in bytes for inline serialization
        
    Returns:
        Dict with one of:
        - {'type': 'asset_ref', 'asset_type': '...', 'uri': '...', 'asset_id': '...'}
        - {'type': 'inline', 'value': {...}}
        - {'type': 'external_ref', 'storage_id': '...', 'size_bytes': N}
        - {'type': 'truncated', 'reason': '...', 'preview': '...'}
    
    Examples:
        >>> # AssetRef types store only reference
        >>> img = ImageRef(uri="file:///path/to/image.png", asset_id="abc123")
        >>> serialize_output_for_event_log(img)
        {'type': 'asset_ref', 'asset_type': 'ImageRef', 'uri': 'file:///...', 'asset_id': 'abc123'}
        
        >>> # Small objects serialize inline
        >>> data = {"status": "ok", "count": 42}
        >>> serialize_output_for_event_log(data)
        {'type': 'inline', 'value': {'status': 'ok', 'count': 42}}
        
        >>> # Large objects get external reference
        >>> big_data = {"data": "x" * 2_000_000}
        >>> serialize_output_for_event_log(big_data)
        {'type': 'external_ref', 'storage_id': 'output_...', 'size_bytes': 2000013}
    """
    
    # Handle None/null
    if value is None:
        return {'type': 'inline', 'value': None}
    
    # Handle AssetRef types - these already store references, not data
    if is_asset_ref(value):
        try:
            return {
                'type': 'asset_ref',
                'asset_type': value.__class__.__name__,
                'uri': getattr(value, 'uri', ''),
                'asset_id': getattr(value, 'asset_id', None),
            }
        except Exception as e:
            log.warning(f"Error serializing AssetRef: {e}")
            return {
                'type': 'truncated',
                'reason': f'AssetRef serialization error: {str(e)[:100]}',
            }
    
    # Try to serialize inline if small enough
    try:
        # First check if it's JSON-serializable
        serialized = json.dumps(value)
        size_bytes = len(serialized.encode('utf-8'))
        
        if size_bytes <= max_size:
            return {'type': 'inline', 'value': value}
        else:
            # Too large for inline storage
            # In future implementation: store in Asset storage
            # For now, just log metadata
            log.debug(f"Output too large for inline storage: {size_bytes} bytes")
            return {
                'type': 'external_ref',
                'storage_id': 'not_implemented',  # TODO: implement external storage
                'size_bytes': size_bytes,
            }
            
    except (TypeError, ValueError) as e:
        # Not JSON-serializable
        # Try to get a string representation
        try:
            str_repr = str(value)
            if len(str_repr) <= 500:
                preview = str_repr
            else:
                preview = str_repr[:500] + '...'
            
            return {
                'type': 'truncated',
                'reason': f'Not JSON-serializable: {type(value).__name__}',
                'preview': preview,
            }
        except Exception as e2:
            return {
                'type': 'truncated',
                'reason': f'Serialization failed: {str(e)[:100]}',
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
                'type': 'truncated',
                'reason': f'Unexpected error: {str(e)[:100]}',
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
    output_type = serialized.get('type')
    
    if output_type == 'inline':
        return serialized.get('value')
    
    elif output_type == 'asset_ref':
        # Reconstruct AssetRef object
        try:
            from nodetool.metadata.types import AssetRef, NameToType
            
            # asset_type is the class name (e.g., 'ImageRef')
            # but we need to look it up using the lowercase type attribute
            asset_type_name = serialized.get('asset_type', 'AssetRef')
            
            # Try to find the class by name first
            # The type system uses lowercase type attributes (e.g., 'image' for ImageRef)
            # So we convert ImageRef -> image for lookup
            type_key = asset_type_name.lower().replace('ref', '')  # ImageRef -> image
            asset_cls = NameToType.get(type_key, AssetRef)
            
            return asset_cls(
                uri=serialized.get('uri', ''),
                asset_id=serialized.get('asset_id'),
            )
        except Exception as e:
            log.warning(f"Error deserializing AssetRef: {e}")
            return {
                '_placeholder': True,
                '_type': 'asset_ref',
                '_uri': serialized.get('uri'),
                '_asset_id': serialized.get('asset_id'),
            }
    
    elif output_type == 'external_ref':
        # Return placeholder - caller should fetch from storage
        return {
            '_placeholder': True,
            '_type': 'external_ref',
            '_storage_id': serialized.get('storage_id'),
            '_size_bytes': serialized.get('size_bytes'),
        }
    
    elif output_type == 'truncated':
        # Return truncated representation
        return {
            '_placeholder': True,
            '_type': 'truncated',
            '_reason': serialized.get('reason'),
            '_preview': serialized.get('preview'),
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
