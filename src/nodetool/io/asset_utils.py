"""
Asset utilities for encoding and processing assets.
"""

from typing import Any

from nodetool.metadata.types import AssetRef


def encode_assets_as_uri(value: Any) -> Any:
    """
    Recursively encodes any AssetRef objects found in the given value as URIs.

    Args:
        value: Any Python value that might contain AssetRef objects

    Returns:
        Any: The value with all AssetRef objects encoded as URIs
    """
    if isinstance(value, AssetRef):
        return value.encode_data_to_uri()
    elif isinstance(value, dict):
        return {k: encode_assets_as_uri(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [encode_assets_as_uri(item) for item in value]
    elif isinstance(value, tuple):
        items = [encode_assets_as_uri(item) for item in value]
        return tuple(items)
    else:
        return value
