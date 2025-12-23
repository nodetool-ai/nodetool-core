"""
Tests for output serialization utilities.
"""

import pytest

from nodetool.workflows.output_serialization import (
    serialize_output_for_event_log,
    serialize_outputs_dict,
    deserialize_output_from_event_log,
    deserialize_outputs_dict,
)


def test_serialize_none():
    """Test serialization of None value."""
    result = serialize_output_for_event_log(None)
    assert result['type'] == 'inline'
    assert result['value'] is None


def test_serialize_small_dict():
    """Test serialization of small dictionary (inline)."""
    data = {"status": "ok", "count": 42, "message": "success"}
    result = serialize_output_for_event_log(data)
    
    assert result['type'] == 'inline'
    assert result['value'] == data


def test_serialize_small_list():
    """Test serialization of small list (inline)."""
    data = [1, 2, 3, 4, 5]
    result = serialize_output_for_event_log(data)
    
    assert result['type'] == 'inline'
    assert result['value'] == data


def test_serialize_large_object():
    """Test serialization of large object (external ref)."""
    # Create object larger than 1MB
    data = {"data": "x" * 2_000_000}
    result = serialize_output_for_event_log(data, max_size=1_000_000)
    
    assert result['type'] == 'external_ref'
    assert 'size_bytes' in result


def test_serialize_non_json_serializable():
    """Test serialization of non-JSON-serializable object."""
    class CustomClass:
        def __init__(self):
            self.value = 42
    
    obj = CustomClass()
    result = serialize_output_for_event_log(obj)
    
    assert result['type'] == 'truncated'
    assert 'reason' in result


def test_serialize_asset_ref():
    """Test serialization of AssetRef types."""
    from nodetool.metadata.types import ImageRef
    
    img = ImageRef(uri="file:///path/to/image.png", asset_id="abc123")
    result = serialize_output_for_event_log(img)
    
    assert result['type'] == 'asset_ref'
    assert result['asset_type'] == 'ImageRef'
    assert result['uri'] == "file:///path/to/image.png"
    assert result['asset_id'] == "abc123"


def test_serialize_outputs_dict_mixed():
    """Test serialization of dictionary with mixed output types."""
    from nodetool.metadata.types import ImageRef, VideoRef
    
    outputs = {
        "image": ImageRef(uri="file:///image.png", asset_id="img1"),
        "video": VideoRef(uri="file:///video.mp4", asset_id="vid1"),
        "result": {"status": "ok", "count": 42},
        "large": {"data": "x" * 2_000_000},
    }
    
    result = serialize_outputs_dict(outputs)
    
    assert result['image']['type'] == 'asset_ref'
    assert result['video']['type'] == 'asset_ref'
    assert result['result']['type'] == 'inline'
    assert result['large']['type'] == 'external_ref'


def test_deserialize_inline():
    """Test deserialization of inline value."""
    serialized = {'type': 'inline', 'value': {"status": "ok", "count": 42}}
    result = deserialize_output_from_event_log(serialized)
    
    assert result == {"status": "ok", "count": 42}


def test_deserialize_asset_ref():
    """Test deserialization of AssetRef."""
    from nodetool.metadata.types import ImageRef
    
    serialized = {
        'type': 'asset_ref',
        'asset_type': 'ImageRef',
        'uri': 'file:///image.png',
        'asset_id': 'abc123',
    }
    result = deserialize_output_from_event_log(serialized)
    
    assert isinstance(result, ImageRef)
    assert result.uri == 'file:///image.png'
    assert result.asset_id == 'abc123'


def test_deserialize_external_ref():
    """Test deserialization of external ref (returns placeholder)."""
    serialized = {
        'type': 'external_ref',
        'storage_id': 'xyz789',
        'size_bytes': 2000013,
    }
    result = deserialize_output_from_event_log(serialized)
    
    assert isinstance(result, dict)
    assert result['_placeholder'] is True
    assert result['_storage_id'] == 'xyz789'


def test_deserialize_truncated():
    """Test deserialization of truncated value."""
    serialized = {
        'type': 'truncated',
        'reason': 'Not JSON-serializable',
        'preview': 'CustomClass(...)',
    }
    result = deserialize_output_from_event_log(serialized)
    
    assert isinstance(result, dict)
    assert result['_placeholder'] is True
    assert result['_type'] == 'truncated'


def test_roundtrip_small_data():
    """Test serialization and deserialization roundtrip for small data."""
    original = {"status": "ok", "count": 42, "data": [1, 2, 3]}
    serialized = serialize_output_for_event_log(original)
    deserialized = deserialize_output_from_event_log(serialized)
    
    assert deserialized == original


def test_roundtrip_asset_ref():
    """Test serialization and deserialization roundtrip for AssetRef."""
    from nodetool.metadata.types import ImageRef
    
    original = ImageRef(uri="file:///image.png", asset_id="abc123")
    serialized = serialize_output_for_event_log(original)
    deserialized = deserialize_output_from_event_log(serialized)
    
    assert isinstance(deserialized, ImageRef)
    assert deserialized.uri == original.uri
    assert deserialized.asset_id == original.asset_id


def test_serialize_outputs_dict_empty():
    """Test serialization of empty outputs dict."""
    result = serialize_outputs_dict({})
    assert result == {}


def test_deserialize_outputs_dict():
    """Test deserialization of outputs dict."""
    serialized = {
        'result': {'type': 'inline', 'value': {"status": "ok"}},
        'image': {
            'type': 'asset_ref',
            'asset_type': 'ImageRef',
            'uri': 'file:///image.png',
            'asset_id': 'abc123',
        },
    }
    
    result = deserialize_outputs_dict(serialized)
    
    assert result['result'] == {"status": "ok"}
    assert hasattr(result['image'], 'uri')
