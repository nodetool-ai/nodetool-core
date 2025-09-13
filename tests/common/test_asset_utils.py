"""
Tests for asset utilities.
"""

import unittest
from unittest.mock import MagicMock

from nodetool.io.asset_utils import encode_assets_as_uri
from nodetool.metadata.types import AssetRef


class TestAssetUtils(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.asset_ref = MagicMock(spec=AssetRef)
        self.asset_ref.encode_data_to_uri.return_value = (
            "data:text/plain;base64,SGVsbG8="
        )

    def test_encode_assets_as_uri_with_asset_ref(self):
        """Test encoding a single AssetRef."""
        result = encode_assets_as_uri(self.asset_ref)
        self.assertEqual(result, "data:text/plain;base64,SGVsbG8=")
        self.asset_ref.encode_data_to_uri.assert_called_once()

    def test_encode_assets_as_uri_with_dict(self):
        """Test encoding a dictionary containing AssetRefs."""
        data = {"key1": "string_value", "key2": self.asset_ref, "key3": 42}
        result = encode_assets_as_uri(data)
        expected = {
            "key1": "string_value",
            "key2": "data:text/plain;base64,SGVsbG8=",
            "key3": 42,
        }
        self.assertEqual(result, expected)

    def test_encode_assets_as_uri_with_list(self):
        """Test encoding a list containing AssetRefs."""
        data = ["string", self.asset_ref, 123]
        result = encode_assets_as_uri(data)
        expected = ["string", "data:text/plain;base64,SGVsbG8=", 123]
        self.assertEqual(result, expected)

    def test_encode_assets_as_uri_with_tuple(self):
        """Test encoding a tuple containing AssetRefs."""
        data = ("string", self.asset_ref, 123)
        result = encode_assets_as_uri(data)
        expected = ("string", "data:text/plain;base64,SGVsbG8=", 123)
        self.assertEqual(result, expected)

    def test_encode_assets_as_uri_with_nested_structures(self):
        """Test encoding nested data structures."""
        data = {
            "list": [self.asset_ref, "text"],
            "dict": {"nested": self.asset_ref},
            "tuple": (self.asset_ref, 42),
        }
        result = encode_assets_as_uri(data)
        expected = {
            "list": ["data:text/plain;base64,SGVsbG8=", "text"],
            "dict": {"nested": "data:text/plain;base64,SGVsbG8="},
            "tuple": ("data:text/plain;base64,SGVsbG8=", 42),
        }
        self.assertEqual(result, expected)

    def test_encode_assets_as_uri_with_primitive_types(self):
        """Test that primitive types are returned unchanged."""
        test_cases = ["string", 42, 3.14, True, None]
        for value in test_cases:
            with self.subTest(value=value):
                result = encode_assets_as_uri(value)
                self.assertEqual(result, value)

    def test_encode_assets_as_uri_with_empty_structures(self):
        """Test encoding empty data structures."""
        test_cases = [{}, [], ()]
        for value in test_cases:
            with self.subTest(value=value):
                result = encode_assets_as_uri(value)
                self.assertEqual(result, value)


if __name__ == "__main__":
    unittest.main()
