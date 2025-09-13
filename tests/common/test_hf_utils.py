"""
Tests for HuggingFace utilities.
"""

import unittest
from unittest.mock import patch, MagicMock

from nodetool.integrations.huggingface.hf_utils import is_model_cached


class TestHFUtils(unittest.TestCase):
    @patch("nodetool.integrations.huggingface.hf_utils.try_to_load_from_cache")
    def test_is_model_cached_returns_true_when_cache_path_exists(
        self, mock_try_to_load
    ):
        """Test that is_model_cached returns True when model is cached."""
        mock_try_to_load.return_value = "/path/to/cached/model"
        result = is_model_cached("test/repo")
        self.assertTrue(result)
        mock_try_to_load.assert_called_once_with("test/repo", "config.json")

    @patch("nodetool.integrations.huggingface.hf_utils.try_to_load_from_cache")
    def test_is_model_cached_returns_false_when_cache_path_none(self, mock_try_to_load):
        """Test that is_model_cached returns False when model is not cached."""
        mock_try_to_load.return_value = None
        result = is_model_cached("test/repo")
        self.assertFalse(result)
        mock_try_to_load.assert_called_once_with("test/repo", "config.json")

    @patch("nodetool.integrations.huggingface.hf_utils.try_to_load_from_cache")
    def test_is_model_cached_handles_exceptions(self, mock_try_to_load):
        """Test that is_model_cached handles exceptions gracefully."""
        mock_try_to_load.side_effect = Exception("Network error")
        result = is_model_cached("test/repo")
        self.assertFalse(result)
        mock_try_to_load.assert_called_once_with("test/repo", "config.json")


if __name__ == "__main__":
    unittest.main()
