#!/usr/bin/env python

import pytest
from fastapi import HTTPException

from nodetool.api.storage import validate_key


class TestStorageKeyValidation:
    """Test the validate_key function for safe normalization."""

    def test_valid_keys(self):
        """Test that valid keys are normalized and accepted."""
        valid_keys = {
            "file.txt": "file.txt",
            "file-with-dashes.txt": "file-with-dashes.txt",
            "folder/file.txt": "folder/file.txt",
            "folder/subfolder/file.txt": "folder/subfolder/file.txt",
            "folder\\windows\\file.txt": "folder/windows/file.txt",
            "nested/./file.txt": "nested/file.txt",
            "trailing/slash/": "trailing/slash",
        }

        for key, expected in valid_keys.items():
            try:
                assert validate_key(key) == expected
            except HTTPException as exc:
                pytest.fail(f"Valid key '{key}' was rejected: {exc.detail}")

    def test_traversal_or_absolute_paths_rejected(self):
        """Test that traversal and absolute paths are rejected."""
        invalid_keys = {
            "../file.txt": "path traversal",
            "..\\file.txt": "path traversal",
            "/etc/passwd": "absolute paths",
            "\\windows\\system32": "absolute paths",
            "folder/../../secret.txt": "path traversal",
            "": "must not be empty",
            "////": "must not be empty",
            ".": "must not be empty",
        }

        for key, expected_detail in invalid_keys.items():
            with pytest.raises(HTTPException) as exc_info:
                validate_key(key)
            assert exc_info.value.status_code == 400
            assert expected_detail in exc_info.value.detail
