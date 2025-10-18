#!/usr/bin/env python

import pytest
from fastapi import HTTPException
from nodetool.api.storage import validate_key


class TestStorageKeyValidation:
    """Test the validate_key function for path separator prevention."""

    def test_valid_keys(self):
        """Test that valid keys (without path separators) pass validation."""
        valid_keys = [
            "file.txt",
            "file-with-dashes.txt",
            "file_with_underscores.txt",
            "file.with.dots.txt",
            "123456789.jpg",
            ".hidden_file",
            "file..txt",
        ]

        for key in valid_keys:
            try:
                validate_key(key)
            except HTTPException:
                pytest.fail(f"Valid key '{key}' was rejected")

    def test_path_separators_rejected(self):
        """Test that keys with path separators are rejected."""
        invalid_keys = [
            "folder/file.txt",
            "folder\\file.txt",
            "../file.txt",
            "..\\file.txt",
            "/etc/passwd",
            "C:/Windows/System32/file.txt",
            "folder/subfolder/file.txt",
            "folder\\subfolder\\file.txt",
            "./file.txt",
            ".\\file.txt",
        ]

        for key in invalid_keys:
            with pytest.raises(HTTPException) as exc_info:
                validate_key(key)
            assert exc_info.value.status_code == 400
            assert "path separators not allowed" in exc_info.value.detail

    def test_edge_cases(self):
        """Test edge cases for validation."""
        # Keys with dots that are valid
        valid_edge_cases = [
            "file..txt",  # double dots in filename
            ".hidden_file",  # hidden file
            "...",  # just dots
            "file...name",
        ]

        for key in valid_edge_cases:
            try:
                validate_key(key)
            except HTTPException:
                pytest.fail(f"Valid edge case key '{key}' was rejected")

        # Keys that should be rejected due to path separators
        invalid_edge_cases = [
            "./",
            "../",
            "folder/.hidden_file",
            "folder.with.dots/file.txt",
        ]

        for key in invalid_edge_cases:
            with pytest.raises(HTTPException) as exc_info:
                validate_key(key)
            assert exc_info.value.status_code == 400
