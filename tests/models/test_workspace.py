"""
Tests for the Workspace model.
"""

import asyncio
import os
import tempfile
import unittest

import pytest

from nodetool.models.workspace import Workspace


class TestWorkspaceModel(unittest.TestCase):
    """Test cases for the Workspace model."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_workspace_is_accessible_with_valid_path(self):
        """Test is_accessible returns True for a valid writable directory."""
        workspace = Workspace(
            id="test-id",
            user_id="user-1",
            name="Test Workspace",
            path=self.temp_dir,
        )
        self.assertTrue(workspace.is_accessible())

    def test_workspace_is_accessible_with_invalid_path(self):
        """Test is_accessible returns False for a non-existent directory."""
        workspace = Workspace(
            id="test-id",
            user_id="user-1",
            name="Test Workspace",
            path="/nonexistent/path/that/does/not/exist",
        )
        self.assertFalse(workspace.is_accessible())

    def test_workspace_is_accessible_with_empty_path(self):
        """Test is_accessible returns False for empty path."""
        workspace = Workspace(
            id="test-id",
            user_id="user-1",
            name="Test Workspace",
            path="",
        )
        self.assertFalse(workspace.is_accessible())

    def test_workspace_validate_path_with_valid_absolute_path(self):
        """Test validate_path returns True for a valid absolute path."""
        workspace = Workspace(
            id="test-id",
            user_id="user-1",
            name="Test Workspace",
            path=self.temp_dir,
        )
        self.assertTrue(workspace.validate_path())

    def test_workspace_validate_path_with_relative_path(self):
        """Test validate_path returns False for a relative path."""
        workspace = Workspace(
            id="test-id",
            user_id="user-1",
            name="Test Workspace",
            path="relative/path",
        )
        self.assertFalse(workspace.validate_path())


class TestWorkspaceModelAsync:
    """Async test cases for the Workspace model."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp = tempfile.mkdtemp()
        yield temp
        import shutil

        shutil.rmtree(temp)


if __name__ == "__main__":
    unittest.main()
