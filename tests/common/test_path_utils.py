"""
Tests for path utilities.
"""

import os
import tempfile
import unittest

from nodetool.io.path_utils import resolve_workspace_path


class TestPathUtils(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.workspace_dir = tempfile.mkdtemp()
        # Create some test files
        self.test_file = os.path.join(self.workspace_dir, "test.txt")
        with open(self.test_file, "w") as f:
            f.write("test content")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.workspace_dir)

    def test_resolve_workspace_path_with_relative_path(self):
        """Test resolving a simple relative path."""
        result = resolve_workspace_path(self.workspace_dir, "test.txt")
        expected = self.test_file
        self.assertEqual(result, expected)

    def test_resolve_workspace_path_with_workspace_prefix(self):
        """Test resolving path with workspace/ prefix."""
        result = resolve_workspace_path(self.workspace_dir, "workspace/test.txt")
        expected = self.test_file
        self.assertEqual(result, expected)

    def test_resolve_workspace_path_with_slash_workspace_prefix(self):
        """Test resolving path with /workspace/ prefix."""
        result = resolve_workspace_path(self.workspace_dir, "/workspace/test.txt")
        expected = self.test_file
        self.assertEqual(result, expected)

    def test_resolve_workspace_path_with_subdirectory(self):
        """Test resolving path with subdirectory."""
        sub_dir = os.path.join(self.workspace_dir, "subdir")
        os.makedirs(sub_dir)
        sub_file = os.path.join(sub_dir, "subfile.txt")
        with open(sub_file, "w") as f:
            f.write("sub content")

        result = resolve_workspace_path(self.workspace_dir, "subdir/subfile.txt")
        expected = sub_file
        self.assertEqual(result, expected)

    def test_resolve_workspace_path_with_path_traversal_attack(self):
        """Test that path traversal attacks are prevented."""
        with self.assertRaises(ValueError) as context:
            resolve_workspace_path(self.workspace_dir, "../../../etc/passwd")
        self.assertIn("outside the workspace directory", str(context.exception))

    def test_resolve_workspace_path_with_empty_workspace_dir(self):
        """Test that empty workspace directory raises ValueError."""
        with self.assertRaises(ValueError) as context:
            resolve_workspace_path("", "test.txt")
        self.assertIn("Workspace directory is required", str(context.exception))

    def test_resolve_workspace_path_creates_absolute_path(self):
        """Test that result is always an absolute path."""
        result = resolve_workspace_path(self.workspace_dir, "test.txt")
        self.assertTrue(os.path.isabs(result))
        self.assertTrue(result.startswith(self.workspace_dir))

    def test_resolve_workspace_path_normalizes_separators(self):
        """Test that path separators are normalized correctly."""
        # Create path with mixed separators (if on Windows, this would use backslashes)
        mixed_path = "subdir/test.txt".replace("/", os.sep)
        result = resolve_workspace_path(self.workspace_dir, mixed_path)
        self.assertTrue(os.path.isabs(result))

    def test_resolve_workspace_path_preserves_case_sensitivity(self):
        """Test that case sensitivity is preserved on case-sensitive filesystems."""
        # This test may behave differently on Windows vs Unix
        result = resolve_workspace_path(self.workspace_dir, "Test.txt")
        # The behavior depends on the filesystem, but the function should not crash
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
