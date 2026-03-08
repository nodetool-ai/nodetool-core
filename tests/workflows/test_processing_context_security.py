import os
import shutil
import tempfile
import unittest
from pathlib import Path

from nodetool.io.path_utils import resolve_workspace_path
from nodetool.workflows.processing_context import ProcessingContext


class TestProcessingContextSecurity(unittest.TestCase):
    def setUp(self):
        self.workspace_base = Path(tempfile.mkdtemp())
        self.workspace_dir = self.workspace_base / "workspace"
        self.workspace_dir.mkdir()

        # Create a directory that starts with the same prefix but is outside
        self.suffix_dir = self.workspace_base / "workspace_suffix"
        self.suffix_dir.mkdir()
        self.suffix_file = self.suffix_dir / "pwned.json"

        self.context = ProcessingContext(auth_token="test", workspace_dir=str(self.workspace_dir))
        # Create a directory strictly outside workspace to try to write to
        self.target_dir = Path(tempfile.mkdtemp())
        self.target_file = self.target_dir / "pwned.json"

    def tearDown(self):
        shutil.rmtree(self.workspace_base)
        shutil.rmtree(self.target_dir)

    def test_arbitrary_file_write_prevention(self):
        """Test that store_step_result prevents writing outside workspace via path traversal."""
        # Calculate relative path to target file
        rel_path = os.path.relpath(self.target_file, self.workspace_dir)
        key = rel_path[:-5] if rel_path.endswith(".json") else rel_path

        # Attempt to store result using path traversal key
        # This should fail (either raise exception or just not write to target)
        try:
            self.context.store_step_result(key, {"pwned": True})
        except ValueError:
            # This is expected behavior (secure)
            pass
        except Exception:
            # Other exceptions are fine too, as long as file is not written
            pass

        # Check if file was written outside workspace
        if self.target_file.exists():
            self.fail("File written outside workspace (classic traversal)")

    def test_partial_path_traversal_prevention(self):
        """Test that resolve_workspace_path prevents partial path traversal."""
        # Attempt to access a sibling directory with same prefix
        # workspace_dir: /tmp/.../workspace
        # suffix_dir:    /tmp/.../workspace_suffix

        # We try to resolve a path that would point to suffix_dir
        # ../workspace_suffix/file.txt

        try:
            p = resolve_workspace_path(str(self.workspace_dir), "../workspace_suffix/file.txt")

            # If we are here, check if it resolved outside workspace
            if not str(p).startswith(str(self.workspace_dir) + os.sep):
                self.fail("Partial path traversal allowed (vulnerability confirmed)")

        except ValueError:
            # Expected behavior (secure)
            pass
