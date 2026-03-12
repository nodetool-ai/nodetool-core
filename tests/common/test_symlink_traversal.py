import os
import tempfile
import unittest

from nodetool.io.path_utils import resolve_workspace_path

class TestSymlinkTraversal(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.workspace_dir = tempfile.mkdtemp()
        self.outside_dir = tempfile.mkdtemp()

        self.outside_file = os.path.join(self.outside_dir, "secret.txt")
        with open(self.outside_file, "w") as f:
            f.write("secret content")

        # Create a symlink inside the workspace pointing to outside directory
        self.symlink_path = os.path.join(self.workspace_dir, "symlink_dir")
        os.symlink(self.outside_dir, self.symlink_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.workspace_dir)
        shutil.rmtree(self.outside_dir)

    def test_resolve_workspace_path_with_symlink_attack(self):
        """Test that symlink traversal attacks are prevented."""
        # Try to access secret.txt via the symlink
        with self.assertRaises(ValueError) as context:
            resolve_workspace_path(self.workspace_dir, "symlink_dir/secret.txt")
        self.assertIn("outside the workspace directory", str(context.exception))

if __name__ == "__main__":
    unittest.main()
