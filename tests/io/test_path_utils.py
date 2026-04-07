import os
import pytest
from nodetool.io.path_utils import resolve_workspace_path

def test_resolve_workspace_path_basic(tmp_path):
    workspace = str(tmp_path)
    # basic resolution
    assert resolve_workspace_path(workspace, "test.txt") == os.path.join(workspace, "test.txt")
    assert resolve_workspace_path(workspace, "/workspace/test.txt") == os.path.join(workspace, "test.txt")
    assert resolve_workspace_path(workspace, "workspace/test.txt") == os.path.join(workspace, "test.txt")
    assert resolve_workspace_path(workspace, "/test.txt") == os.path.join(workspace, "test.txt")

def test_resolve_workspace_path_traversal(tmp_path):
    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace)
    with pytest.raises(ValueError):
        resolve_workspace_path(workspace, "../test.txt")

def test_resolve_workspace_path_symlink_traversal(tmp_path):
    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace)

    # Create a symlink pointing outside the workspace
    outside_dir = str(tmp_path / "outside")
    os.makedirs(outside_dir)

    symlink_path = os.path.join(workspace, "symlink_dir")
    os.symlink(outside_dir, symlink_path)

    # Resolving a path through the symlink should raise ValueError if properly protected
    with pytest.raises(ValueError):
        resolve_workspace_path(workspace, "symlink_dir/test.txt")
