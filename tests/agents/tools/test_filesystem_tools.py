from unittest.mock import MagicMock

import pytest

from nodetool.agents.tools.filesystem_tools import ListDirectoryTool, ReadFileTool, WriteFileTool
from nodetool.workflows.processing_context import ProcessingContext


@pytest.fixture
def mock_context():
    context = MagicMock(spec=ProcessingContext)
    # Simulate a workspace path resolver that rejects paths outside
    def resolve_workspace_path(path):
        import os
        base = "/workspace"
        full = os.path.abspath(os.path.join(base, path))
        if not full.startswith(base):
            raise ValueError(f"Path '{path}' is outside the allowed directory")
        return full
    context.resolve_workspace_path.side_effect = resolve_workspace_path
    return context

@pytest.mark.asyncio
async def test_write_file_path_traversal(mock_context):
    tool = WriteFileTool()
    result = await tool.process(mock_context, {"path": "../../../etc/passwd", "content": "test"})
    assert result["success"] is False
    assert "outside the allowed directory" in result["error"]

@pytest.mark.asyncio
async def test_read_file_path_traversal(mock_context):
    tool = ReadFileTool()
    result = await tool.process(mock_context, {"path": "../../../etc/passwd"})
    assert result["success"] is False
    assert "outside the allowed directory" in result["error"]

@pytest.mark.asyncio
async def test_list_directory_path_traversal(mock_context):
    tool = ListDirectoryTool()
    result = await tool.process(mock_context, {"path": "../../../etc/passwd"})
    assert result["success"] is False
    assert "outside the allowed directory" in result["error"]
