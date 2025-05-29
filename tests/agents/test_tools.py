import os
from types import SimpleNamespace
import pytest

from nodetool.agents.tools.base import (
    Tool,
    sanitize_node_name,
    get_tool_by_name,
    _tool_registry,
)
from nodetool.agents.tools.workspace_tools import WriteFileTool, ReadFileTool
from nodetool.agents.tools.http_tools import DownloadFileTool
from nodetool.agents.tools.asset_tools import SaveAssetTool, ReadAssetTool
from nodetool.agents.tools.mcp_tools import MCPTool
from nodetool.workflows.processing_context import ProcessingContext


def test_sanitize_node_name_basic():
    assert sanitize_node_name("foo.bar") == "foo__bar"


def test_sanitize_node_name_invalid_type():
    assert sanitize_node_name(123) == ""


def test_sanitize_node_name_truncates():
    long_name = "a" * 70
    assert len(sanitize_node_name(long_name)) == 64


def test_tool_registration_cleanup():
    class DummyTool(Tool):
        name = "dummy_tool_for_test"

    try:
        assert get_tool_by_name("dummy_tool_for_test") is DummyTool
    finally:
        _tool_registry.pop("dummy_tool_for_test", None)


@pytest.mark.asyncio
async def test_write_and_read_file(context: ProcessingContext, monkeypatch):
    write_tool = WriteFileTool()
    await write_tool.process(context, {"path": "sample.txt", "content": "hello"})

    full_path = context.resolve_workspace_path("sample.txt")
    assert os.path.exists(full_path)
    with open(full_path, "r", encoding="utf-8") as f:
        assert f.read() == "hello"

    read_tool = ReadFileTool()
    monkeypatch.setattr(ReadFileTool, "count_tokens", lambda self, text: len(text))
    result = await read_tool.process(context, {"path": "sample.txt"})
    assert result["success"] is True
    assert result["content"] == "hello"


class DummyResponse:
    def __init__(self, content=b"data"):
        self.status = 200
        self.headers = {
            "Content-Type": "text/plain",
            "Content-Length": str(len(content)),
        }
        self._content = content

    async def read(self):
        return self._content

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class DummySession:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def get(self, *args, **kwargs):
        return self._response


@pytest.mark.asyncio
async def test_download_file(monkeypatch, context: ProcessingContext):
    response = DummyResponse(b"content")
    session = DummySession(response)
    monkeypatch.setattr("aiohttp.ClientSession", lambda: session)

    tool = DownloadFileTool()
    result = await tool.process(
        context, {"url": "http://example.com/file", "output_file": "file.txt"}
    )
    assert result["success"] is True
    full_path = context.resolve_workspace_path("file.txt")
    with open(full_path, "rb") as f:
        assert f.read() == b"content"


@pytest.mark.asyncio
async def test_save_and_read_asset(monkeypatch):
    saved = {}

    async def create_asset(name, content_type, content):
        saved[name] = content.read()
        return SimpleNamespace(id="asset123")

    async def find_asset_by_filename(name):
        if name in saved:
            return SimpleNamespace(
                id="asset123", file_name=name, content_type="text/plain"
            )
        return None

    async def download_asset(asset_id):
        from io import BytesIO

        return BytesIO(saved["test.txt"])

    context = SimpleNamespace(
        create_asset=create_asset,
        find_asset_by_filename=find_asset_by_filename,
        download_asset=download_asset,
    )

    save_tool = SaveAssetTool()
    result = await save_tool.process(context, {"text": "hi", "filename": "test.txt"})
    assert result["success"] is True
    assert result["asset_id"] == "asset123"

    read_tool = ReadAssetTool()
    result = await read_tool.process(context, {"filename": "test.txt"})
    assert result["success"] is True
    assert result["content"] == b"hi"
    assert result["filename"] == "test.txt"


class DummyHTTPResponse:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


@pytest.mark.asyncio
async def test_mcp_tool(monkeypatch, context: ProcessingContext):
    async def fake_post(url, **kwargs):
        assert url == "http://localhost:8000/tools/test"
        assert kwargs["json"] == {"input": "hello"}
        return DummyHTTPResponse({"result": "ok"})

    monkeypatch.setattr(context, "http_post", fake_post)
    tool = MCPTool(tool="test")
    result = await tool.process(context, {"input": "hello"})
    assert result == {"result": "ok"}
    env = tool.get_container_env()
    assert env["MCP_API_URL"] == "http://localhost:8000"
