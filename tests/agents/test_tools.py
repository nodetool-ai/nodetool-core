from types import SimpleNamespace

import pytest

from nodetool.agents.tools.asset_tools import ReadAssetTool, SaveAssetTool
from nodetool.agents.tools.http_tools import DownloadFileTool
from nodetool.workflows.base_node import sanitize_node_name
from nodetool.workflows.processing_context import ProcessingContext


def test_sanitize_node_name_basic():
    assert sanitize_node_name("foo.bar") == "foo_bar"


def test_sanitize_node_name_truncates():
    long_name = "a" * 70
    result = sanitize_node_name(long_name)
    assert len(result) == 64


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
