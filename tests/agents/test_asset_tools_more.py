import pytest
from types import SimpleNamespace

from nodetool.agents.tools.asset_tools import (
    SaveAssetTool,
    ReadAssetTool,
    ListAssetsDirectoryTool,
)


@pytest.mark.asyncio
async def test_save_asset_tool_missing_filename():
    tool = SaveAssetTool()
    # context is not used when filename is missing
    context = SimpleNamespace()
    result = await tool.process(context, {"text": "hi"})
    assert result == {"success": False, "error": "Filename is required"}


def test_save_asset_tool_user_message_truncate():
    tool = SaveAssetTool()
    msg = tool.user_message({"filename": "a" * 100})
    assert msg == "Saving text asset..."


def test_read_asset_tool_user_message():
    tool = ReadAssetTool()
    msg = tool.user_message({"filename": "file.txt"})
    assert msg == "Reading asset file.txt..."


@pytest.mark.asyncio
async def test_read_asset_tool_asset_not_found():
    async def find_asset_by_filename(name):
        return None

    context = SimpleNamespace(find_asset_by_filename=find_asset_by_filename)
    tool = ReadAssetTool()
    result = await tool.process(context, {"filename": "missing.txt"})
    assert result == {
        "success": False,
        "error": "Asset with filename missing.txt not found",
    }


@pytest.mark.asyncio
async def test_list_assets_directory_tool_success():
    asset = SimpleNamespace(id="1", file_name="foo.txt", content_type="text/plain")

    async def list_assets(parent_id=None, recursive=False, mime_type=None):
        return [asset], 1

    context = SimpleNamespace(list_assets=list_assets)
    tool = ListAssetsDirectoryTool()
    result = await tool.process(
        context,
        {"parent_id": None, "recursive": True, "filter_mime_type": "text/plain"},
    )
    assert result["success"] is True
    assert result["count"] == 1
    assert result["assets"][0]["filename"] == "foo.txt"


@pytest.mark.asyncio
async def test_list_assets_directory_tool_error():
    async def list_assets(*args, **kwargs):
        raise RuntimeError("boom")

    context = SimpleNamespace(list_assets=list_assets)
    tool = ListAssetsDirectoryTool()
    result = await tool.process(context, {})
    assert result["success"] is False
    assert "boom" in result["error"]


def test_list_assets_directory_tool_user_message():
    tool = ListAssetsDirectoryTool()
    assert tool.user_message({"any": "param"}) == "Listing assets..."
