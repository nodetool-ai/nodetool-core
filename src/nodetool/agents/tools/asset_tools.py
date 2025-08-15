"""
Asset management tools module.

This module provides tools for managing assets:
- SaveTextAssetTool: Save text to an asset file
- ReadTextAssetTool: Read text from an asset file
- ListAssetsDirectoryTool: List assets in a directory
"""

from io import BytesIO
from typing import Any

from nodetool.workflows.processing_context import ProcessingContext
from .base import Tool


class SaveAssetTool(Tool):
    name = "save_text_asset"
    description = "Save text content as an asset file in the assets directory"
    input_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text content to save",
            },
            "filename": {
                "type": "string",
                "description": "Name of the output file.",
            },
        },
        "required": ["text", "filename"],
    }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            text = params["text"]
            filename = params.get("filename")

            if not filename:
                return {
                    "success": False,
                    "error": "Filename is required",
                }

            # Create BytesIO object with the text content
            file_data = BytesIO(text.encode("utf-8"))

            # Use the context to create the asset
            asset = await context.create_asset(filename, "text/plain", file_data)

            return {
                "success": True,
                "asset_id": asset.id,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def user_message(self, params: dict) -> str:
        filename = params.get("filename", "an asset")
        msg = f"Saving text asset as {filename}..."
        if len(msg) > 80:
            msg = "Saving text asset..."
        return msg


class ReadAssetTool(Tool):
    name = "read_asset"
    description = "Read content from an asset file"
    input_schema = {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Filename of the asset to read",
            },
        },
        "required": ["filename"],
    }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            filename = params["filename"]
            asset = await context.find_asset_by_filename(filename)

            if not asset:
                return {
                    "success": False,
                    "error": f"Asset with filename {filename} not found",
                }

            # Read the asset content
            content = await context.download_asset(asset.id)

            return {
                "success": True,
                "content": content.read(),
                "filename": asset.file_name,
                "mime_type": asset.content_type,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def user_message(self, params: dict) -> str:
        filename = params.get("filename", "an asset")
        msg = f"Reading asset {filename}..."
        if len(msg) > 80:
            msg = "Reading an asset..."
        return msg


class ListAssetsDirectoryTool(Tool):
    name = "list_assets_directory"
    description = "List assets in a directory"
    input_schema = {
        "type": "object",
        "properties": {
            "parent_id": {
                "type": "string",
                "description": "ID of the parent directory (optional, use None for root)",
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether to list assets recursively",
                "default": False,
            },
        },
        "required": [],
    }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            parent_id = params.get("parent_id")
            recursive = params.get("recursive", False)

            # Attempt to list assets using context's methods
            assets, _ = await context.list_assets(
                parent_id=parent_id,
                recursive=recursive,
            )

            # Format the asset information
            asset_list = []
            for asset in assets:
                asset_info = {
                    "id": asset.id,
                    "filename": asset.file_name,
                    "mime_type": asset.content_type,
                }

                asset_list.append(asset_info)

            return {
                "success": True,
                "assets": asset_list,
                "count": len(asset_list),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def user_message(self, params: dict) -> str:
        return "Listing assets..."
