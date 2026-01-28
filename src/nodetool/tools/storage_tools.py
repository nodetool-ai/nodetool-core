"""Storage management tools.

These tools provide functionality for managing NodeTool storage.
"""

from __future__ import annotations

from typing import Any

from io import BytesIO
from nodetool.runtime.resources import require_scope


class StorageTools:
    """Storage management tools."""

    @staticmethod
    async def download_file_from_storage(
        key: str,
        temp: bool = False,
    ) -> dict[str, Any]:
        """
        Download a file from NodeTool storage.

        Args:
            key: File key/name to download
            temp: If True, download from temp storage; otherwise from asset storage

        Returns:
            File content (base64-encoded) and metadata
        """
        if "/" in key or "\\" in key:
            raise ValueError("Invalid key: path separators not allowed")

        scope = require_scope()
        storage = scope.get_temp_storage() if temp else scope.get_asset_storage()

        if not await storage.file_exists(key):
            raise ValueError(f"File not found: {key}")

        stream = BytesIO()
        await storage.download(key, stream)
        file_data = stream.getvalue()

        import base64

        size = await storage.get_size(key)
        last_modified = await storage.get_mtime(key)

        return {
            "key": key,
            "content": base64.b64encode(file_data).decode("utf-8"),
            "size": size,
            "last_modified": last_modified.isoformat() if last_modified else None,
            "storage": "temp" if temp else "asset",
        }

    @staticmethod
    async def get_file_metadata(
        key: str,
        temp: bool = False,
    ) -> dict[str, Any]:
        """
        Get metadata about a file in storage without downloading it.

        Args:
            key: File key/name
            temp: If True, check temp storage; otherwise check asset storage

        Returns:
            File metadata (size, last modified, etc.)
        """
        if "/" in key or "\\" in key:
            raise ValueError("Invalid key: path separators not allowed")

        scope = require_scope()
        storage = scope.get_temp_storage() if temp else scope.get_asset_storage()

        if not await storage.file_exists(key):
            raise ValueError(f"File not found: {key}")

        size = await storage.get_size(key)
        last_modified = await storage.get_mtime(key)

        return {
            "key": key,
            "exists": True,
            "size": size,
            "last_modified": last_modified.isoformat() if last_modified else None,
            "storage": "temp" if temp else "asset",
        }

    @staticmethod
    async def list_storage_files(
        temp: bool = False,
        limit: int = 100,
    ) -> dict[str, Any]:
        """
        List files in storage (note: this may not be supported by all storage backends).

        Args:
            temp: If True, list temp storage; otherwise list asset storage
            limit: Maximum number of files to return (default: 100, max: 200)

        Returns:
            List of file keys and metadata
        """
        if limit > 200:
            limit = 200

        scope = require_scope()
        storage = scope.get_temp_storage() if temp else scope.get_asset_storage()

        try:
            list_files_func = getattr(storage, "list_files", None)
            if callable(list_files_func):
                files = list_files_func(limit=limit)
                return {
                    "files": [
                        {
                            "key": f.get("key"),
                            "size": f.get("size"),
                            "last_modified": f.get("last_modified"),
                        }
                        for f in files[:limit]
                    ],
                    "count": len(files[:limit]),
                    "storage": "temp" if temp else "asset",
                }
            else:
                return {
                    "message": "Storage backend does not support listing files",
                    "storage": "temp" if temp else "asset",
                }
        except Exception as e:
            return {
                "error": str(e),
                "message": "Failed to list files - storage backend may not support this operation",
                "storage": "temp" if temp else "asset",
            }

    @staticmethod
    def get_tool_functions() -> dict[str, Any]:
        """Get all storage tool functions."""
        return {
            "download_file_from_storage": StorageTools.download_file_from_storage,
            "get_file_metadata": StorageTools.get_file_metadata,
            "list_storage_files": StorageTools.list_storage_files,
        }
