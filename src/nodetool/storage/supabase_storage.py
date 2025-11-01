from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import IO, Any, AsyncIterator

from nodetool.models.supabase_adapter import SupabaseAsyncClient

from .abstract_storage import AbstractStorage


class SupabaseStorage(AbstractStorage):
    """
    Storage adapter backed by Supabase Storage buckets.

    Notes:
    - `get_url` returns a public URL. Ensure the bucket is public for direct access.
    - For private buckets or signed URLs, extend to use Supabase's create_signed_url.
    - The provided client must expose `storage.from_(bucket)` with `upload`, `download`,
      `remove`, `list`, and `get_public_url` methods compatible with supabase-py v2.
    """

    def __init__(self, bucket_name: str, supabase_url: str, client: SupabaseAsyncClient):
        self.bucket_name = bucket_name
        self.client = client
        self.bucket = self.client.storage.from_(self.bucket_name)

    async def get_url(self, key: str) -> str:
        return await self.bucket.get_public_url(key)

    async def _info(self, key: str) -> dict:
        """
        {'id': '53b47f2e-c362-4658-8672-c4a205dea61e', 
        'name': 'cfe027d2b6e811f0b6ce0000516a875d_thumb.jpg', 
        'version': '0c92a31f-a66f-4ca3-8134-1be8ffd2fe06', 
        'bucket_id': 'assets', 
        'size': 41822, 
        'content_type': 'text/plain', 
        'cache_control': 'no-cache', 
        'etag': '"07451a7dd937d79d5d106c37d6a15f55"', 
        'metadata': {}, 
        'last_modified': '2025-11-01T06:05:49.592Z', 
        'created_at': '2025-11-01T06:05:49.592Z'}
        """
        return await self.bucket.info(key)

    async def file_exists(self, key: str) -> bool:
        # Try listing the parent folder and searching for the file
        return await self.bucket.exists(key)

    async def get_mtime(self, key: str) -> datetime:
        info = await self._info(key)
        last_modified = info.get("last_modified")
        assert last_modified

        # Parse ISO8601 date string with fractional seconds and Z (UTC)
        # Example: '2025-11-01T06:05:49.592Z'
        dt = datetime.strptime(last_modified, "%Y-%m-%dT%H:%M:%S.%fZ")
        return dt.replace(tzinfo=timezone.utc)

    async def get_size(self, key: str) -> int:
        info = await self._info(key)
        return info["size"]

    async def download_stream(self, key: str) -> AsyncIterator[bytes]:
        payload = await self.bucket.download(key)
        for i in range(0, len(payload), 8192):
            yield payload[i : i + 8192]

    async def download(self, key: str, stream: IO):
        data = await self.bucket.download(key)
        stream.write(data)

    async def upload(self, key: str, content: IO) -> str:
        content.seek(0)
        await self.bucket.upload(key, content) # type: ignore
        res = await self.bucket.create_signed_url(key, 3600*24)
        return res["signedURL"]

    def upload_sync(self, key: str, content: IO) -> str:
        return asyncio.run(self.upload(key, content))

    async def delete(self, file_name: str):
        await self.bucket.remove([file_name])