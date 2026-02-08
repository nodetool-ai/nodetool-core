import asyncio
import os
from datetime import datetime
from typing import IO, AsyncIterator

import aiofiles

from nodetool.storage.abstract_storage import AbstractStorage, FileMetadata


class FileStorage(AbstractStorage):
    base_path: str
    base_url: str

    def __init__(self, base_path: str, base_url: str):
        self.base_path = base_path
        self.base_url = base_url
        # Note: synchronous os.makedirs in __init__ is acceptable for initialization
        # Async initialization would require a factory pattern which is overkill here
        os.makedirs(base_path, exist_ok=True)

    async def get_url(self, key: str) -> str:
        return f"{self.base_url}/{key}"

    def generate_presigned_url(self, client_method: str, object_name: str, expiration=3600 * 24 * 7):
        return f"{self.base_url}/{object_name}"

    async def file_exists(self, key: str) -> bool:
        return await asyncio.to_thread(os.path.isfile, os.path.join(self.base_path, key))

    async def get_mtime(self, key: str):
        try:
            mtime = await asyncio.to_thread(os.path.getmtime, os.path.join(self.base_path, key))
            return datetime.fromtimestamp(mtime, tz=datetime.now().astimezone().tzinfo)
        except FileNotFoundError:
            return None

    async def get_size(self, key: str) -> int:
        return await asyncio.to_thread(os.path.getsize, os.path.join(self.base_path, key))

    async def download_stream(self, key: str) -> AsyncIterator[bytes]:
        async with aiofiles.open(os.path.join(self.base_path, key), "rb") as f:
            while True:
                chunk = await f.read(8192)
                if not chunk:
                    break
                yield chunk

    async def download(self, key: str, stream: IO):
        async with aiofiles.open(os.path.join(self.base_path, key), "rb") as f:
            while True:
                chunk = await f.read(8192)
                if not chunk:
                    break
                stream.write(chunk)

    async def upload(self, key: str, content: IO) -> str:
        full_path = os.path.join(self.base_path, key)
        await asyncio.to_thread(os.makedirs, os.path.dirname(full_path), exist_ok=True)
        async with aiofiles.open(full_path, "wb") as f:
            while True:
                chunk = content.read(1024 * 1024)  # Read in 1MB chunks
                if not chunk:
                    break
                await f.write(chunk)

        return self.generate_presigned_url("get_object", key)

    def upload_sync(self, key: str, content: IO) -> str:
        async def _write():
            full_path = os.path.join(self.base_path, key)
            await asyncio.to_thread(os.makedirs, os.path.dirname(full_path), exist_ok=True)
            async with aiofiles.open(full_path, "wb") as f:
                while True:
                    chunk = content.read(1024 * 1024)
                    if not chunk:
                        break
                    await f.write(chunk)

        asyncio.run(_write())
        return self.generate_presigned_url("get_object", key)

    async def delete(self, file_name: str):
        await asyncio.to_thread(os.remove, os.path.join(self.base_path, file_name))

    async def get_file_metadata(self, key: str) -> FileMetadata:
        """Optimized metadata retrieval using a single stat() call."""
        full_path = os.path.join(self.base_path, key)

        def _stat():
            try:
                stat_result = os.stat(full_path)
                return FileMetadata(
                    exists=True,
                    size=stat_result.st_size,
                    mtime=datetime.fromtimestamp(stat_result.st_mtime, tz=datetime.now().astimezone().tzinfo),
                )
            except FileNotFoundError:
                return FileMetadata(exists=False)

        return await asyncio.to_thread(_stat)
