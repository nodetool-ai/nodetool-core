from datetime import datetime
import asyncio
import os
import aiofiles
from typing import IO, AsyncIterator

from nodetool.storage.abstract_storage import AbstractStorage


class FileStorage(AbstractStorage):
    base_path: str
    base_url: str

    def __init__(self, base_path: str, base_url: str):
        self.base_path = base_path
        self.base_url = base_url
        os.makedirs(base_path, exist_ok=True)

    def get_url(self, key: str):
        return f"{self.base_url}/{key}"

    def generate_presigned_url(
        self, client_method: str, object_name: str, expiration=3600 * 24 * 7
    ):
        return f"{self.base_url}/{object_name}"

    async def file_exists(self, file_name: str) -> bool:
        return await asyncio.to_thread(
            os.path.isfile, os.path.join(self.base_path, file_name)
        )

    async def get_mtime(self, key: str):
        try:
            mtime = await asyncio.to_thread(
                os.path.getmtime, os.path.join(self.base_path, key)
            )
            return datetime.fromtimestamp(mtime, tz=datetime.now().astimezone().tzinfo)
        except FileNotFoundError:
            return None

    async def get_size(self, key: str) -> int:
        return await asyncio.to_thread(
            os.path.getsize, os.path.join(self.base_path, key)
        )

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
        async with aiofiles.open(os.path.join(self.base_path, key), "wb") as f:
            while True:
                chunk = content.read(1024 * 1024)  # Read in 1MB chunks
                if not chunk:
                    break
                await f.write(chunk)

        return self.generate_presigned_url("get_object", key)

    def upload_sync(self, key: str, content: IO) -> str:
        async def _write():
            async with aiofiles.open(os.path.join(self.base_path, key), "wb") as f:
                while True:
                    chunk = content.read(1024 * 1024)
                    if not chunk:
                        break
                    await f.write(chunk)

        asyncio.run(_write())
        return self.generate_presigned_url("get_object", key)

    async def delete(self, file_name: str):
        await asyncio.to_thread(os.remove, os.path.join(self.base_path, file_name))
