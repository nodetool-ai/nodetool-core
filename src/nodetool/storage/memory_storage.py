import io
from datetime import datetime
from typing import IO, AsyncIterator, Iterator

from nodetool.concurrency.async_iterators import AsyncByteStream
from nodetool.storage.abstract_storage import AbstractStorage, FileMetadata


class MemoryStorage(AbstractStorage):
    storage: dict[str, bytes]
    mtimes: dict[str, datetime]
    base_url: str

    def __init__(self, base_url: str):
        self.storage = {}
        self.mtimes = {}
        self.base_url = base_url

    async def get_url(self, key: str) -> str:
        return f"{self.base_url}/{key}"

    async def file_exists(self, key: str) -> bool:
        return key in self.storage

    async def get_mtime(self, key: str) -> datetime:
        return self.mtimes.get(key, datetime.now())

    async def get_size(self, key: str) -> int:
        if key in self.storage:
            return len(self.storage[key])
        else:
            raise FileNotFoundError(f"File {key} not found")

    def download_stream(self, key: str) -> AsyncIterator[bytes]:
        if key in self.storage:
            return AsyncByteStream(self.storage[key])
        else:
            raise FileNotFoundError(f"File {key} not found")

    async def upload_stream(self, key: str, content: Iterator[bytes]) -> str:
        bytes_io = io.BytesIO()
        for chunk in content:
            bytes_io.write(chunk)
        bytes_io.seek(0)
        self.storage[key] = bytes_io.getvalue()
        return f"{self.base_url}/{key}"

    async def download(self, key: str, stream: IO):
        if key in self.storage:
            stream.write(self.storage[key])
        else:
            raise FileNotFoundError(f"File {key} not found")

    async def upload(self, key: str, content: IO) -> str:
        self.storage[key] = content.read()
        return f"{self.base_url}/{key}"

    def upload_sync(self, key: str, content: IO) -> str:
        self.storage[key] = content.read()
        return f"{self.base_url}/{key}"

    async def delete(self, file_name: str) -> None:
        if file_name in self.storage:
            del self.storage[file_name]

    async def get_file_metadata(self, key: str) -> FileMetadata:
        """Optimized metadata retrieval for in-memory storage."""
        if key in self.storage:
            return FileMetadata(
                exists=True,
                size=len(self.storage[key]),
                mtime=self.mtimes.get(key, datetime.now()),
            )
        return FileMetadata(exists=False)
