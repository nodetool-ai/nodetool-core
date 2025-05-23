from datetime import datetime
import io
from typing import AsyncIterator, Dict, Iterator
from nodetool.common.async_iterators import AsyncByteStream
from nodetool.storage.abstract_storage import AbstractStorage


class MemoryStorage(AbstractStorage):
    storage: Dict[str, bytes]
    mtimes: Dict[str, datetime]
    base_url: str

    def __init__(self, base_url: str):
        self.storage = {}
        self.mtimes = {}
        self.base_url = base_url

    def get_url(self, key: str):
        return f"{self.base_url}/{key}"

    async def file_exists(self, file_name: str) -> bool:
        return file_name in self.storage

    async def get_mtime(self, key: str):
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

    def upload_stream(self, key: str, content: Iterator[bytes]) -> str:
        bytes_io = io.BytesIO()
        for chunk in content:
            bytes_io.write(chunk)
        bytes_io.seek(0)
        self.storage[key] = bytes_io.getvalue()
        return self.get_url(key)

    async def download(self, key: str, stream: io.BytesIO):
        if key in self.storage:
            stream.write(self.storage[key])
        else:
            raise FileNotFoundError(f"File {key} not found")

    async def upload(self, key: str, content: io.BytesIO) -> str:
        self.storage[key] = content.read()
        return self.get_url(key)

    def upload_sync(self, key: str, content: io.BytesIO) -> str:
        self.storage[key] = content.read()
        return self.get_url(key)

    async def delete(self, file_name: str):
        if file_name in self.storage:
            del self.storage[file_name]
