import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import IO, AsyncIterator


@dataclass
class FileMetadata:
    """Batch file metadata to reduce system calls."""
    exists: bool
    size: int | None = None
    mtime: datetime | None = None


class AbstractStorage(ABC):
    @abstractmethod
    async def get_url(self, key: str) -> str:
        pass

    @abstractmethod
    async def file_exists(self, key: str) -> bool:
        pass

    @abstractmethod
    async def get_mtime(self, key: str) -> datetime:
        pass

    @abstractmethod
    async def get_size(self, key: str) -> int:
        pass

    @abstractmethod
    def download_stream(self, key: str) -> AsyncIterator[bytes]:
        pass

    @abstractmethod
    async def download(self, key: str, stream: IO):
        pass

    @abstractmethod
    async def upload(self, key: str, content: IO) -> str:
        pass

    @abstractmethod
    def upload_sync(self, key: str, content: IO) -> str:
        pass

    @abstractmethod
    async def delete(self, file_name: str):
        pass

    async def get_file_metadata(self, key: str) -> FileMetadata:
        """Get file metadata in a single system call for better performance.

        Default implementation calls individual methods, but concrete implementations
        can override this for better performance (e.g., single stat() call).
        """
        exists = await self.file_exists(key)
        if not exists:
            return FileMetadata(exists=False)

        size, mtime = await asyncio.gather(
            self.get_size(key),
            self.get_mtime(key),
        )
        return FileMetadata(exists=True, size=size, mtime=mtime)
