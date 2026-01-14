from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import datetime
from typing import IO


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
