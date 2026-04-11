from abc import ABC, abstractmethod
from typing import IO


class AbstractStorage(ABC):
    @abstractmethod
    async def upload(self, key: str, data: IO) -> None: ...
    @abstractmethod
    async def download(self, key: str, dest: IO) -> None: ...
    @abstractmethod
    async def delete(self, key: str) -> None: ...
    @abstractmethod
    async def file_exists(self, key: str) -> bool: ...
    def get_url(self, key: str) -> str:
        return ""
