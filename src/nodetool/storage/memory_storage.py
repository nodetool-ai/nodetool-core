from typing import IO
from nodetool.storage.abstract_storage import AbstractStorage


class MemoryStorage(AbstractStorage):
    def __init__(self, base_url: str = ""):
        self._store: dict[str, bytes] = {}
        self.base_url = base_url

    async def upload(self, key: str, data: IO) -> None:
        self._store[key] = data.read()

    async def download(self, key: str, dest: IO) -> None:
        if key not in self._store:
            raise FileNotFoundError(key)
        dest.write(self._store[key])

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def file_exists(self, key: str) -> bool:
        return key in self._store

    def get_url(self, key: str) -> str:
        if self.base_url:
            return f"{self.base_url}/{key}"
        return f"memory://{key}"
