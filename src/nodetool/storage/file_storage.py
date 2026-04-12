from pathlib import Path
from typing import IO
from nodetool.storage.abstract_storage import AbstractStorage


class FileStorage(AbstractStorage):
    def __init__(self, base_path: str, base_url: str = ""):
        self.base_path = Path(base_path)
        self.base_url = base_url
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.base_path / key

    async def upload(self, key: str, data: IO) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data.read())

    async def download(self, key: str, dest: IO) -> None:
        with open(self._path(key), "rb") as f:
            dest.write(f.read())

    async def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()

    async def file_exists(self, key: str) -> bool:
        return self._path(key).exists()

    def get_url(self, key: str) -> str:
        if self.base_url:
            return f"{self.base_url}/{key}"
        return str(self._path(key))
