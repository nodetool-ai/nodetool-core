import io
import pytest
from datetime import datetime, timezone

from nodetool.storage.supabase_storage import SupabaseStorage


class _FakeBucket:
    def __init__(self, name: str, base_url: str, store: dict):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self._store = store.setdefault(name, {})  # per-bucket dict

    async def upload(self, path: str, content: bytes, *args, **kwargs):
        now = datetime.now(timezone.utc).isoformat()
        self._store[path] = {
            "bytes": content,
            "updated_at": now,
            "size": len(content),
        }
        return {"data": {"path": path}}

    async def download(self, path: str):
        entry = self._store.get(path)
        if not entry:
            raise FileNotFoundError(path)
        return entry["bytes"]

    async def remove(self, paths):
        for p in paths:
            self._store.pop(p, None)
        return {"data": True}

    async def list(self, path: str = ""):
        # Return only direct children under path
        res = []
        prefix = path.rstrip("/")
        for key, entry in self._store.items():
            parent, name = _split_path(key)
            if parent == prefix:
                res.append(
                    {
                        "name": name,
                        "updated_at": entry.get("updated_at"),
                        "size": entry.get("size"),
                    }
                )
        return res

    def get_public_url(self, path: str):
        return f"{self.base_url}/storage/v1/object/public/{self.name}/{path}"


class _FakeStorage:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self._buckets: dict = {}

    def from_(self, name: str):
        return _FakeBucket(name, self.base_url, self._buckets)


class _FakeSupabaseClient:
    def __init__(self, url: str):
        self.storage = _FakeStorage(url)


def _split_path(key: str):
    if "/" in key:
        idx = key.rfind("/")
        return key[:idx], key[idx + 1 :]
    return "", key


@pytest.fixture()
def storage() -> SupabaseStorage:
    client = _FakeSupabaseClient("https://example.supabase.co")
    return SupabaseStorage(
        bucket_name="assets",
        supabase_url="https://example.supabase.co",
        client=client, # type: ignore
    )


@pytest.mark.asyncio
async def test_upload_and_exists(storage: SupabaseStorage):
    key = "images/test.jpg"
    data = b"hello world"
    url = await storage.upload(key, io.BytesIO(data))
    assert url.endswith(key)
    assert await storage.file_exists(key)


@pytest.mark.asyncio
async def test_download_and_stream(storage: SupabaseStorage):
    key = "docs/readme.txt"
    data = b"some content here"
    await storage.upload(key, io.BytesIO(data))

    # download
    buff = io.BytesIO()
    await storage.download(key, buff)
    assert buff.getvalue() == data

    # stream
    chunks = b""
    async for chunk in storage.download_stream(key):
        chunks += chunk
    assert chunks == data


@pytest.mark.asyncio
async def test_mtime_and_size(storage: SupabaseStorage):
    key = "video/clip.mp4"
    data = b"0" * 1024
    await storage.upload(key, io.BytesIO(data))
    mtime = await storage.get_mtime(key)
    assert isinstance(mtime, datetime)
    size = await storage.get_size(key)
    assert size == len(data)


@pytest.mark.asyncio
async def test_delete(storage: SupabaseStorage):
    key = "audio/track.wav"
    await storage.upload(key, io.BytesIO(b"xyz"))
    assert await storage.file_exists(key)
    await storage.delete(key)
    assert not await storage.file_exists(key)


@pytest.mark.asyncio
async def test_get_url(storage: SupabaseStorage):
    key = "images/photo.png"
    url = storage.get_url(key)
    assert await url == f"https://example.supabase.co/storage/v1/object/public/assets/{key}"

