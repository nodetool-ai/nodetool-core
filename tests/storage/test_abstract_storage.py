"""Tests for AbstractStorage interface compliance."""

import io
from datetime import datetime
from unittest.mock import Mock

import pytest

from nodetool.storage.abstract_storage import AbstractStorage, FileMetadata


class MockStorage(AbstractStorage):
    """Mock implementation of AbstractStorage for testing."""

    def __init__(self):
        self.files = {}
        self.metadata = {}

    async def get_url(self, key: str) -> str:
        return f"https://example.com/{key}"

    async def file_exists(self, key: str) -> bool:
        return key in self.files

    async def get_mtime(self, key: str) -> datetime:
        if key in self.files:
            return self.metadata.get(key, {}).get("mtime", datetime.now())
        raise FileNotFoundError(f"File {key} not found")

    async def get_size(self, key: str) -> int:
        if key in self.files:
            return self.metadata.get(key, {}).get("size", 0)
        raise FileNotFoundError(f"File {key} not found")

    async def download_stream(self, key: str):
        if key in self.files:
            yield self.files[key]
        else:
            raise FileNotFoundError(f"File {key} not found")

    async def download(self, key: str, stream: io.IOBase):
        if key in self.files:
            stream.write(self.files[key])
        else:
            raise FileNotFoundError(f"File {key} not found")

    async def upload(self, key: str, content: io.IOBase) -> str:
        data = content.read()
        self.files[key] = data
        return f"https://example.com/{key}"

    def upload_sync(self, key: str, content: io.IOBase) -> str:
        data = content.read()
        self.files[key] = data
        return f"https://example.com/{key}"

    async def delete(self, file_name: str):
        if file_name in self.files:
            del self.files[file_name]
        else:
            raise FileNotFoundError(f"File {file_name} not found")


class TestFileMetadata:
    """Tests for FileMetadata dataclass."""

    def test_file_metadata_creation(self):
        """Test creating FileMetadata with required fields."""
        metadata = FileMetadata(exists=True)
        assert metadata.exists is True
        assert metadata.size is None
        assert metadata.mtime is None

    def test_file_metadata_with_all_fields(self):
        """Test creating FileMetadata with all fields."""
        now = datetime.now()
        metadata = FileMetadata(exists=True, size=1024, mtime=now)
        assert metadata.exists is True
        assert metadata.size == 1024
        assert metadata.mtime == now

    def test_file_metadata_not_exists(self):
        """Test FileMetadata for non-existent file."""
        metadata = FileMetadata(exists=False)
        assert metadata.exists is False


class TestAbstractStorageInterface:
    """Tests for AbstractStorage interface compliance."""

    @pytest.mark.asyncio
    async def test_storage_has_all_required_methods(self):
        """Test that storage implements all required async methods."""
        storage = MockStorage()
        assert hasattr(storage, "get_url")
        assert hasattr(storage, "file_exists")
        assert hasattr(storage, "get_mtime")
        assert hasattr(storage, "get_size")
        assert hasattr(storage, "download_stream")
        assert hasattr(storage, "download")
        assert hasattr(storage, "upload")
        assert hasattr(storage, "upload_sync")
        assert hasattr(storage, "delete")
        assert hasattr(storage, "get_file_metadata")

    @pytest.mark.asyncio
    async def test_get_url_returns_string(self):
        """Test that get_url returns a URL string."""
        storage = MockStorage()
        url = await storage.get_url("test.txt")
        assert isinstance(url, str)
        assert url.startswith("http")

    @pytest.mark.asyncio
    async def test_file_exists_returns_bool(self):
        """Test that file_exists returns boolean."""
        storage = MockStorage()
        storage.files["existing.txt"] = b"content"

        assert await storage.file_exists("existing.txt") is True
        assert await storage.file_exists("nonexistent.txt") is False

    @pytest.mark.asyncio
    async def test_get_mtime_returns_datetime(self):
        """Test that get_mtime returns datetime."""
        storage = MockStorage()
        storage.files["test.txt"] = b"content"
        storage.metadata["test.txt"] = {"mtime": datetime.now()}

        mtime = await storage.get_mtime("test.txt")
        assert isinstance(mtime, datetime)

    @pytest.mark.asyncio
    async def test_get_size_returns_int(self):
        """Test that get_size returns integer size."""
        storage = MockStorage()
        storage.files["test.txt"] = b"content"
        storage.metadata["test.txt"] = {"size": 1024}

        size = await storage.get_size("test.txt")
        assert isinstance(size, int)
        assert size == 1024

    @pytest.mark.asyncio
    async def test_upload_and_download(self):
        """Test that upload and download work correctly."""
        storage = MockStorage()
        content = io.BytesIO(b"test content")

        url = await storage.upload("test.txt", content)
        assert "test.txt" in url

        # Verify file was stored
        assert "test.txt" in storage.files
        assert storage.files["test.txt"] == b"test content"

    @pytest.mark.asyncio
    async def test_delete_removes_file(self):
        """Test that delete removes file."""
        storage = MockStorage()
        storage.files["test.txt"] = b"content"

        await storage.delete("test.txt")
        assert "test.txt" not in storage.files

    @pytest.mark.asyncio
    async def test_get_file_metadata_for_existing_file(self):
        """Test get_file_metadata for existing file."""
        storage = MockStorage()
        storage.files["test.txt"] = b"content"
        storage.metadata["test.txt"] = {"size": 1024, "mtime": datetime.now()}

        metadata = await storage.get_file_metadata("test.txt")
        assert metadata.exists is True
        assert metadata.size == 1024
        assert isinstance(metadata.mtime, datetime)

    @pytest.mark.asyncio
    async def test_get_file_metadata_for_nonexistent_file(self):
        """Test get_file_metadata for non-existent file."""
        storage = MockStorage()

        metadata = await storage.get_file_metadata("nonexistent.txt")
        assert metadata.exists is False
        assert metadata.size is None
        assert metadata.mtime is None

    @pytest.mark.asyncio
    async def test_download_to_stream(self):
        """Test download to IO stream."""
        storage = MockStorage()
        storage.files["test.txt"] = b"download content"

        output = io.BytesIO()
        await storage.download("test.txt", output)

        output.seek(0)
        assert output.read() == b"download content"

    @pytest.mark.asyncio
    async def test_upload_sync_returns_url(self):
        """Test synchronous upload returns URL."""
        storage = MockStorage()
        content = io.BytesIO(b"sync content")

        url = storage.upload_sync("test.txt", content)
        assert isinstance(url, str)
        assert "test.txt" in url
        assert "test.txt" in storage.files

    @pytest.mark.asyncio
    async def test_download_stream_yields_bytes(self):
        """Test that download_stream yields bytes."""
        storage = MockStorage()
        storage.files["test.txt"] = b"stream content"

        chunks = []
        async for chunk in storage.download_stream("test.txt"):
            chunks.append(chunk)

        assert b"".join(chunks) == b"stream content"
