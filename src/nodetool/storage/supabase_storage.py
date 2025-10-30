from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import IO, Any, AsyncIterator

from .abstract_storage import AbstractStorage


class SupabaseStorage(AbstractStorage):
    """
    Storage adapter backed by Supabase Storage buckets.

    Notes:
    - `get_url` returns a public URL. Ensure the bucket is public for direct access.
    - For private buckets or signed URLs, extend to use Supabase's create_signed_url.
    - The provided client must expose `storage.from_(bucket)` with `upload`, `download`,
      `remove`, `list`, and `get_public_url` methods compatible with supabase-py v2.
    """

    def __init__(self, bucket_name: str, supabase_url: str, client: Any):
        self.bucket_name = bucket_name
        self.client = client
        # Construct the public base for direct object URLs (public buckets only)
        self._public_base = f"{supabase_url.rstrip('/')}/storage/v1/object/public/{bucket_name}"

    def _bucket(self):
        return self.client.storage.from_(self.bucket_name)

    def get_url(self, key: str) -> str:
        # Prefer client-provided public URL if available
        try:
            public = self._bucket().get_public_url(key)
            # Some client versions return dict with 'data' and 'publicUrl'
            if isinstance(public, dict):
                # supabase-py v2 returns { 'data': {'publicUrl': '...'}}
                data = public.get("data") or {}
                url = data.get("publicUrl") or data.get("public_url")
                if isinstance(url, str) and url:
                    return url
            if isinstance(public, str) and public:
                return public
        except Exception:
            pass
        # Fallback to computed public path (bucket must be public)
        return f"{self._public_base}/{key}"

    async def file_exists(self, key: str) -> bool:
        # Try listing the parent folder and searching for the file
        path, name = _split_path(key)
        try:
            entries = await _maybe_await(self._bucket().list(path))
            for e in entries or []:
                # supabase returns dicts with 'name' among other fields
                if isinstance(e, dict) and e.get("name") == name:
                    return True
            return False
        except Exception:
            # As a fallback, try download to check existence
            try:
                await _maybe_await(self._bucket().download(key))
                return True
            except Exception:
                return False

    async def get_mtime(self, key: str) -> datetime:
        path, name = _split_path(key)
        entries = await _maybe_await(self._bucket().list(path))
        for e in entries or []:
            if isinstance(e, dict) and e.get("name") == name:
                # Prefer RFC3339 fields commonly returned by Supabase
                ts = e.get("updated_at") or e.get("last_modified") or e.get("created_at")
                if isinstance(ts, str):
                    try:
                        # Handle Zulu times
                        if ts.endswith("Z"):
                            ts = ts[:-1] + "+00:00"
                        return datetime.fromisoformat(ts).astimezone(timezone.utc)
                    except Exception:
                        pass
                # If no timestamp, return now as a safe fallback
                return datetime.now(timezone.utc)
        # Not found: raise to keep semantics consistent with other adapters
        raise FileNotFoundError(f"File not found: {key}")

    async def get_size(self, key: str) -> int:
        path, name = _split_path(key)
        entries = await _maybe_await(self._bucket().list(path))
        for e in entries or []:
            if isinstance(e, dict) and e.get("name") == name:
                size = e.get("size") or (e.get("metadata") or {}).get("size")
                if isinstance(size, int):
                    return size
        # Fallback: download and measure
        data = await _maybe_await(self._bucket().download(key))
        if isinstance(data, (bytes, bytearray)):
            return len(data)
        # Some clients might return a response-like object
        body = getattr(data, "data", None) or getattr(data, "content", None)
        if isinstance(body, (bytes, bytearray)):
            return len(body)
        raise FileNotFoundError(f"File not found or size unknown: {key}")

    async def download_stream(self, key: str) -> AsyncIterator[bytes]:
        data = await _maybe_await(self._bucket().download(key))
        payload = _to_bytes(data)
        # Yield in 8KB chunks
        for i in range(0, len(payload), 8192):
            yield payload[i : i + 8192]

    async def download(self, key: str, stream: IO):
        data = await _maybe_await(self._bucket().download(key))
        stream.write(_to_bytes(data))

    async def upload(self, key: str, content: IO) -> str:
        # Read all bytes to support both sync and async upload APIs
        # (supabase storage usually accepts raw bytes)
        content.seek(0)
        bytes_data = content.read()
        await _maybe_await(self._bucket().upload(key, bytes_data))
        return self.get_url(key)

    def upload_sync(self, key: str, content: IO) -> str:
        async def _run():
            return await self.upload(key, content)

        return asyncio.run(_run())

    async def delete(self, file_name: str):
        await _maybe_await(self._bucket().remove([file_name]))


def _split_path(key: str) -> tuple[str, str]:
    if "/" in key:
        idx = key.rfind("/")
        return key[:idx], key[idx + 1 :]
    return "", key


def _to_bytes(data: Any) -> bytes:
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    # supabase-py may return objects with a `.data` or `.content` attribute
    payload = getattr(data, "data", None) or getattr(data, "content", None)
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    # As a last resort, try reading from a file-like object
    read = getattr(data, "read", None)
    if callable(read):
        return read() or b""
    raise TypeError("Unsupported download payload type for Supabase storage")


async def _maybe_await(v: Any) -> Any:
    if asyncio.iscoroutine(v):
        return await v
    return v

