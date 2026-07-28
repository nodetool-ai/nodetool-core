"""
Resource scope management for per-execution isolation.
"""

from __future__ import annotations

import contextvars
import time
from typing import Any, Optional

import httpx

from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

_current_scope: contextvars.ContextVar[Optional[ResourceScope]] = contextvars.ContextVar(
    "_current_scope", default=None
)


def require_scope() -> ResourceScope:
    scope = _current_scope.get()
    if scope is None:
        raise RuntimeError("No ResourceScope is currently bound")
    return scope


def maybe_scope() -> Optional[ResourceScope]:
    return _current_scope.get()


class _MemoryUriCache:
    """Simple in-memory TTL cache for URI objects."""

    def __init__(self, default_ttl: int = 300):
        self._store: dict[str, tuple[Any, float]] = {}
        self._ttl = default_ttl

    def get(self, key: str) -> Any:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires = entry
        if time.monotonic() > expires:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        self._store[key] = (value, time.monotonic() + (ttl or self._ttl))

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()


class ResourceScope:
    """Per-execution resource scope."""

    def __init__(self) -> None:
        self._token: Optional[contextvars.Token] = None
        self._asset_storage: Any = None
        self._temp_storage: Any = None
        self._memory_uri_cache: _MemoryUriCache | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._owns_http_client = False
        self._closed = False

        # Remember the parent scope, but resolve its resources lazily: creating
        # a nested scope must not mkdir the asset folder or open an HTTP client
        # for a node that never touches assets or the network.
        self._parent: Optional[ResourceScope] = maybe_scope()

    def _check_open(self, what: str) -> None:
        if self._closed:
            raise RuntimeError(
                f"Cannot create {what}: this ResourceScope has already exited. "
                "The scope was likely used outside its async context (for example "
                "from an async generator resumed by a different task)."
            )

    async def __aenter__(self) -> ResourceScope:
        self._token = _current_scope.set(self)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._closed = True
        if self._token is not None:
            token, self._token = self._token, None
            try:
                _current_scope.reset(token)
            except (ValueError, RuntimeError) as e:
                log.warning(
                    "ResourceScope exited in a different context than it was entered "
                    f"({e}); the current scope contextvar was left untouched."
                )
        if self._http_client is not None and self._owns_http_client:
            try:
                await self._http_client.aclose()
            except Exception as e:
                log.warning(f"Error closing HTTP client: {e}")
        self._asset_storage = None
        self._temp_storage = None
        self._memory_uri_cache = None
        self._http_client = None
        self._owns_http_client = False
        self._parent = None

    def get_asset_storage(self) -> Any:
        if self._asset_storage is None:
            if self._parent is not None:
                self._asset_storage = self._parent.get_asset_storage()
                return self._asset_storage
            self._check_open("asset storage")
            from nodetool.storage.file_storage import FileStorage

            self._asset_storage = FileStorage(
                base_path=Environment.get_asset_folder(),
                base_url=Environment.get_storage_api_url(),
            )
        return self._asset_storage

    def get_temp_storage(self) -> Any:
        if self._temp_storage is None:
            if self._parent is not None:
                self._temp_storage = self._parent.get_temp_storage()
                return self._temp_storage
            self._check_open("temp storage")
            from nodetool.storage.memory_storage import MemoryStorage

            self._temp_storage = MemoryStorage(
                base_url=Environment.get_temp_storage_api_url(),
            )
        return self._temp_storage

    def get_memory_uri_cache(self) -> _MemoryUriCache:
        if self._memory_uri_cache is None:
            if self._parent is not None:
                self._memory_uri_cache = self._parent.get_memory_uri_cache()
                return self._memory_uri_cache
            self._check_open("memory URI cache")
            self._memory_uri_cache = _MemoryUriCache(default_ttl=300)
        return self._memory_uri_cache

    def get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            if self._parent is not None:
                self._http_client = self._parent.get_http_client()
                return self._http_client
            self._check_open("HTTP client")
            # 🛡️ Sentinel: Enforce SSL certificate verification to prevent MITM attacks
            self._http_client = httpx.AsyncClient(
                verify=True,
                follow_redirects=True,
                timeout=600,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "*/*",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
            self._owns_http_client = True
        return self._http_client
