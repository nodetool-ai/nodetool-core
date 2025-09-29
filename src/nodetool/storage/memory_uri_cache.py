from typing import Any, Optional
import time
import weakref

from nodetool.config.logging_config import get_logger

from .abstract_node_cache import AbstractNodeCache

log = get_logger(__name__)


class MemoryUriCache(AbstractNodeCache):
    """
    A simple in-memory cache keyed by URI strings (including memory:// URIs).

    - Default TTL is 5 minutes (300 seconds)
    - Expired entries are evicted lazily on get/set
    - Values can be arbitrary Python objects (e.g., PIL.Image, bytes, numpy arrays)
    """

    def __init__(self, default_ttl: int = 1800):
        self._cache: dict[str, tuple[Any, float]] = {}
        self._default_ttl = int(default_ttl) if default_ttl and default_ttl > 0 else 300

    def _now(self) -> float:
        return time.time()

    def _is_expired(self, expiry_time: float) -> bool:
        return self._now() >= expiry_time

    def _cleanup_expired(self) -> None:
        # Remove expired entries; avoid modifying dict during iteration
        to_delete: list[str] = []
        now = self._now()
        log.debug("Cleaning up expired entries: %s", self._cache)
        for key, (stored_value, expiry) in self._cache.items():
            if now >= expiry:
                to_delete.append(key)
                continue
        for key in to_delete:
            log.debug("Deleting expired entry: %s", key)
            self._cache.pop(key, None)

    def get(self, key: str) -> Any:
        if not key:
            return None
        item = self._cache.get(key)
        if item is None:
            return None
        stored_value, expiry = item
        if self._is_expired(expiry):
            # Evict and miss
            self._cache.pop(key, None)
            return None
        return stored_value

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        if not key:
            return
        # Enforce TTL with default 5 minutes even when ttl is 0/None
        ttl_seconds = self._default_ttl if not ttl or ttl <= 0 else int(ttl)
        expiry_time = self._now() + ttl_seconds

        self._cache[key] = (value, expiry_time)
        # Opportunistically clean up to keep memory bounded over time
        self._cleanup_expired()

    def clear(self):
        self._cache.clear()
