import threading
import time
from typing import Any

from nodetool.config.logging_config import get_logger

from .abstract_node_cache import AbstractNodeCache

log = get_logger(__name__)


class MemoryUriCache(AbstractNodeCache):
    """
    A simple in-memory cache keyed by URI strings (including memory:// URIs).

    - Default TTL is 5 minutes (300 seconds)
    - Expired entries are evicted lazily on get/set
    - Values can be arbitrary Python objects (e.g., PIL.Image, bytes, numpy arrays)
    - Can be initialized with existing data from a shared dict
    """

    def __init__(self, default_ttl: int = 1800, initial_data: dict[str, Any] | None = None):
        self._cache: dict[str, tuple[Any, float]] = {}
        self._default_ttl = int(default_ttl) if default_ttl and default_ttl > 0 else 300
        # If initial_data is provided, use it as the backing store (but convert to our format)
        if initial_data is not None:
            # Use the provided dict directly as the backing store
            self._cache = initial_data
            # Convert any non-tuple values to tuples with far-future expiry
            for key, value in list(self._cache.items()):
                if not isinstance(value, tuple) or len(value) != 2:
                    self._cache[key] = (value, float("inf"))  # Never expires

    def _now(self) -> float:
        return time.time()

    def _is_expired(self, expiry_time: float) -> bool:
        return self._now() >= expiry_time

    def _cleanup_expired(self) -> None:
        # Remove expired entries; avoid modifying dict during iteration
        to_delete: list[str] = []
        now = self._now()
        log.debug("Cleaning up expired entries: %s", self._cache)
        for key, (_stored_value, expiry) in self._cache.items():
            if now >= expiry:
                to_delete.append(key)
                continue
        for key in to_delete:
            log.debug("Deleting expired entry: %s", key)
            self._cache.pop(key, None)

    def get(self, key: str) -> Any:
        if not key:
            return None
        thread_id = threading.get_ident()
        item = self._cache.get(key)
        if item is None:
            log.debug(f"Cache MISS for key '{key}' on thread {thread_id}")
            return None

        # Handle both tuple format (value, expiry) and raw value format
        if isinstance(item, tuple) and len(item) == 2:
            stored_value, expiry = item
            if self._is_expired(expiry):
                # Evict and miss
                self._cache.pop(key, None)
                log.debug(f"Cache EXPIRED for key '{key}' on thread {thread_id}")
                return None
            log.debug(f"Cache HIT for key '{key}' on thread {thread_id}")
            return stored_value
        else:
            # Raw value stored directly (from ProcessingContext._memory_set)
            log.debug(f"Cache HIT for key '{key}' on thread {thread_id} (raw value)")
            return item

    def set(self, key: str, value: Any, ttl: int | None = None):
        if not key:
            return
        thread_id = threading.get_ident()
        # Enforce TTL with default 5 minutes even when ttl is 0/None
        ttl_seconds = self._default_ttl if not ttl or ttl <= 0 else int(ttl)
        expiry_time = self._now() + ttl_seconds

        log.debug(f"Cache SET for key '{key}' on thread {thread_id} (TTL: {ttl_seconds}s)")
        # Always store as tuple for our internal format
        self._cache[key] = (value, expiry_time)
        # Opportunistically clean up to keep memory bounded over time
        self._cleanup_expired()

    def clear(self):
        self._cache.clear()
