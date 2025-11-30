"""
Model Cache Module

Provides disk-based caching for model discovery across all providers.
Caches model lists with a 6-hour TTL to reduce API calls.
"""

import hashlib
import pickle
import time
from pathlib import Path
from typing import Any, TypeVar

from nodetool.config.logging_config import get_logger
from nodetool.config.settings import get_system_cache_path

log = get_logger(__name__)

T = TypeVar("T")


class ModelCache:
    """
    A class to manage disk-based caching with TTL support.
    Caches data in the nodetool cache folder.
    """

    def __init__(self, cache_subdir: str = "models"):
        """
        Initialize disk cache.

        Args:
            cache_subdir: Subdirectory name within the nodetool cache folder
        """
        self.cache_dir = get_system_cache_path(cache_subdir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Generate a cache file path from a key."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def get(self, key: str) -> Any:
        """
        Retrieve cached value if it exists and hasn't expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                _stored_key, value, expiry_time = pickle.load(f)

            if time.time() < expiry_time:
                log.debug(f"Cache hit for key: {key}")
                return value
            else:
                # Remove expired cache file
                log.debug(f"Cache expired for key: {key}")
                cache_path.unlink(missing_ok=True)
                return None
        except Exception as e:
            log.warning(f"Error reading cache for key {key}: {e}")
            # Remove corrupted cache file
            cache_path.unlink(missing_ok=True)
            return None

    def set(self, key: str, value: Any, ttl: int = 21600):
        """
        Store value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default: 21600 = 6 hours)
        """
        cache_path = self._get_cache_path(key)
        expiry_time = time.time() + ttl

        try:
            log.debug(
                f"Attempting to cache key: {key}, value type: {type(value)}, length: {len(value) if hasattr(value, '__len__') else 'N/A'}"
            )
            with open(cache_path, "wb") as f:
                # Store key along with value and expiry for pattern matching
                pickle.dump((key, value, expiry_time), f)
            log.debug(
                f"✓ Successfully cached value for key: {key} (TTL: {ttl}s) at {cache_path}"
            )
        except Exception as e:
            log.error(f"✗ Error writing cache for key {key}: {e}", exc_info=True)
            import traceback

            log.error(f"Traceback: {traceback.format_exc()}")

    def delete(self, key: str):
        """
        Delete a specific cache entry.

        Args:
            key: Cache key to delete
        """
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink(missing_ok=True)
            log.debug(f"Deleted cache entry for key: {key}")

    def delete_pattern(self, pattern: str):
        """
        Delete all cache entries matching a pattern.

        Args:
            pattern: Pattern to match against cache keys (e.g., "cached_hf_*")
                    Supports wildcards: * (any chars), ? (single char)
        """
        import fnmatch

        deleted_count = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                with open(cache_file, "rb") as f:
                    stored_key, _, _ = pickle.load(f)

                if fnmatch.fnmatch(stored_key, pattern):
                    cache_file.unlink(missing_ok=True)
                    deleted_count += 1
                    log.debug(f"Deleted cache entry for key: {stored_key}")
            except Exception as e:
                log.warning(f"Error processing cache file {cache_file}: {e}")
                # Remove corrupted cache file
                cache_file.unlink(missing_ok=True)

        if deleted_count > 0:
            log.info(
                f"Deleted {deleted_count} cache entries matching pattern: {pattern}"
            )
        else:
            log.debug(f"No cache entries found matching pattern: {pattern}")

    def clear(self):
        """Remove all cached files."""
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink(missing_ok=True)
        log.info("Model cache cleared")


# Global cache instance
_model_cache = ModelCache()
