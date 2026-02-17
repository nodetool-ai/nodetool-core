import asyncio
import functools
import hashlib
import inspect
import time
from collections.abc import Callable
from typing import Any, TypeVar

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

T = TypeVar("T")
K = TypeVar("K")


def _make_hashable(value: Any) -> Any:
    """
    Convert a value to a hashable type for use as a cache key.

    Args:
        value: Any value to make hashable.

    Returns:
        A hashable representation of the value.
    """
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_make_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in value.items()))
    if isinstance(value, set):
        return frozenset(_make_hashable(v) for v in value)

    # For unhashable types, use repr as fallback
    return repr(value)


def _normalize_args(func: Callable, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """
    Normalize arguments to handle keyword argument ordering.

    Args:
        func: The function being called.
        args: Positional arguments.
        kwargs: Keyword arguments.

    Returns:
        Normalized (args, kwargs) tuple.
    """
    # Get function signature
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()

    # Convert all to kwargs and sort
    normalized_kwargs = dict(bound_args.arguments)
    return (), normalized_kwargs


def _generate_cache_key(func: Callable, args: tuple, kwargs: dict) -> str:
    """
    Generate a cache key from function name and arguments.

    Args:
        func: The function being cached.
        args: Positional arguments.
        kwargs: Keyword arguments.

    Returns:
        A string cache key.
    """
    # Get function name
    func_name = func.__name__

    # Normalize arguments to handle kwarg ordering
    norm_args, norm_kwargs = _normalize_args(func, args, kwargs)

    # Make args and kwargs hashable
    hashable_args = _make_hashable(norm_args)
    hashable_kwargs = _make_hashable(norm_kwargs)

    # Create a string representation
    key_str = f"{func_name}:{hashable_args}:{hashable_kwargs}"

    # Use hashlib for a shorter, fixed-length key
    return hashlib.sha256(key_str.encode()).hexdigest()


class AsyncCache:
    """
    A simple async cache with time-to-live (TTL) support.

    This cache stores results of async functions and automatically
    expires entries based on their TTL. It's thread-safe for async
    contexts and provides efficient lookups.

    Example:
        cache = AsyncCache(default_ttl=60.0)

        @cache.cache_result
        async def fetch_user(user_id: int):
            return await database.get_user(user_id)

        # First call executes the function
        user1 = await fetch_user(1)

        # Subsequent calls within TTL return cached result
        user2 = await fetch_user(1)
    """

    def __init__(self, default_ttl: float = 300.0, max_size: int = 1000):
        """
        Initialize the cache.

        Args:
            default_ttl: Default time-to-live in seconds (default: 300).
            max_size: Maximum number of cached entries (default: 1000).
                     Uses FIFO eviction when limit is reached.
        """
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._cache: dict[str, tuple[Any, float]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        """
        Get a value from the cache.

        Args:
            key: Cache key.

        Returns:
            Cached value if exists and not expired, None otherwise.
        """
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            value, expiry = entry
            if time.time() > expiry:
                # Entry has expired
                del self._cache[key]
                return None

            return value

    async def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """
        Set a value in the cache.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time-to-live in seconds (uses default if None).
        """
        ttl = ttl if ttl is not None else self.default_ttl
        expiry = time.time() + ttl

        async with self._lock:
            # Evict oldest entry if at max size
            if len(self._cache) >= self.max_size and key not in self._cache:
                # Simple FIFO: remove first item
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            self._cache[key] = (value, expiry)

    async def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.

        Args:
            key: Cache key.

        Returns:
            True if key was deleted, False if it didn't exist.
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> None:
        """Clear all cached entries."""
        async with self._lock:
            self._cache.clear()

    async def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.

        Returns:
            Number of entries removed.
        """
        async with self._lock:
            now = time.time()
            expired_keys = [k for k, (_, expiry) in self._cache.items() if expiry < now]

            for key in expired_keys:
                del self._cache[key]

            return len(expired_keys)

    def cache_result(
        self,
        ttl: float | None = None,
        key_func: Callable[[Callable, tuple, dict], str] | None = None,
    ) -> Callable:
        """
        Decorator to cache async function results.

        Args:
            ttl: Time-to-live in seconds (uses cache default if None).
            key_func: Custom function to generate cache keys.
                      Signature: (func, args, kwargs) -> str

        Returns:
            Decorated async function.

        Example:
            @cache.cache_result(ttl=60.0)
            async def fetch_data(id: str):
                return await api_call(id)
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Generate cache key
                if key_func is not None:
                    cache_key = key_func(func, args, kwargs)
                else:
                    cache_key = _generate_cache_key(func, args, kwargs)

                # Try to get from cache
                cached_value = await self.get(cache_key)
                if cached_value is not None:
                    log.debug(f"Cache hit for key: {cache_key[:16]}...")
                    return cached_value

                # Cache miss - execute function
                log.debug(f"Cache miss for key: {cache_key[:16]}...")
                result = await func(*args, **kwargs)

                # Store in cache
                await self.set(cache_key, result, ttl)

                return result

            return wrapper

        return decorator

    async def invalidate(
        self,
        func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """
        Invalidate a specific cached function call.

        Args:
            func: The cached function.
            *args: Positional arguments of the cached call.
            **kwargs: Keyword arguments of the cached call.

        Returns:
            True if entry was invalidated, False otherwise.
        """
        cache_key = _generate_cache_key(func, args, kwargs)
        return await self.delete(cache_key)

    @property
    def size(self) -> int:
        """Return the current number of cached entries."""
        return len(self._cache)


def cached(
    ttl: float = 300.0,
    cache_instance: AsyncCache | None = None,
    key_func: Callable[[Callable, tuple, dict], str] | None = None,
) -> Callable:
    """
    Decorator to cache async function results with TTL support.

    This is a convenience function that uses either a provided cache
    or the global default cache.

    Args:
        ttl: Time-to-live in seconds (default: 300).
        cache_instance: Custom AsyncCache instance (uses default if None).
        key_func: Custom function to generate cache keys.

    Returns:
        Decorated async function.

    Example:
        @cached(ttl=60.0)
        async def fetch_user(user_id: int) -> dict:
            return await database.query(user_id)

        # First call executes the function
        user = await fetch_user(1)

        # Subsequent calls within 60 seconds return cached result
        user = await fetch_user(1)
    """

    def decorator(func: Callable) -> Callable:
        # Create a new cache instance for this function if none provided
        # This avoids sharing cache between different decorated functions
        cache = cache_instance if cache_instance is not None else AsyncCache(default_ttl=ttl)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate cache key
            if key_func is not None:
                cache_key = key_func(func, args, kwargs)
            else:
                cache_key = _generate_cache_key(func, args, kwargs)

            # Try to get from cache
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                log.debug(f"Cache hit for {func.__name__}: {cache_key[:16]}...")
                return cached_value

            # Cache miss - execute function
            log.debug(f"Cache miss for {func.__name__}: {cache_key[:16]}...")
            result = await func(*args, **kwargs)

            # Store in cache
            await cache.set(cache_key, result, ttl)

            return result

        # Add cache management methods to the wrapped function
        wrapper.cache = cache  # type: ignore[attr-defined]
        wrapper.cache_key = lambda *args, **kwargs: (  # type: ignore[attr-defined]
            _generate_cache_key(func, args, kwargs) if key_func is None
            else key_func(func, args, kwargs)
        )
        wrapper.invalidate = lambda *args, **kwargs: cache.invalidate(  # type: ignore[attr-defined]
            func, *args, **kwargs
        )

        return wrapper

    return decorator


__all__ = [
    "AsyncCache",
    "cached",
]
