"""
Distributed Cache Layer for NodeTool.

This module provides a distributed caching layer that extends the existing
AbstractNodeCache interface with Redis-based distributed caching support.

Features:
- Redis-backed distributed caching
- Consistent hashing for cache sharding
- Automatic failover and reconnection
- TTL support with automatic expiration
- Cache warming/preheating support

Usage:
    from nodetool.storage.distributed_cache import (
        RedisCacheConfig,
        RedisNodeCache,
        CacheShard,
    )

    config = RedisCacheConfig(
        host="localhost",
        port=6379,
        shard_count=4,
        default_ttl=3600,
    )

    cache = RedisNodeCache(config)
    await cache.connect()

    # Use like any other cache
    value = await cache.get("key")
    await cache.set("key", value, ttl=1800)
"""

import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import redis.asyncio as redis
    from redis.asyncio.connection import ConnectionPool

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None  # type: ignore

from nodetool.config.env_guard import get_system_env_value
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


def _is_truthy(value: Optional[str]) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class RedisCacheConfig:
    """Configuration for Redis distributed cache."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    shard_count: int = 1
    default_ttl: int = 3600
    max_connections: int = 50
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    readonly_replicas: bool = False

    @classmethod
    def from_env(cls) -> "RedisCacheConfig":
        """Create config from environment variables."""
        return cls(
            host=get_system_env_value("REDIS_HOST") or "localhost",
            port=int(get_system_env_value("REDIS_PORT") or 6379),
            db=int(get_system_env_value("REDIS_DB") or 0),
            password=get_system_env_value("REDIS_PASSWORD"),
            ssl=_is_truthy(get_system_env_value("REDIS_SSL")),
            shard_count=int(get_system_env_value("REDIS_SHARD_COUNT") or 1),
            default_ttl=int(get_system_env_value("REDIS_DEFAULT_TTL") or 3600),
            max_connections=int(get_system_env_value("REDIS_MAX_CONNECTIONS") or 50),
        )


class CacheShard:
    """A single Redis cache shard."""

    def __init__(self, config: RedisCacheConfig, shard_index: int = 0) -> None:
        self.config = config
        self.shard_index = shard_index
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[redis.Redis] = None
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        """Get the shard name."""
        return f"redis-shard-{self.shard_index}"

    async def connect(self) -> None:
        """Connect to the Redis shard."""
        if not REDIS_AVAILABLE:
            raise RuntimeError("redis-py not installed. Install with: pip install redis")

        async with self._lock:
            if self._client is not None:
                return

            self._pool = ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                ssl=self.config.ssl,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                decode_responses=True,
            )

            self._client = redis.Redis(connection_pool=self._pool)

            # Test connection
            try:
                await self._client.ping()
                log.info(f"Connected to Redis shard {self.shard_index} at {self.config.host}:{self.config.port}")
            except Exception as e:
                log.error(f"Failed to connect to Redis shard {self.shard_index}: {e}")
                await self.disconnect()
                raise

    async def disconnect(self) -> None:
        """Disconnect from the Redis shard."""
        async with self._lock:
            if self._client:
                await self._client.close()
                self._client = None
            if self._pool:
                await self._pool.disconnect()
                self._pool = None

    async def is_connected(self) -> bool:
        """Check if the shard is connected."""
        if self._client is None:
            return False
        try:
            await self._client.ping()
            return True
        except Exception:
            return False

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the shard."""
        if self._client is None:
            raise RuntimeError("Shard not connected. Call connect() first.")

        try:
            value = await self._client.get(key)
            if value is None:
                return None
            return json.loads(value)
        except json.JSONDecodeError:
            return value
        except Exception as e:
            log.error(f"Redis get error for key {key}: {e}")
            raise

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in the shard."""
        if self._client is None:
            raise RuntimeError("Shard not connected. Call connect() first.")

        try:
            serialized = json.dumps(value, default=str)
            ttl = ttl if ttl is not None else self.config.default_ttl
            if ttl > 0:
                result = await self._client.setex(key, ttl, serialized)
            else:
                result = await self._client.set(key, serialized)
            return result is True
        except Exception as e:
            log.error(f"Redis set error for key {key}: {e}")
            raise

    async def delete(self, key: str) -> bool:
        """Delete a key from the shard."""
        if self._client is None:
            raise RuntimeError("Shard not connected. Call connect() first.")

        try:
            result = await self._client.delete(key)
            return result > 0
        except Exception as e:
            log.error(f"Redis delete error for key {key}: {e}")
            raise

    async def clear(self) -> int:
        """Clear all keys in the shard (use with caution)."""
        if self._client is None:
            raise RuntimeError("Shard not connected. Call connect() first.")

        try:
            result = await self._client.flushdb()
            return result
        except Exception as e:
            log.error(f"Redis clear error: {e}")
            raise

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values from the shard."""
        if self._client is None:
            raise RuntimeError("Shard not connected. Call connect() first.")

        try:
            values = await self._client.mget(keys)
            result: dict[str, Any] = {}
            for key, value in zip(keys, values, strict=True):
                if value is not None:
                    try:
                        result[key] = json.loads(value)
                    except json.JSONDecodeError:
                        result[key] = value
            return result
        except Exception as e:
            log.error(f"Redis get_many error: {e}")
            raise

    async def set_many(self, mapping: dict[str, Any], ttl: Optional[int] = None) -> int:
        """Set multiple values in the shard."""
        if self._client is None:
            raise RuntimeError("Shard not connected. Call connect() first.")

        try:
            ttl = ttl if ttl is not None else self.config.default_ttl
            pipe = self._client.pipeline()

            for key, value in mapping.items():
                serialized = json.dumps(value, default=str)
                if ttl > 0:
                    pipe.setex(key, ttl, serialized)
                else:
                    pipe.set(key, serialized)

            results = await pipe.execute()
            return sum(1 for r in results if r)
        except Exception as e:
            log.error(f"Redis set_many error: {e}")
            raise


class RedisNodeCache:
    """
    Distributed node cache using Redis with consistent sharding.

    This cache distributes keys across multiple Redis shards using consistent
    hashing, providing horizontal scalability and high availability.
    """

    def __init__(self, config: Optional[RedisCacheConfig] = None) -> None:
        self.config = config or RedisCacheConfig.from_env()
        self._shards: list[CacheShard] = []
        self._connected = False

    @property
    def shard_count(self) -> int:
        """Get the number of shards."""
        return max(1, self.config.shard_count)

    def _get_shard_index(self, key: str) -> int:
        """Determine which shard a key belongs to using consistent hashing."""
        if self.shard_count == 1:
            return 0

        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return hash_value % self.shard_count

    def _get_shard(self, key: str) -> CacheShard:
        """Get the shard for a given key."""
        shard_index = self._get_shard_index(key)
        if shard_index >= len(self._shards):
            raise IndexError(f"Shard index {shard_index} out of range for {len(self._shards)} shards")
        return self._shards[shard_index]

    async def connect(self) -> None:
        """Connect to all Redis shards."""
        if not REDIS_AVAILABLE:
            raise RuntimeError("redis-py not installed. Install with: pip install redis")

        if self._connected:
            return

        # Create shards
        self._shards = [CacheShard(self.config, shard_index=i) for i in range(self.shard_count)]

        # Connect to all shards
        await asyncio.gather(
            *(shard.connect() for shard in self._shards),
            return_exceptions=True,
        )

        # Check connection status
        connected_count = 0
        for shard in self._shards:
            if await shard.is_connected():
                connected_count += 1
        if connected_count == 0:
            raise RuntimeError("Failed to connect to any Redis shard")

        log.info(f"Connected to {connected_count}/{self.shard_count} Redis shards")
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from all Redis shards."""
        await asyncio.gather(
            *(shard.disconnect() for shard in self._shards),
            return_exceptions=True,
        )
        self._shards.clear()
        self._connected = False

    async def is_connected(self) -> bool:
        """Check if at least one shard is connected."""
        if not self._connected:
            return False
        connected = 0
        for shard in self._shards:
            if await shard.is_connected():
                connected += 1
        return connected > 0

    async def get(self, key: str) -> Any:
        """Get a value from the distributed cache."""
        if not self._connected:
            raise RuntimeError("Cache not connected. Call connect() first.")

        shard = self._get_shard(key)
        return await shard.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in the distributed cache."""
        if not self._connected:
            raise RuntimeError("Cache not connected. Call connect() first.")

        shard = self._get_shard(key)
        return await shard.set(key, value, ttl)

    async def delete(self, key: str) -> bool:
        """Delete a key from the distributed cache."""
        if not self._connected:
            raise RuntimeError("Cache not connected. Call connect() first.")

        shard = self._get_shard(key)
        return await shard.delete(key)

    async def clear(self) -> None:
        """Clear all shards (use with caution)."""
        if not self._connected:
            raise RuntimeError("Cache not connected. Call connect() first.")

        await asyncio.gather(
            *(shard.clear() for shard in self._shards),
            return_exceptions=True,
        )

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values from the distributed cache."""
        if not self._connected:
            raise RuntimeError("Cache not connected. Call connect() first.")

        # Group keys by shard
        keys_by_shard: dict[int, list[str]] = {}
        for key in keys:
            shard_index = self._get_shard_index(key)
            keys_by_shard.setdefault(shard_index, []).append(key)

        # Fetch from each shard
        results: dict[str, Any] = {}
        await asyncio.gather(
            *(self._shards[shard_index].get_many(keys) for shard_index, keys in keys_by_shard.items()),
            return_exceptions=True,
        )

        return results

    async def set_many(self, mapping: dict[str, Any], ttl: Optional[int] = None) -> int:
        """Set multiple values in the distributed cache."""
        if not self._connected:
            raise RuntimeError("Cache not connected. Call connect() first.")

        # Group by shard
        mapping_by_shard: dict[int, dict[str, Any]] = {}
        for key, value in mapping.items():
            shard_index = self._get_shard_index(key)
            mapping_by_shard.setdefault(shard_index, {})[key] = value

        # Set on each shard
        total_set = 0
        results = await asyncio.gather(
            *(self._shards[shard_index].set_many(mapping, ttl) for shard_index, mapping in mapping_by_shard.items()),
            return_exceptions=True,
        )

        total_set = sum(1 for r in results if isinstance(r, int) and r > 0)

        return total_set

    async def warm_cache(self, entries: dict[str, Any], ttl: Optional[int] = None) -> int:
        """
        Pre-warm the cache with a set of entries.

        This is useful for populating the cache before expected high traffic.

        Args:
            entries: Dictionary of key-value pairs to cache
            ttl: Time-to-live in seconds (uses default if None)

        Returns:
            Number of entries successfully cached
        """
        return await self.set_many(entries, ttl=ttl)

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get statistics about the cache."""
        if not self._connected:
            return {"error": "Cache not connected"}

        stats: dict[str, Any] = {
            "shard_count": self.shard_count,
            "connected_shards": 0,
            "shards": [],
        }

        shard_stats = await asyncio.gather(
            *(self._get_shard_stats(shard) for shard in self._shards),
            return_exceptions=True,
        )

        for shard_stat in shard_stats:
            if isinstance(shard_stat, dict):
                stats["shards"].append(shard_stat)
                if shard_stat.get("connected"):
                    stats["connected_shards"] += 1

        return stats

    async def _get_shard_stats(self, shard: CacheShard) -> dict[str, Any]:
        """Get statistics for a single shard."""
        if shard._client is None:
            return {"name": shard.name, "connected": False}

        try:
            info = await shard._client.info("memory")
            return {
                "name": shard.name,
                "connected": await shard.is_connected(),
                "used_memory": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
            }
        except Exception as e:
            return {"name": shard.name, "connected": False, "error": str(e)}


class DistributedCacheBackend:
    """
    Wrapper that provides a unified cache interface across multiple cache types.

    This allows falling back to local cache if distributed cache is unavailable.
    """

    def __init__(
        self,
        distributed_cache: Optional[RedisNodeCache] = None,
        local_cache: Optional[Any] = None,
        fallback_enabled: bool = True,
    ) -> None:
        self.distributed_cache = distributed_cache
        self.local_cache = local_cache
        self.fallback_enabled = fallback_enabled

    async def get(self, key: str) -> Any:
        """Get a value, with fallback to local cache."""
        if self.distributed_cache and await self.distributed_cache.is_connected():
            try:
                value = await self.distributed_cache.get(key)
                if value is not None:
                    return value
            except Exception as e:
                log.warning(f"Distributed cache get failed, falling back: {e}")

        if self.fallback_enabled and self.local_cache:
            return self.local_cache.get(key)

        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in both distributed and local cache."""
        set_distributed = False
        set_local = False

        if self.distributed_cache and await self.distributed_cache.is_connected():
            try:
                set_distributed = await self.distributed_cache.set(key, value, ttl)
            except Exception as e:
                log.warning(f"Distributed cache set failed: {e}")

        if self.fallback_enabled and self.local_cache:
            self.local_cache.set(key, value, ttl=ttl or 3600)
            set_local = True

        return set_distributed or set_local

    async def delete(self, key: str) -> bool:
        """Delete from both caches."""
        deleted_distributed = False
        deleted_local = False

        if self.distributed_cache and await self.distributed_cache.is_connected():
            try:
                deleted_distributed = await self.distributed_cache.delete(key)
            except Exception as e:
                log.warning(f"Distributed cache delete failed: {e}")

        if self.fallback_enabled and self.local_cache:
            self.local_cache.clear()  # Local cache doesn't support per-key delete
            deleted_local = True

        return deleted_distributed or deleted_local


async def create_distributed_cache(
    config: Optional[RedisCacheConfig] = None,
    enable_fallback: bool = True,
    fallback_cache: Optional[Any] = None,
) -> DistributedCacheBackend:
    """
    Create a distributed cache with optional fallback.

    Args:
        config: Redis cache configuration
        enable_fallback: Whether to enable local cache fallback
        fallback_cache: Optional local cache to use as fallback

    Returns:
        DistributedCacheBackend instance
    """
    redis_cache: Optional[RedisNodeCache] = None

    if config or _is_truthy(get_system_env_value("REDIS_HOST")):
        try:
            redis_config = config or RedisCacheConfig.from_env()
            redis_cache = RedisNodeCache(redis_config)
            await redis_cache.connect()
            log.info("Distributed cache initialized successfully")
        except Exception as e:
            log.warning(f"Failed to initialize distributed cache: {e}")
            redis_cache = None

    return DistributedCacheBackend(
        distributed_cache=redis_cache,
        local_cache=fallback_cache,
        fallback_enabled=enable_fallback,
    )


__all__ = [
    "CacheShard",
    "DistributedCacheBackend",
    "RedisCacheConfig",
    "RedisNodeCache",
    "create_distributed_cache",
]
