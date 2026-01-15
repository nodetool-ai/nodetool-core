# Distributed Cache Layer

**Insight**: Added a Redis-based distributed caching layer extending the existing AbstractNodeCache interface with sharding and failover support.

**Rationale**: The existing caching infrastructure provides memory-based and memcached backends. A Redis-based distributed cache adds:
- Consistent hashing for automatic key sharding across multiple Redis instances
- Automatic failover and reconnection handling
- TTL support with automatic expiration
- Cache warming/preheating for high-traffic scenarios
- Fallback to local cache when distributed cache is unavailable

**Example**:
```python
from nodetool.storage.distributed_cache import (
    RedisCacheConfig,
    RedisNodeCache,
    create_distributed_cache,
)

# Direct configuration
config = RedisCacheConfig(
    host="localhost",
    port=6379,
    shard_count=4,
    default_ttl=3600,
)
cache = RedisNodeCache(config)
await cache.connect()

# Or from environment
cache = await create_distributed_cache()

# Use like any other cache
value = await cache.get("key")
await cache.set("key", value, ttl=1800)
```

**Impact**:
- Horizontal scalability through sharding
- High availability with automatic reconnection
- Consistent performance across distributed deployments
- Seamless fallback to local cache

**Files**: `src/nodetool/storage/distributed_cache.py`

**Date**: 2026-01-15
