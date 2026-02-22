# Async Resource Pool

**Insight**: `AsyncResourcePool` provides a generic pool for managing reusable async resources like HTTP clients, database connections, or other expensive-to-create objects that need to be shared across async tasks.

**Rationale**: When building async systems that interact with external services or resources, creating new connections/clients for each operation is expensive and inefficient. A resource pool allows reusing expensive resources across multiple async tasks while controlling the maximum number of resources and handling cleanup properly.

**Implementation Patterns**:

1. **Hashable Resources**: The pool tracks resources using a set, so resources must be hashable. For wrapper objects, implement `__hash__` and `__eq__` based on the underlying resource identity.

2. **Generic Type Support**: Use `Generic[T]` to type-hint the pooled resource type, allowing the pool to work with any resource type.

3. **Lazy Creation**: Resources are created on-demand when first acquired, not eagerly.

4. **Expiration & Pruning**: Resources can expire based on age (max_age) or idle time (max_idle_time), automatically pruning unused resources.

5. **Graceful Shutdown**: The close() method waits for in-use resources to be returned before closing all resources.

**Example Usage**:
```python
from nodetool.concurrency import AsyncResourcePool

# Create a pool of HTTP clients
async def create_client():
    return aiohttp.ClientSession()

async def close_client(client):
    await client.close()

pool = AsyncResourcePool(
    factory=create_client,
    closer=close_client,
    max_size=10,
    max_idle_time=300.0,
)

# Use with context manager for automatic release
async with pool.acquire_context() as client:
    response = await client.get("https://example.com")
    data = await response.text()

# Or use explicit acquire/release
resource = await pool.acquire()
try:
    await use_resource(resource)
finally:
    await pool.release(resource)
```

**Key Features**:
- **max_size**: Maximum number of resources in the pool
- **min_size**: Minimum number of resources to keep ready (maintained when possible)
- **max_age**: Maximum age for a resource before it's expired
- **max_idle_time**: Maximum idle time before a resource is pruned
- **acquisition_timeout**: Seconds to wait for resource acquisition
- **stats()**: Get pool statistics (size, created, acquired, released, expired, pruned)

**Use Cases**:
- Database connection pooling (SQLite, PostgreSQL)
- HTTP client pooling for API requests
- WebSocket connection pooling
- Any expensive-to-create resource that can be reused

**Impact**: Reduces resource creation overhead, improves performance under load, and provides structured lifecycle management for expensive async resources.

**Files**:
- `src/nodetool/concurrency/async_resource_pool.py`
- `src/nodetool/concurrency/__init__.py`
- `tests/concurrency/test_async_resource_pool.py`

**Date**: 2026-02-21
