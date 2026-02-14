# Storage Abstraction Test Patterns

**Insight**: Created mock-based tests for storage abstractions to ensure interface compliance across implementations.

**Rationale**: Abstract base classes (`AbstractStorage`, `AbstractNodeCache`) define the contract for storage backends. Testing these contracts with mock implementations:
1. Verifies all required methods are implemented
2. Tests interface behavior without real resources (S3, filesystem)
3. Provides examples for future storage implementations

**Example**:
```python
class MockStorage(AbstractStorage):
    """Mock implementation of AbstractStorage for testing."""
    async def get_url(self, key: str) -> str:
        return f"https://example.com/{key}"
    # ... implement other methods
```

**Impact**:
- Added 31 tests covering storage and caching abstractions
- Tests verify upload/download, metadata retrieval, and TTL functionality
- Mock implementations allow fast, isolated testing without external dependencies

**Files**:
- tests/storage/test_abstract_storage.py
- tests/storage/test_abstract_node_cache.py
- src/nodetool/storage/abstract_storage.py
- src/nodetool/storage/abstract_node_cache.py

**Date**: 2026-02-14
