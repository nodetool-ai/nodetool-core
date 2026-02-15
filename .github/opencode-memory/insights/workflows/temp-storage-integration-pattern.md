# Temp Storage Integration Pattern for Workflow Outputs

**Insight**: Workflow outputs can exceed reasonable database sizes (1MB+), causing event log bloat. The solution is to integrate with existing temp storage infrastructure while maintaining backward compatibility.

**Rationale**:
- **Problem**: Large JSON outputs (>1MB) and streaming outputs (1000+ chunks) bloat the event log database
- **Solution**: Store large outputs externally in temp storage, log only reference IDs
- **Key**: Maintain async-first patterns and use existing storage infrastructure

**Pattern**:

```python
# 1. Async storage function
async def store_large_output_in_temp_storage(value: Any, storage: Any) -> str | None:
    """Store large output in temp storage."""
    try:
        serialized = json.dumps(value)
        data = serialized.encode("utf-8")
        storage_id = f"output_{uuid.uuid4().hex}"
        data_stream = io.BytesIO(data)
        await storage.upload(storage_id, data_stream)
        return storage_id
    except Exception as e:
        log.error(f"Failed to store: {e}")
        return None

# 2. Async retrieval function
async def retrieve_output_from_temp_storage(storage_id: str, storage: Any) -> Any | None:
    """Retrieve output from temp storage."""
    try:
        data_stream = io.BytesIO()
        await storage.download(storage_id, data_stream)
        data_stream.seek(0)
        data = data_stream.read()
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        log.error(f"Failed to retrieve: {e}")
        return None

# 3. Integration with existing functions (optional storage parameter)
def serialize_output_for_event_log(
    value: Any,
    max_size: int = MAX_INLINE_SIZE,
    storage: Any = None,  # Optional: enables temp storage
) -> dict:
    # ... existing logic ...
    if size_bytes > max_size and storage is not None:
        storage_id = f"output_{uuid.uuid4().hex}"
        return {"type": "external_ref", "storage_id": storage_id, "size_bytes": size_bytes}
```

**Implementation Guidelines**:

1. **Keep it optional**: Add storage as optional parameter to maintain backward compatibility
2. **Return storage ID**: Event log stores only the reference ID, not the full data
3. **Use unique IDs**: Prefix IDs with type (e.g., `output_`, `streaming_`) for clarity
4. **Handle errors gracefully**: Return `None` on failure, log errors for debugging
5. **Document async pattern**: Clearly mark async functions with `async def`
6. **Test roundtrips**: Verify store → retrieve cycle preserves data integrity

**Testing Pattern**:

```python
@pytest.mark.asyncio
async def test_store_and_retrieve_roundtrip():
    """Test roundtrip of storing and retrieving large output."""
    storage = MemoryStorage(base_url="temp://")
    original = {"data": "large" * 1_000_000}

    # Store
    storage_id = await store_large_output_in_temp_storage(original, storage)
    assert storage_id is not None

    # Retrieve
    retrieved = await retrieve_output_from_temp_storage(storage_id, storage)
    assert retrieved == original
```

**Impact**: Reduces database size by orders of magnitude for large outputs while maintaining full recoverability through temp storage. Streaming nodes no longer create write contention from thousands of individual chunk entries.

**Files**:
- `src/nodetool/workflows/output_serialization.py`
- `tests/workflows/test_output_serialization.py`

**Date**: 2026-02-14
