# Async Debounce Implementation

**Insight**: Implemented `AsyncDebounce` class and `debounce` decorator for debouncing async function calls in the concurrency module.

**Rationale**: Debouncing is essential for handling bursty async operations in workflows, preventing resource exhaustion from rapid API calls or user inputs. It complements the existing rate limiting utilities by providing call coalescing rather than rate throttling.

**Implementation Details**:
- Uses generation-based tracking to identify stale calls
- Employs asyncio.Lock for thread-safe state management
- Supports both direct `AsyncDebounce` instantiation and decorator usage
- Returns `None` for cancelled/debounced calls, allowing callers to detect suppression

**Example Usage**:
```python
# Direct usage
debounced_save = AsyncDebounce(save_to_database, wait=1.0)
await debounced_save(data)

# Decorator usage
@debounce(wait=0.5)
async def process_item(item: Item) -> None:
    await expensive_operation(item)
```

**Files**: `src/nodetool/concurrency/debounce.py`, `tests/concurrency/test_debounce.py`

**Date**: 2026-01-19
