# Async Bounded Buffer Implementation

**Insight**: A fixed-size FIFO buffer with configurable overflow strategies provides essential flow control for producer-consumer patterns where memory usage must be bounded.

**Rationale**: While `AsyncChannel` provides unbounded or bounded queues, it lacks explicit overflow handling strategies. The bounded buffer pattern is essential for:
- Preventing unbounded memory growth in high-throughput systems
- Providing backpressure to fast producers when consumers are slow
- Configurable drop policies for real-time systems where stale data is worse than missing data

**Implementation**: `AsyncBoundedBuffer` class in `src/nodetool/concurrency/async_bounded_buffer.py` supports:

1. **Overflow Strategies**:
   - `BLOCK`: Producer blocks until space available (default, provides backpressure)
   - `DROP_OLDEST`: Automatically drops oldest item to make room (for real-time/"latest only" scenarios)
   - `DROP_NEWEST`: Rejects new items when full (signals producer to back off)
   - `RAISE`: Raises `BufferFullError` exception (for explicit error handling)

2. **Core Features**:
   - FIFO ordering guarantees
   - Async iteration support for consumption
   - `BufferStatistics` tracking (puts, gets, drops, overflows)
   - Both blocking (`put`/`get`) and non-blocking (`put_nowait`/`get_nowait`) APIs
   - Graceful shutdown with `close()`

3. **Thread Safety**:
   - Uses `asyncio.Lock` for internal synchronization
   - Uses `asyncio.Condition` for producer/consumer notification
   - Proper handling of `asyncio.CancelledError`

**Example Usage**:

```python
from nodetool.concurrency import AsyncBoundedBuffer, OverflowStrategy

# Real-time sensor data - keep only latest N readings
buffer = AsyncBoundedBuffer[float](
    capacity=100,
    overflow_strategy=OverflowStrategy.DROP_OLDEST
)

async def sensor_producer():
    while True:
        reading = await read_sensor()
        await buffer.put(reading)  # Oldest automatically dropped if full

async def data_processor():
    async for reading in buffer:
        await process(reading)  # Always get latest data
```

**Impact**: Provides a production-ready primitive for building resilient async systems with bounded memory usage and configurable flow control strategies.

**Files**: `src/nodetool/concurrency/async_bounded_buffer.py`, `tests/concurrency/test_async_bounded_buffer.py`

**Date**: 2026-02-20
