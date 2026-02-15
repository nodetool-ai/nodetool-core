# Fan-Out Channel Closing Pattern

**Insight**: When implementing fan-out (distributing items from one channel to multiple output channels), always close output channels in a `finally` block and send to all outputs concurrently.

**Rationale**: The fan-out pattern has two deadlock risks:
1. **Sequential blocking**: Sending to output channels one-by-one can deadlock if one channel is full or slow
2. **Unclosed outputs**: Consumers using `async for` on output channels will hang forever waiting for `StopAsyncIteration` that never comes

**Example**:
```python
# BAD - sequential sends, no cleanup
async def fan_out(channel, *outputs):
    async for item in channel:
        for out in outputs:  # Blocks on each send
            await out.send(item)  # Deadlock if out is full!

# GOOD - concurrent sends, proper cleanup
async def fan_out(channel, *outputs):
    try:
        async for item in channel:
            await asyncio.gather(*[out.send(item) for out in outputs])
    finally:
        for out in outputs:
            out.close()  # Signals consumers to stop
```

**Impact**: Fixed deadlock in `AsyncChannel.fan_out` function, enabling test_fan_out to pass reliably. Pattern applies to any fan-out implementation.

**Files**:
- `src/nodetool/concurrency/async_channel.py`
- `tests/concurrency/test_async_channel.py`

**Date**: 2026-02-14
