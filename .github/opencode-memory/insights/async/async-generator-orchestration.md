# Async Generator Orchestration

**Insight**: Multiple async generators often need to be consumed in coordinated ways beyond simple merging.

**Rationale**: When working with multiple async data sources (APIs, logs, streams), simple merging doesn't provide control over:
- Fair scheduling between fast and slow sources
- Priority-based consumption patterns
- Selective filtering across multiple sources
- Race conditions for fastest-response scenarios

**Example**:
```python
from nodetool.concurrency import AsyncGeneratorOrchestrator

async def high_priority_logs():
    yield "ERROR: critical"
    yield "ERROR: another"

async def low_priority_logs():
    yield "INFO: routine"
    yield "DEBUG: detail"

orchestrator = AsyncGeneratorOrchestrator(high_priority_logs(), low_priority_logs())

# Round-robin: fair interleaving
async for item in orchestrator.round_robin():
    print(item)

# Priority: high priority gets more items per cycle
async for item in orchestrator.priority_round_robin([3, 1]):
    print(item)

# Selective: only consume errors
async for item in orchestrator.selective_consume(
    condition=lambda x, i: "ERROR" in x
):
    print(item)

# Race: fastest responder wins
async for item in orchestrator.race():
    print(item)
```

**Impact**: Enables complex multi-source data ingestion patterns with a single, well-tested utility instead of ad-hoc implementations.

**Files**: 
- `src/nodetool/concurrency/async_generator_orchestrator.py`
- `tests/concurrency/test_async_generator_orchestrator.py`

**Date**: 2026-02-09
