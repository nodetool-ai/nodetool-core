# State Manager Implementation

**Status**: Not Started  
**Priority**: High  
**Objective**: Eliminate SQLite write contention during parallel node execution

---

## Problem Statement

### Current Issue

When running workflows with parallel node execution, SQLite deadlocks occur because:

1. **Parallel Node Actors**: All node actors run concurrently via `asyncio.gather()`
2. **Multiple DB Writes**: Each actor makes ~4 writes to `run_node_state`:
   - `get_or_create()` at start (SELECT + INSERT)
   - `mark_scheduled()` (UPDATE)
   - `mark_running()` (UPDATE)
   - `mark_completed()` or `mark_failed()` (UPDATE)

3. **SQLite Serialization**: While WAL mode allows concurrent reads, **writes are serialized**:
   ```
   Node A: BEGIN; UPDATE run_node_state SET status='scheduled' WHERE ...;
   Node B: BEGIN; UPDATE run_node_state SET status='running' WHERE ...;  <-- BLOCKED
   Node C: BEGIN; INSERT INTO run_node_state ...;                          <-- BLOCKED
   ```

4. **Result**: "database locked" errors, timeouts, and deadlocks

### Impact

- Workflow failures due to DB contention
- Poor performance under parallel execution
- Unpredictable behavior with no clear retry strategy

---

## Solution: Queue-Based Single Writer Pattern

### Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Node Actor 1   │     │              │     │                 │
│  Node Actor 2   │────>│ Async Queue  │────>│ Batch Writer    │
│  Node Actor N   │     │              │     │ (Single Task)   │
└─────────────────┘     └──────────────┘     │                 │
                                             │  Groups Updates │
                                             │  Single Tx      │
                                             └────────┬────────┘
                                                      │
                                                      ▼
                                             ┌─────────────────┐
                                             │ run_node_state  │
                                             │ SQLite Table    │
                                             └─────────────────┘
```

### Key Design Principles

1. **Single Writer**: Only one async task performs DB writes
2. **Non-Blocking API**: Actors push to queue and continue immediately
3. **Batched Transactions**: Updates grouped and committed together
4. **Graceful Degradation**: Workflow continues if state tracking fails
5. **Backward Compatible**: Fallback to direct writes if needed

---

## Implementation Files

### New File: `src/nodetool/workflows/state_manager.py`

This is the core implementation. See detailed specification below.

### Modified Files:

| File | Changes |
|------|---------|
| `workflow_runner.py` | Add StateManager instance, integrate lifecycle |
| `actor.py` | Use StateManager for state updates (with fallback) |

---

## File 1: `src/nodetool/workflows/state_manager.py`

### Class Structure

```python
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class NodeStateUpdate:
    """
    Represents a single node state update.
    
    Attributes:
        run_id: Workflow run identifier
        node_id: Node being updated
        operation: Type of update (scheduled|running|completed|failed|suspended)
        data: Additional update data
        timestamp: Ordering within batch (monotonic)
    """
    run_id: str
    node_id: str
    operation: str
    data: dict[str, Any]
    timestamp: float


class StateManager:
    """
    Queue-based batch writer for node state updates.
    
    Accumulates updates from multiple node actors and flushes them
    in batches within single database transactions.
    
    Usage:
        # In WorkflowRunner:
        await state_manager.start()
        try:
            # ... run workflow ...
        finally:
            await state_manager.stop()
        
        # In NodeActor:
        await runner.state_manager.update_node_state(
            node_id=node._id,
            operation="scheduled",
            attempt=1,
        )
    """
    
    def __init__(
        self,
        run_id: str,
        batch_size: int = 50,
        flush_interval: float = 0.1,
        max_queue_size: int = 10000,
    ):
        """
        Initialize StateManager.
        
        Args:
            run_id: Workflow run identifier
            batch_size: Max updates per batch before flushing
            flush_interval: Max seconds between flushes
            max_queue_size: Max queue depth (0 = unlimited)
        """
        self.run_id = run_id
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_queue_size = max_queue_size
        
        self._queue: asyncio.Queue[NodeStateUpdate] = asyncio.Queue(maxsize=max_queue_size)
        self._writer_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()
        self._started = False
        
        # Per-batch cache to reduce DB loads
        self._state_cache: dict[str, Any] = {}
    
    @property
    def is_running(self) -> bool:
        """Check if batch writer is active."""
        return self._started and self._writer_task is not None and not self._writer_task.done()
    
    async def start(self) -> None:
        """
        Start the batch writer task.
        
        Idempotent: multiple calls only start one writer.
        """
        if self._started:
            return
        
        self._shutdown_event.clear()
        self._writer_task = asyncio.create_task(self._batch_writer())
        self._started = True
        log.info(f"StateManager started for run {self.run_id}")
    
    async def stop(self, timeout: float = 10.0) -> None:
        """
        Stop the batch writer and flush remaining updates.
        
        Args:
            timeout: Seconds to wait for graceful shutdown
        """
        if not self._started:
            return
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Send sentinel to wake writer
        await self._queue.put(NodeStateUpdate(
            run_id=self.run_id,
            node_id="",
            operation="_shutdown_",
            data={},
            timestamp=0.0,
        ))
        
        # Wait for writer to finish
        if self._writer_task:
            try:
                await asyncio.wait_for(self._writer_task, timeout=timeout)
            except asyncio.TimeoutError:
                log.warning(f"StateManager shutdown timed out after {timeout}s, cancelling")
                self._writer_task.cancel()
                try:
                    await self._writer_task
                except asyncio.CancelledError:
                    pass
            self._writer_task = None
        
        self._started = False
        log.info(f"StateManager stopped for run {self.run_id}")
    
    async def update_node_state(
        self,
        node_id: str,
        operation: str,
        **data,
    ) -> bool:
        """
        Queue a node state update.
        
        Non-blocking: pushes to queue and returns immediately.
        
        Args:
            node_id: Node identifier
            operation: Type of update (scheduled|running|completed|failed|suspended)
            **data: Additional update fields
            
        Returns:
            True if queued successfully, False if queue is full
        """
        if not self.is_running:
            log.warning(f"StateManager not running, update dropped: {node_id}/{operation}")
            return False
        
        try:
            update = NodeStateUpdate(
                run_id=self.run_id,
                node_id=node_id,
                operation=operation,
                data=data,
                timestamp=asyncio.get_event_loop().time(),
            )
            self._queue.put_nowait(update)
            return True
        except asyncio.QueueFull:
            log.warning(f"StateManager queue full, update dropped: {node_id}/{operation}")
            return False
    
    async def _batch_writer(self) -> None:
        """
        Main batch processing loop.
        
        Accumulates updates and flushes when:
        - Batch size reached
        - Flush interval elapsed
        - Shutdown requested
        
        Never crashes: catches all exceptions and continues.
        """
        batch: list[NodeStateUpdate] = []
        last_flush_time = asyncio.get_event_loop().time()
        flush_pending = False
        
        while True:
            try:
                # Wait for next update or timeout
                timeout = self.flush_interval if not batch else 0.05
                update = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=timeout,
                )
                
                # Check for shutdown sentinel
                if update.operation == "_shutdown_":
                    break
                
                batch.append(update)
                flush_pending = True
                
                # Flush if batch size reached
                if len(batch) >= self.batch_size:
                    await self._flush_batch(batch)
                    batch = []
                    last_flush_time = asyncio.get_event_loop().time()
                    flush_pending = False
                    
            except asyncio.TimeoutError:
                # Flush if interval elapsed
                elapsed = asyncio.get_event_loop().time() - last_flush_time
                if batch and elapsed >= self.flush_interval:
                    await self._flush_batch(batch)
                    batch = []
                    last_flush_time = asyncio.get_event_loop().time()
                    flush_pending = False
        
        # Final flush on shutdown
        if batch:
            await self._flush_batch(batch)
    
    async def _flush_batch(self, batch: list[NodeStateUpdate]) -> None:
        """
        Execute a batch of updates in a single transaction.
        
        Args:
            batch: List of updates to execute
        """
        if not batch:
            return
        
        log.debug(f"Flushing {len(batch)} state updates for run {self.run_id}")
        
        # Group by operation for efficiency
        by_operation: dict[str, list[NodeStateUpdate]] = {}
        for update in batch:
            by_operation.setdefault(update.operation, []).append(update)
        
        # Clear cache for this batch
        self._state_cache.clear()
        
        try:
            from nodetool.models.run_node_state import RunNodeState
            adapter = await RunNodeState.adapter()
            
            # Begin transaction
            await adapter.connection.execute("BEGIN")
            
            try:
                # Process scheduled updates
                for update in by_operation.get("scheduled", []):
                    await self._apply_scheduled(update, adapter)
                
                # Process running updates
                for update in by_operation.get("running", []):
                    await self._apply_running(update, adapter)
                
                # Process completed updates
                for update in by_operation.get("completed", []):
                    await self._apply_completed(update, adapter)
                
                # Process failed updates
                for update in by_operation.get("failed", []):
                    await self._apply_failed(update, adapter)
                
                # Process suspended updates
                for update in by_operation.get("suspended", []):
                    await self._apply_suspended(update, adapter)
                
                # Commit
                await adapter.connection.commit()
                log.debug(f"Successfully flushed {len(batch)} updates")
                
            except Exception as e:
                await adapter.connection.rollback()
                log.error(f"Batch flush failed, rolled back: {e}", exc_info=True)
                # Don't re-raise - workflow continues
                
        except ImportError as e:
            log.error(f"Failed to import RunNodeState: {e}")
        except Exception as e:
            log.error(f"Unexpected error in batch flush: {e}", exc_info=True)
    
    async def _get_cached_state(self, node_id: str, adapter) -> Any | None:
        """
        Get state from cache or load from DB.
        
        Args:
            node_id: Node identifier
            adapter: Database adapter
            
        Returns:
            State dict or None if not found
        """
        if node_id in self._state_cache:
            return self._state_cache[node_id]
        
        from nodetool.models.condition_builder import Field, ConditionBuilder
        
        condition = Field("run_id").equals(self.run_id) & Field("node_id").equals(node_id)
        results, _ = await adapter.query(
            condition=condition,
            limit=1,
            columns=["id", "run_id", "node_id", "status", "attempt", 
                     "scheduled_at", "started_at", "completed_at", "failed_at",
                     "suspended_at", "updated_at", "last_error", "retryable",
                     "suspension_reason", "resume_state_json", "outputs_json"],
        )
        
        state = results[0] if results else None
        if state:
            self._state_cache[node_id] = state
        
        return state
    
    async def _apply_scheduled(self, update: NodeStateUpdate, adapter) -> None:
        """Apply scheduled update."""
        from nodetool.models.run_node_state import RunNodeState
        
        existing = await self._get_cached_state(update.node_id, adapter)
        
        if existing:
            # Update existing
            existing["status"] = "scheduled"
            existing["scheduled_at"] = datetime.now().isoformat()
            if "attempt" in update.data:
                existing["attempt"] = update.data["attempt"]
            elif existing.get("started_at"):
                existing["attempt"] = existing.get("attempt", 1) + 1
            existing["updated_at"] = datetime.now().isoformat()
            await adapter.save(existing)
        else:
            # Create new
            state = RunNodeState(
                run_id=self.run_id,
                node_id=update.node_id,
                status="scheduled",
                attempt=update.data.get("attempt", 1),
                scheduled_at=datetime.now(),
            )
            await adapter.save(state.model_dump())
        
        self._state_cache[update.node_id] = existing or state.model_dump()
    
    async def _apply_running(self, update: NodeStateUpdate, adapter) -> None:
        """Apply running update."""
        state = await self._get_cached_state(update.node_id, adapter)
        if state:
            state["status"] = "running"
            state["started_at"] = datetime.now().isoformat()
            state["updated_at"] = datetime.now().isoformat()
            await adapter.save(state)
            self._state_cache[update.node_id] = state
    
    async def _apply_completed(self, update: NodeStateUpdate, adapter) -> None:
        """Apply completed update."""
        state = await self._get_cached_state(update.node_id, adapter)
        if state:
            state["status"] = "completed"
            state["completed_at"] = datetime.now().isoformat()
            state["updated_at"] = datetime.now().isoformat()
            if "outputs_json" in update.data:
                state["outputs_json"] = json.dumps(update.data["outputs_json"])
            await adapter.save(state)
            self._state_cache[update.node_id] = state
    
    async def _apply_failed(self, update: NodeStateUpdate, adapter) -> None:
        """Apply failed update."""
        state = await self._get_cached_state(update.node_id, adapter)
        if state:
            state["status"] = "failed"
            state["failed_at"] = datetime.now().isoformat()
            state["updated_at"] = datetime.now().isoformat()
            if "error" in update.data:
                state["last_error"] = update.data["error"]
            if "retryable" in update.data:
                state["retryable"] = update.data["retryable"]
            await adapter.save(state)
            self._state_cache[update.node_id] = state
    
    async def _apply_suspended(self, update: NodeStateUpdate, adapter) -> None:
        """Apply suspended update."""
        state = await self._get_cached_state(update.node_id, adapter)
        if state:
            state["status"] = "suspended"
            state["suspended_at"] = datetime.now().isoformat()
            state["updated_at"] = datetime.now().isoformat()
            if "reason" in update.data:
                state["suspension_reason"] = update.data["reason"]
            if "resume_state_json" in update.data:
                state["resume_state_json"] = json.dumps(update.data["resume_state_json"])
            await adapter.save(state)
            self._state_cache[update.node_id] = state
```

---

## File 2: `src/nodetool/workflows/workflow_runner.py`

### Required Changes

#### 1. Add Import

```python
from nodetool.workflows.state_manager import StateManager
```

#### 2. Add Instance Variable in `__init__`

```python
def __init__(
    self,
    job_id: str,
    device: str | None = None,
    disable_caching: bool = False,
    buffer_limit: int | None = 3,
    enable_event_logging: bool = True,
):
    # ... existing code ...
    
    # State manager for batching node state updates
    self.state_manager: StateManager | None = None
```

#### 3. Add Helper Methods

```python
async def _start_state_manager(self):
    """Initialize and start the state manager."""
    self.state_manager = StateManager(
        run_id=self.job_id,
        batch_size=50,
        flush_interval=0.1,
    )
    await self.state_manager.start()
    log.debug(f"Started StateManager for job {self.job_id}")

async def _stop_state_manager(self):
    """Flush and stop the state manager."""
    if self.state_manager:
        await self.state_manager.stop()
        self.state_manager = None
        log.debug(f"Stopped StateManager for job {self.job_id}")
```

#### 4. Modify `run()` Method

Add state manager lifecycle:

```python
async def run(
    self,
    request: RunJobRequest,
    context: ProcessingContext,
    send_job_updates: bool = True,
    initialize_graph: bool = True,
    validate_graph: bool = True,
):
    log.info("Starting workflow run: job_id=%s", self.job_id)
    
    # Start state manager
    await self._start_state_manager()
    
    try:
        # ... EXISTING RUN LOGIC ...
        
        # After graph initialization:
        # existing: if initialize_graph: await self._initialize_graph()
        
        # Before execution starts:
        # existing: for node in self._graph.sorted_nodes():
        
    finally:
        # Stop state manager in finally block
        await self._stop_state_manager()
```

**Critical**: The `finally` block ensures state manager stops even if workflow fails:

```python
finally:
    # ... existing cleanup ...
    await self._stop_state_manager()
    await self._cleanup_resources()
```

---

## File 3: `src/nodetool/workflows/actor.py`

### Required Changes

#### Modify `NodeActor.run()` Method

Replace direct state operations with state manager calls. Find and replace:

**REMOVE:**
```python
# Create or get node_state (source of truth)
node_state = None
try:
    node_state = await RunNodeState.get_or_create(
        run_id=self.runner.job_id,
        node_id=node._id,
    )
    await node_state.mark_scheduled(attempt=1)
except Exception as e:
    self.logger.error(f"Failed to create/update node_state: {e}")
```

**REPLACE WITH:**
```python
# Queue state update (non-blocking)
if self.runner.state_manager:
    queued = await self.runner.state_manager.update_node_state(
        node_id=node._id,
        operation="scheduled",
        attempt=1,
    )
    if not queued:
        self.logger.warning(f"Failed to queue scheduled state for {node._id}")
else:
    # Fallback: direct DB write
    try:
        from nodetool.models.run_node_state import RunNodeState
        node_state = await RunNodeState.get_or_create(
            run_id=self.runner.job_id,
            node_id=node._id,
        )
        await node_state.mark_scheduled(attempt=1)
    except Exception as e:
        self.logger.error(f"Failed to update node_state: {e}")
```

**REPLACE:**
```python
# Mark node as running
if node_state:
    try:
        await node_state.mark_running()
```

**WITH:**
```python
# Mark node as running
if self.runner.state_manager:
    await self.runner.state_manager.update_node_state(
        node_id=node._id,
        operation="running",
    )
elif node_state:
    try:
        await node_state.mark_running()
```

**REPLACE:**
```python
# Mark node as completed
if node_state:
    try:
        await node_state.mark_completed(
            outputs_json={},
            duration_ms=duration_ms,
        )
```

**WITH:**
```python
# Mark node as completed
if self.runner.state_manager:
    await self.runner.state_manager.update_node_state(
        node_id=node._id,
        operation="completed",
        outputs_json={},
        duration_ms=duration_ms,
    )
elif node_state:
    try:
        await node_state.mark_completed(
            outputs_json={},
            duration_ms=duration_ms,
        )
```

**REPLACE:**
```python
# Mark node as failed
if node_state:
    try:
        await node_state.mark_failed(
            error=str(e)[:1000],
            retryable=False,
        )
```

**WITH:**
```python
# Mark node as failed
if self.runner.state_manager:
    await self.runner.state_manager.update_node_state(
        node_id=node._id,
        operation="failed",
        error=str(e)[:1000],
        retryable=False,
    )
elif node_state:
    try:
        await node_state.mark_failed(
            error=str(e)[:1000],
            retryable=False,
        )
```

---

## Testing Requirements

### Unit Tests: `tests/workflows/test_state_manager.py`

```python
import pytest
import asyncio
from nodetool.workflows.state_manager import StateManager, NodeStateUpdate


class TestStateManager:
    """Unit tests for StateManager."""
    
    @pytest.fixture
    def state_manager(self):
        """Create StateManager for testing."""
        return StateManager(run_id="test-run", batch_size=10, flush_interval=0.1)
    
    @pytest.mark.asyncio
    async def test_start_stop(self, state_manager):
        """Test start and stop lifecycle."""
        assert not state_manager.is_running
        
        await state_manager.start()
        assert state_manager.is_running
        
        await state_manager.stop(timeout=1.0)
        assert not state_manager.is_running
    
    @pytest.mark.asyncio
    async def test_update_queues(self, state_manager):
        """Test that updates are queued."""
        await state_manager.start()
        
        result = await state_manager.update_node_state(
            node_id="node-1",
            operation="scheduled",
            attempt=1,
        )
        assert result is True
        
        await state_manager.stop()
    
    @pytest.mark.asyncio
    async def test_batch_flush_on_size(self, state_manager):
        """Test that batch flushes when size reached."""
        state_manager.batch_size = 3
        
        await state_manager.start()
        
        # Queue 3 updates
        for i in range(3):
            await state_manager.update_node_state(
                node_id=f"node-{i}",
                operation="scheduled",
            )
        
        # Wait for flush
        await asyncio.sleep(0.2)
        
        await state_manager.stop()
    
    @pytest.mark.asyncio
    async def test_queue_full_handling(self, state_manager):
        """Test that queue full is handled gracefully."""
        state_manager.max_queue_size = 2
        
        await state_manager.start()
        
        # Fill queue
        await state_manager.update_node_state("n1", "scheduled")
        await state_manager.update_node_state("n2", "scheduled")
        
        # This should fail
        result = await state_manager.update_node_state("n3", "scheduled")
        assert result is False
        
        await state_manager.stop()
    
    @pytest.mark.asyncio
    async def test_graceful_error_handling(self, state_manager):
        """Test that errors don't crash state manager."""
        await state_manager.start()
        
        # Queue some updates
        await state_manager.update_node_state("node-1", "scheduled")
        
        # Stop without waiting for flush
        await state_manager.stop(timeout=0.1)
```


### Integration Tests: `tests/workflows/test_state_manager_integration.py`

```python
import pytest
from nodetool.workflows.workflow_runner import WorkflowRunner


class TestStateManagerIntegration:
    """Integration tests for StateManager with WorkflowRunner."""
    
    @pytest.mark.asyncio
    async test_parallel_nodes_no_deadlock(self):
        """Test parallel node execution doesn't cause deadlocks."""
        # Run workflow with multiple parallel nodes
        # Verify all complete successfully
        # Verify no database locked errors in logs
        pass
    
    @pytest.mark.asyncio
    async test_state_persisted_correctly(self):
        """Test that all node states are persisted."""
        # Run workflow
        # Query run_node_state table
        # Verify all nodes have correct final state
        pass
    
    @pytest.mark.asyncio
    async test_workflow_failure_shutdown(self):
        """Test state manager handles workflow failure."""
        # Cause a node to fail
        # Verify state manager stops cleanly
        # Verify failed state is recorded
        pass
```

---

## Configuration

### Environment Variables

Add to `.env.example`:

```bash
# State Manager Configuration
NODETOOL_STATE_BATCH_SIZE=50
NODETOOL_STATE_FLUSH_INTERVAL_MS=100
NODETOOL_STATE_MANAGER_ENABLED=true
```

### Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `NODETOOL_STATE_BATCH_SIZE` | 50 | Max updates per batch |
| `NODETOOL_STATE_FLUSH_INTERVAL_MS` | 100 | Max ms between flushes |
| `NODETOOL_STATE_MANAGER_ENABLED` | true | Enable/disable state manager |

---

## Performance Tuning

### Recommended Settings

| Use Case | batch_size | flush_interval |
|----------|-----------|----------------|
| Low latency | 10 | 50ms |
| Balanced | 50 | 100ms |
| High throughput | 100 | 500ms |
| Critical consistency | 1 | 10ms |

### Expected Improvements

- **Before**: N nodes × 4 writes = 4N DB transactions
- **After**: N nodes × 4 updates → ~4 transactions (batched)

For 100 parallel nodes:
- Before: ~400 transactions
- After: ~4 transactions (100x reduction)

---

## Migration Checklist

- [ ] Create `state_manager.py` with StateManager class
- [ ] Implement NodeStateUpdate dataclass
- [ ] Implement queue-based batch writer with _batch_writer()
- [ ] Add transaction management (BEGIN/COMMIT/ROLLBACK)
- [ ] Implement state cache to reduce DB loads
- [ ] Add error handling (never crash workflow)
- [ ] Add logging at key points
- [ ] Integrate StateManager into WorkflowRunner
- [ ] Update NodeActor to use StateManager
- [ ] Add backward compatibility fallback
- [ ] Add environment variable configuration
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Test with parallel node execution
- [ ] Test error scenarios
- [ ] Test shutdown scenarios
- [ ] Code review
- [ ] Deploy to staging
- [ ] Monitor for issues
- [ ] Deploy to production

---

## Success Criteria

### Must Have

- [ ] No database locked errors during parallel execution
- [ ] All node states persisted correctly (no data loss)
- [ ] Workflow completes at same or faster speed
- [ ] State manager stops cleanly on completion/failure
- [ ] Backward compatible with existing code

### Nice to Have

- [ ] Reduced total DB transactions
- [ ] Configurable batch parameters
- [ ] Metrics for queue depth and flush timing
- [ ] Graceful degradation on DB errors
