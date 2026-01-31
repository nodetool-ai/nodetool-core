# Checkpoint Manager - Zero-Overhead Resume Feature

## Overview

The `CheckpointManager` provides an efficient, opt-in checkpoint system for resumable workflows with **zero runtime overhead** when disabled (default).

## Key Features

- ✅ **Zero Overhead by Default**: No database writes or state tracking during normal execution
- ✅ **Explicit Checkpoints**: State is only saved when explicitly requested
- ✅ **Clean Separation**: No modifications to runner or actor code required
- ✅ **Leverages Existing Tables**: Uses `Job` and `RunNodeState` tables
- ✅ **Optional Integration**: Hook-based design for clean integration

## Design Principles

1. **Opt-In**: Checkpointing is disabled by default (`enabled=False`)
2. **Zero Runtime Cost**: No DB writes, no state tracking during execution
3. **Explicit Saves**: Checkpoints only saved when explicitly requested
4. **Clean Separation**: No modifications to runner or actor code
5. **Leverage Existing**: Uses existing Job and RunNodeState tables

## Usage

### Basic Usage

```python
from nodetool.workflows.checkpoint_manager import CheckpointManager
from nodetool.workflows.graph import Graph

# Create checkpoint manager (disabled by default for zero overhead)
checkpoint_mgr = CheckpointManager(run_id="job-123", enabled=False)

# Enable for this workflow
checkpoint_mgr = CheckpointManager(run_id="job-123", enabled=True)

# Save checkpoint at strategic points
await checkpoint_mgr.save_checkpoint(
    graph=graph,
    completed_nodes={"node1", "node2"},
    active_nodes={"node3"},
    pending_nodes={"node4", "node5"},
)

# Check if workflow can be resumed
if await checkpoint_mgr.can_resume():
    # Restore from checkpoint
    checkpoint_data = await checkpoint_mgr.restore_checkpoint(graph)
    
    if checkpoint_data:
        print(f"Restored: {len(checkpoint_data.completed_nodes)} completed nodes")
        print(f"Active: {len(checkpoint_data.active_nodes)} nodes")
        print(f"Pending: {len(checkpoint_data.pending_nodes)} nodes")
```

### Integration with Workflow Runner

The checkpoint manager provides hook functions for optional integration without modifying core runner code:

```python
from nodetool.workflows.checkpoint_manager import (
    CheckpointManager,
    create_checkpoint_hook,
    restore_checkpoint_hook,
)

# Initialize checkpoint manager (optional)
checkpoint_mgr = CheckpointManager(run_id=job_id, enabled=enable_checkpoints)

# Before workflow execution - check for resumption
restored = await restore_checkpoint_hook(checkpoint_mgr, graph)
if restored:
    # Skip completed nodes, resume from active/pending
    pass

# During workflow execution - save checkpoints at strategic points
await create_checkpoint_hook(
    checkpoint_mgr=checkpoint_mgr,
    graph=graph,
    completed_nodes=completed,
    active_nodes=active,
    pending_nodes=pending,
)
```

### Strategic Checkpoint Placement

Checkpoints should be saved at meaningful boundaries:

1. **After batch completion**: After processing a batch of nodes
2. **Before expensive operations**: Before starting compute-intensive tasks
3. **After major milestones**: After completing significant workflow stages
4. **On user request**: When explicitly requested via API

```python
# Example: Checkpoint after every 10 nodes completed
if len(completed_nodes) % 10 == 0:
    await checkpoint_mgr.save_checkpoint(
        graph=graph,
        completed_nodes=completed_nodes,
        active_nodes=active_nodes,
        pending_nodes=pending_nodes,
    )
```

## API Reference

### CheckpointManager

```python
class CheckpointManager:
    def __init__(self, run_id: str, enabled: bool = False):
        """
        Initialize checkpoint manager.
        
        Args:
            run_id: Workflow run identifier
            enabled: Whether checkpointing is enabled (default: False)
        """
```

#### Methods

**`save_checkpoint()`**
```python
async def save_checkpoint(
    self,
    graph: Graph,
    completed_nodes: set[str] | None = None,
    active_nodes: set[str] | None = None,
    pending_nodes: set[str] | None = None,
    context_data: dict[str, Any] | None = None,
) -> bool:
    """
    Explicitly save a checkpoint of current workflow state.
    
    This is the ONLY method that writes to the database.
    
    Returns:
        True if checkpoint saved successfully, False otherwise
    """
```

**`restore_checkpoint()`**
```python
async def restore_checkpoint(self, graph: Graph) -> CheckpointData | None:
    """
    Restore workflow state from the last checkpoint.
    
    Returns:
        CheckpointData if restoration successful, None otherwise
    """
```

**`can_resume()`**
```python
async def can_resume(self) -> bool:
    """
    Check if this workflow can be resumed from a checkpoint.
    
    Returns:
        True if the workflow has a valid checkpoint and can be resumed
    """
```

**`clear_checkpoint()`**
```python
async def clear_checkpoint(self) -> bool:
    """
    Clear all checkpoint data for this workflow.
    
    Removes all node state records but keeps the job record.
    
    Returns:
        True if cleared successfully, False otherwise
    """
```

**`get_stats()`**
```python
def get_stats(self) -> dict[str, Any]:
    """
    Get checkpoint statistics.
    
    Returns:
        Dictionary with checkpoint stats (run_id, enabled, checkpoint_count)
    """
```

### CheckpointData

```python
class CheckpointData:
    """Represents a complete checkpoint of workflow state."""
    
    run_id: str
    checkpoint_time: datetime
    node_states: dict[str, NodeStateSnapshot]
    completed_nodes: set[str]
    active_nodes: set[str]
    pending_nodes: set[str]
    context_data: dict[str, Any]
```

### NodeStateSnapshot

```python
class NodeStateSnapshot:
    """Lightweight snapshot of a node's execution state."""
    
    node_id: str
    status: NodeStatus
    attempt: int
    outputs: dict[str, Any] | None
    error: str | None
    resume_state: dict[str, Any] | None
```

## Performance Characteristics

### When Disabled (Default)
- **Overhead**: Effectively zero (single boolean check)
- **Database Writes**: None
- **Memory Usage**: Minimal (manager object only)
- **Latency Impact**: < 1 microsecond per operation

### When Enabled
- **Checkpoint Save Time**: 1-5ms per checkpoint (depends on node count)
- **Restore Time**: 2-10ms (depends on state size)
- **Database Writes**: Only during explicit `save_checkpoint()` calls
- **Memory Usage**: O(n) where n is number of nodes (checkpoint data)

## Comparison with Previous Implementation

### Previous Implementation (StateManager + EventLogger)
- ❌ Automatic state tracking during execution (overhead on every node)
- ❌ Continuous database writes
- ❌ Complex integration with runner code
- ❌ Always-on, cannot be disabled

### New Implementation (CheckpointManager)
- ✅ Zero overhead when disabled (default)
- ✅ Explicit checkpoints only when requested
- ✅ Clean separation from runner/actor
- ✅ Opt-in via flag

## Testing

Comprehensive test suite with 29 tests covering:

- **Disabled Mode** (11 tests): Validates zero-overhead behavior
- **Save/Restore** (8 tests): Tests checkpoint persistence and recovery
- **Resume Capability** (5 tests): Tests resume detection logic
- **Hook Integration** (3 tests): Tests optional hook functions
- **Integration** (2 tests): End-to-end scenarios

Run tests:
```bash
uv run pytest tests/workflows/test_checkpoint_manager.py -v
```

## Migration Guide

### From StateManager to CheckpointManager

The CheckpointManager does NOT replace StateManager. They serve different purposes:

- **StateManager**: Queue-based state updates during active workflow execution
- **CheckpointManager**: Explicit checkpoint save/restore for resumption

If you were using automatic state tracking for resumption, migrate to explicit checkpoints:

```python
# Old approach (automatic, overhead during execution)
state_manager = StateManager(run_id)
await state_manager.start()
# ... execution happens with automatic tracking
await state_manager.stop()

# New approach (explicit, zero overhead until checkpoint)
checkpoint_mgr = CheckpointManager(run_id, enabled=True)
# ... execution happens with zero overhead
# Save checkpoint at strategic point
await checkpoint_mgr.save_checkpoint(graph, completed, active, pending)
```

## Future Enhancements

Possible future improvements:

1. **Automatic Checkpoint Intervals**: Optional time-based or node-count-based automatic checkpointing
2. **Incremental Checkpoints**: Save only changed state since last checkpoint
3. **Compression**: Compress checkpoint data for large workflows
4. **Retention Policies**: Automatically clean up old checkpoints
5. **Multi-Level Checkpoints**: Support for full and incremental checkpoints

## Related Modules

- `recovery.py`: Workflow recovery service (complements checkpoint manager)
- `state_manager.py`: Queue-based state updates during execution
- `suspendable_node.py`: Support for node-level suspension
- `event_logger.py`: Async event logging for audit trail

## Support

For questions or issues:

1. Check the test suite for usage examples
2. Review the docstrings in `checkpoint_manager.py`
3. Look at existing recovery and state management code
4. File an issue on GitHub
