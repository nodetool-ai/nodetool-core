# Suspendable Nodes - Implementation Guide

## Overview

Suspendable nodes are a new feature that allows workflows to pause execution at specific nodes, save their state, and resume later. This enables:

- **Human-in-the-loop operations** - Wait for approvals, feedback, or manual input
- **External callbacks** - Pause for webhook responses or API callbacks
- **Long-running tasks** - Checkpoint and resume expensive computations
- **Scheduled workflows** - Pause and resume at specific times

## Architecture

Suspendable nodes build on the event sourcing infrastructure to provide durable suspension/resumption:

1. **Node suspends** - Calls `suspend_workflow()` with state to save
2. **Events logged** - NodeSuspended and RunSuspended events written to log
3. **Workflow exits** - Runner catches WorkflowSuspendedException and exits cleanly
4. **State persisted** - Suspension state stored in projection via events
5. **External update** - External system can update suspension state
6. **Resume triggered** - Recovery service detects suspended nodes
7. **State restored** - Node gets saved state and continues execution

## Creating a Suspendable Node

```python
from nodetool.workflows.suspendable_node import SuspendableNode
from nodetool.workflows.processing_context import ProcessingContext

class WaitForApproval(SuspendableNode):
    """Node that waits for external approval before continuing."""
    
    request_id: str = ""
    request_data: dict = {}
    
    async def process(self, context: ProcessingContext) -> dict:
        # Check if we're resuming from suspension
        if self.is_resuming():
            saved_state = await self.get_saved_state()
            
            if saved_state.get('approved'):
                return {
                    'status': 'approved',
                    'approved_by': saved_state.get('approved_by'),
                    'approved_at': saved_state.get('approved_at'),
                }
            else:
                return {
                    'status': 'rejected',
                    'reason': saved_state.get('rejection_reason'),
                }
        
        # First execution - suspend and wait for approval
        await self.suspend_workflow(
            reason=f"Waiting for approval of request {self.request_id}",
            state={
                'request_id': self.request_id,
                'request_data': self.request_data,
                'submitted_at': datetime.now().isoformat(),
            },
            metadata={
                'approver_email': 'admin@example.com',
                'timeout_hours': 24,
            }
        )
        
        # Execution never reaches here on first run
        # The suspend_workflow() call raises an exception
```

## API Methods

### SuspendableNode Methods

#### `is_suspendable() -> bool`
Returns True to indicate this node supports suspension.

#### `is_resuming() -> bool`
Check if the node is resuming from a previous suspension.

```python
if self.is_resuming():
    # Resumption path
    saved = await self.get_saved_state()
else:
    # First execution path
    await self.suspend_workflow(...)
```

#### `async get_saved_state() -> dict`
Get the state that was saved when workflow suspended.

```python
saved_state = await self.get_saved_state()
approval_status = saved_state.get('approved', False)
```

Raises `ValueError` if called when not resuming.

#### `async suspend_workflow(reason: str, state: dict, metadata: dict = None)`
Suspend workflow execution and save state.

```python
await self.suspend_workflow(
    reason="Waiting for user input",
    state={'partial_result': computed_value},
    metadata={'timeout': 3600}
)
```

This method:
- Logs NodeSuspended event with state
- Logs RunSuspended event
- Raises WorkflowSuspendedException to exit execution
- Never returns (workflow is suspended)

#### `async update_suspended_state(state_updates: dict, context: ProcessingContext = None)`
Update state while suspended (called externally).

```python
# External API endpoint
node.update_suspended_state({
    'approved': True,
    'approved_by': 'admin@example.com',
    'approved_at': datetime.now().isoformat(),
})
```

## Workflow Suspension Flow

### 1. Initial Execution

```
WorkflowRunner.run()
  ├─> NodeActor executes node
  ├─> node.process() calls suspend_workflow()
  ├─> WorkflowSuspendedException raised
  ├─> Runner catches exception
  ├─> Logs NodeSuspended event (with state)
  ├─> Logs RunSuspended event
  ├─> Updates projection (status = "suspended")
  └─> Exits cleanly
```

### 2. External State Update (Optional)

```
API endpoint /workflows/{run_id}/update-suspended-state
  ├─> Loads suspended node
  ├─> Calls node.update_suspended_state({...})
  ├─> Merges updates into saved state
  └─> State ready for resumption
```

### 3. Workflow Resumption

```
WorkflowRecoveryService.resume_workflow()
  ├─> Loads projection
  ├─> Detects suspended node(s)
  ├─> Logs NodeResumed event (with saved state)
  ├─> Logs RunResumed event
  ├─> Sets node._set_resuming_state()
  ├─> Schedules node for execution
  └─> WorkflowRunner.run() continues

NodeActor executes node (resuming=True)
  ├─> node.is_resuming() returns True
  ├─> node.get_saved_state() returns saved state
  ├─> node.process() continues from saved state
  └─> Workflow completes normally
```

## Event Log

### New Event Types

**NodeSuspended**
```json
{
  "event_type": "NodeSuspended",
  "node_id": "node_123",
  "payload": {
    "reason": "Waiting for approval",
    "state": {"request_id": "req_456", "data": {...}},
    "metadata": {"approver": "admin@example.com"}
  }
}
```

**RunSuspended**
```json
{
  "event_type": "RunSuspended",
  "payload": {
    "node_id": "node_123",
    "reason": "Waiting for approval",
    "metadata": {"approver": "admin@example.com"}
  }
}
```

**NodeResumed**
```json
{
  "event_type": "NodeResumed",
  "node_id": "node_123",
  "payload": {
    "state": {"request_id": "req_456", "approved": true, "data": {...}}
  }
}
```

**RunResumed**
```json
{
  "event_type": "RunResumed",
  "payload": {
    "node_id": "node_123",
    "metadata": {"resumed_by": "system"}
  }
}
```

## Projection State

Suspended nodes are tracked in the projection:

```python
projection = await RunProjection.get(run_id)

# Run-level state
assert projection.status == "suspended"
assert projection.metadata["suspended_node_id"] == "node_123"

# Node-level state
node_state = projection.node_states["node_123"]
assert node_state["status"] == "suspended"
assert node_state["suspension_reason"] == "Waiting for approval"
assert "suspension_state" in node_state  # Saved state
```

## Recovery Integration

The WorkflowRecoveryService automatically handles suspended nodes:

```python
recovery = WorkflowRecoveryService()

# Check if workflow can resume (suspended workflows are resumable)
can_resume = await recovery.can_resume(run_id)

# Resume workflow (handles suspended nodes automatically)
success, message = await recovery.resume_workflow(
    run_id=run_id,
    graph=workflow_graph,
    context=processing_context,
)
```

Resumption plan for suspended nodes:
```python
{
  "node_123": {
    "action": "resume_suspended",
    "reason": "resuming_suspended_node",
    "attempt": 1,
    "saved_state": {"request_id": "req_456", "approved": true}
  }
}
```

## Example Use Cases

### 1. Human Approval

```python
class ApprovalNode(SuspendableNode):
    document_id: str = ""
    
    async def process(self, context: ProcessingContext) -> dict:
        if self.is_resuming():
            state = await self.get_saved_state()
            return {'approved': state['approved'], 'by': state['approved_by']}
        
        # Send notification and suspend
        await send_approval_email(self.document_id)
        await self.suspend_workflow(
            reason="Waiting for document approval",
            state={'document_id': self.document_id},
        )
```

### 2. External API Callback

```python
class WebhookWaitNode(SuspendableNode):
    webhook_url: str = ""
    
    async def process(self, context: ProcessingContext) -> dict:
        if self.is_resuming():
            state = await self.get_saved_state()
            return {'webhook_data': state['callback_data']}
        
        # Register webhook and suspend
        webhook_id = await register_webhook(self.webhook_url)
        await self.suspend_workflow(
            reason=f"Waiting for webhook callback",
            state={'webhook_id': webhook_id},
        )
```

### 3. Scheduled Resumption

```python
class ScheduledPauseNode(SuspendableNode):
    resume_at: str = ""  # ISO datetime
    
    async def process(self, context: ProcessingContext) -> dict:
        if self.is_resuming():
            state = await self.get_saved_state()
            return {'resumed_at': datetime.now().isoformat()}
        
        # Schedule resumption and suspend
        await schedule_resumption(self.resume_at, self._id)
        await self.suspend_workflow(
            reason=f"Scheduled pause until {self.resume_at}",
            state={'scheduled_for': self.resume_at},
        )
```

## Testing

Run suspendable node tests:
```bash
pytest tests/workflows/test_suspendable_nodes.py -v
```

Test coverage:
- Basic suspension/resumption
- State saving and retrieval
- Event logging
- Projection updates
- Recovery integration

## Best Practices

1. **Always check `is_resuming()`** - Handle both first execution and resumption
2. **Save minimal state** - Only save what's needed to resume
3. **Use descriptive reasons** - Make suspension reason clear for humans
4. **Add metadata** - Include context like timeout, approver, etc.
5. **Handle errors** - Consider what happens if resumption fails
6. **Test both paths** - Test both suspension and resumption code paths
7. **Document state** - Clearly document what state fields mean

## Limitations

1. **No nested suspension** - A node can only suspend once per execution
2. **State size** - Keep state reasonable (< 1MB recommended)
3. **Timeout handling** - Manual timeout implementation required
4. **No automatic retry** - Suspension is intentional, not a failure mode

## Future Enhancements

- Automatic timeout and expiration
- Suspension policies (max duration, auto-resume)
- Suspension history tracking
- Multi-stage suspensions (checkpoint sequence)
- Distributed suspension coordination
