# Trigger Nodes with Auto-Suspension

## Overview

Trigger nodes have been consolidated with the suspendable node mechanism to provide efficient, scalable trigger-based workflows. Instead of running indefinitely, trigger workflows now suspend after 5 minutes of inactivity and wake up automatically when trigger events arrive.

## Key Benefits

1. **Resource Efficiency** - No indefinite running processes consuming resources
2. **Scalability** - Hundreds of trigger workflows can coexist in suspended state
3. **Durability** - Trigger state persisted via event log
4. **Unified Architecture** - Single mechanism for all suspendable/trigger nodes
5. **Automatic Wake-up** - Trigger events automatically resume workflows

## Architecture

### Traditional Trigger Workflows (Old Approach)
```
┌─────────────────────────────────────────────┐
│  Trigger Workflow                           │
│  ┌──────────────┐                          │
│  │ Start        │                          │
│  └──────┬───────┘                          │
│         │                                   │
│         ▼                                   │
│  ┌──────────────┐                          │
│  │ Trigger Node │◄──────┐                 │
│  │ (Running)    │       │ Wait forever    │
│  └──────┬───────┘       │ for events      │
│         │               │                  │
│         └───────────────┘                  │
│                                             │
│  ⚠️ Problems:                              │
│  - Consumes resources indefinitely         │
│  - Limited scalability                     │
│  - Process crashes = lost state            │
└─────────────────────────────────────────────┘
```

### New Trigger Nodes (With Auto-Suspension)
```
┌──────────────────────────────────────────────────────────┐
│  Trigger Workflow with Auto-Suspension                   │
│  ┌──────────────┐                                        │
│  │ Start        │                                        │
│  └──────┬───────┘                                        │
│         │                                                 │
│         ▼                                                 │
│  ┌──────────────────┐                                    │
│  │ Trigger Node     │                                    │
│  │ Wait for event   │                                    │
│  └──────┬───────────┘                                    │
│         │                                                 │
│         ├──────────┐                                     │
│         │          │                                      │
│         │          │ No event for 5 min                  │
│         │          ▼                                      │
│         │   ┌─────────────────┐                         │
│         │   │ Auto-Suspend    │                         │
│         │   │ (Save state)    │                         │
│         │   └────────┬────────┘                         │
│         │            │                                    │
│         │            ▼                                    │
│         │   ┌─────────────────┐                         │
│         │   │ Workflow        │                         │
│         │   │ Suspended       │◄────────┐               │
│         │   └─────────────────┘         │               │
│         │                                │               │
│         │                         ┌──────┴──────┐       │
│         │                         │ External    │       │
│         │                         │ Trigger     │       │
│         │                         │ Event       │       │
│         │                         └──────┬──────┘       │
│         │                                │               │
│         │                         ┌──────▼──────┐       │
│         │                         │ Wake-up     │       │
│         │                         │ Service     │       │
│         │                         └──────┬──────┘       │
│         │                                │               │
│         │   ┌─────────────────┐         │               │
│         │   │ Resume Workflow │◄────────┘               │
│         │   │ (Load state)    │                         │
│         │   └────────┬────────┘                         │
│         │            │                                    │
│         │            │                                    │
│         │            └────────────────────────┐          │
│         │                                     │          │
│         │ Event arrived                       │          │
│         ▼                                     ▼          │
│  ┌──────────────┐                  ┌─────────────────┐ │
│  │ Process Event│                  │ Continue from   │ │
│  │ Continue     │                  │ Saved State     │ │
│  └──────────────┘                  └─────────────────┘ │
│                                                          │
│  ✅ Benefits:                                           │
│  - Resources released during inactivity                 │
│  - Scales to thousands of triggers                     │
│  - State persisted in event log                        │
│  - Automatic wake-up on trigger events                 │
└──────────────────────────────────────────────────────────┘
```

## Implementation

### Creating a Trigger Node

```python
from nodetool.workflows.trigger_node import TriggerNode, TriggerInactivityTimeout
from nodetool.workflows.processing_context import ProcessingContext

class IntervalTrigger(TriggerNode):
    """Trigger that fires at regular intervals."""
    
    interval_seconds: int = 60
    
    async def process(self, context: ProcessingContext) -> dict:
        # Check if resuming from suspension
        if self.is_resuming():
            return await self.process_trigger_resumption(context)
        
        try:
            # Wait for trigger event with auto-suspension timeout
            event = await self.wait_for_trigger_event(
                timeout_seconds=self.get_inactivity_timeout()  # Default: 300s (5 min)
            )
            
            # Process the event
            return {
                'triggered_at': event['timestamp'],
                'interval_seconds': self.interval_seconds,
                'event_data': event,
            }
            
        except TriggerInactivityTimeout:
            # No events for 5 minutes - suspend workflow
            await self.suspend_for_inactivity({
                'interval_seconds': self.interval_seconds,
                'last_trigger': datetime.now().isoformat(),
            })
```

### Webhook Trigger Example

```python
class WebhookTrigger(TriggerNode):
    """Trigger that waits for webhook calls."""
    
    webhook_url: str = ""
    webhook_id: str = ""
    
    async def process(self, context: ProcessingContext) -> dict:
        if self.is_resuming():
            # Resuming from suspension
            saved_state = await self.get_saved_state()
            log.info(f"Webhook trigger resumed: {saved_state}")
            # Continue processing...
        
        # Register webhook endpoint
        if not self.webhook_id:
            self.webhook_id = await register_webhook(
                self.webhook_url,
                callback=self.on_webhook_received
            )
        
        try:
            # Wait for webhook events
            event = await self.wait_for_trigger_event()
            
            return {
                'webhook_id': self.webhook_id,
                'webhook_data': event['payload'],
                'timestamp': event['timestamp'],
            }
            
        except TriggerInactivityTimeout:
            # Suspend while waiting for webhooks
            await self.suspend_for_inactivity({
                'webhook_id': self.webhook_id,
                'webhook_url': self.webhook_url,
            })
    
    async def on_webhook_received(self, payload: dict):
        """Called by external webhook handler."""
        await self.send_trigger_event({
            'payload': payload,
            'timestamp': datetime.now().isoformat(),
        })
```

### Schedule Trigger Example

```python
class ScheduleTrigger(TriggerNode):
    """Trigger that fires at scheduled times."""
    
    schedule_cron: str = "0 * * * *"  # Every hour
    
    async def process(self, context: ProcessingContext) -> dict:
        if self.is_resuming():
            saved_state = await self.get_saved_state()
            next_run = saved_state.get('next_run_time')
            log.info(f"Schedule resumed, next run: {next_run}")
        
        # Calculate next run time
        next_run = calculate_next_cron_time(self.schedule_cron)
        wait_seconds = (next_run - datetime.now()).total_seconds()
        
        try:
            # Wait until scheduled time (with inactivity timeout)
            await asyncio.sleep(min(wait_seconds, self.get_inactivity_timeout()))
            
            return {
                'scheduled_time': next_run.isoformat(),
                'triggered_at': datetime.now().isoformat(),
            }
            
        except TriggerInactivityTimeout:
            await self.suspend_for_inactivity({
                'next_run_time': next_run.isoformat(),
                'schedule_cron': self.schedule_cron,
            })
```

## Workflow Flow

### 1. Initial Execution

```
1. Workflow starts
2. Trigger node begins waiting for events
3. Node calls wait_for_trigger_event(timeout=300)
4. If event arrives → process and continue
5. If timeout (5 min) → suspend workflow
```

### 2. Suspension

```
1. TriggerInactivityTimeout raised
2. Node calls suspend_for_inactivity()
3. NodeSuspended + RunSuspended events logged
4. TriggerWakeupService registers suspended workflow
5. Workflow exits cleanly (resources released)
```

### 3. External Trigger Event

```
1. External system receives trigger event
2. Calls trigger wakeup API or service
3. TriggerWakeupService.wake_up_trigger_workflow()
4. WorkflowRecoveryService resumes workflow
5. NodeResumed + RunResumed events logged
```

### 4. Resumption

```
1. Workflow runner starts execution
2. Trigger node detects is_resuming() == True
3. Node retrieves saved state via get_saved_state()
4. Node continues processing from saved point
5. Workflow completes or suspends again
```

## API Methods

### TriggerNode Class

#### Core Methods

**`wait_for_trigger_event(timeout_seconds: int = None) -> dict`**
- Waits for trigger event with timeout
- Returns event data when event arrives
- Raises TriggerInactivityTimeout on timeout
- Automatically tracks activity time

**`send_trigger_event(event_data: dict) -> None`**
- Send trigger event to node (called externally)
- Queues event for processing
- Updates activity timestamp

**`suspend_for_inactivity(additional_state: dict = None) -> None`**
- Convenience method to suspend due to inactivity
- Automatically includes trigger metadata
- Calls underlying suspend_workflow()

**`process_trigger_resumption(context) -> dict`**
- Helper for handling resumption
- Returns saved state and resumption info
- Called by subclass process() method

#### Configuration

**`get_inactivity_timeout() -> int`**
- Returns inactivity timeout in seconds (default: 300)

**`set_inactivity_timeout(seconds: int) -> None`**
- Sets custom inactivity timeout
- Must be at least 1 second

#### Status Methods

**`is_trigger_node() -> bool`**
- Returns True (identifies as trigger node)

**`get_last_activity_time() -> datetime`**
- Returns last activity timestamp

**`get_inactivity_duration() -> timedelta`**
- Returns time since last activity

**`should_suspend_for_inactivity() -> bool`**
- Checks if timeout exceeded

### TriggerWakeupService

**Singleton Service for Managing Suspended Triggers**

**`register_suspended_trigger(workflow_id, node_id, metadata)`**
- Registers suspended trigger for wake-up
- Called automatically by WorkflowRunner

**`unregister_suspended_trigger(workflow_id, node_id)`**
- Removes trigger from wake-up registry
- Called on successful resumption

**`wake_up_trigger_workflow(workflow_id, node_id, trigger_event) -> (bool, str)`**
- Wakes up suspended trigger workflow
- Delivers trigger event to workflow
- Returns success status and message

**`list_suspended_triggers() -> dict`**
- Lists all suspended trigger workflows
- Returns mapping of trigger keys to metadata

## Event Log Integration

### New Trigger Events

Trigger suspensions use the existing suspension event types but add trigger-specific metadata:

**NodeSuspended (with trigger metadata)**:
```json
{
  "event_type": "NodeSuspended",
  "node_id": "trigger_node_123",
  "payload": {
    "reason": "Trigger inactivity timeout (300s)",
    "state": {
      "suspended_at": "2025-12-26T10:00:00",
      "interval_seconds": 60,
      "last_trigger_time": "2025-12-26T09:55:00"
    },
    "metadata": {
      "trigger_node": true,
      "inactivity_suspension": true,
      "inactivity_timeout_seconds": 300
    }
  }
}
```

**NodeResumed (after wake-up)**:
```json
{
  "event_type": "NodeResumed",
  "node_id": "trigger_node_123",
  "payload": {
    "state": {
      "suspended_at": "2025-12-26T10:00:00",
      "interval_seconds": 60,
      "woke_up_at": "2025-12-26T10:05:00",
      "trigger_event": {...}
    }
  }
}
```

## Migration from Old Trigger Workflows

### Before (Old Approach)

```python
# Trigger workflow runs indefinitely
class OldIntervalTrigger(BaseNode):
    async def process(self, context):
        while True:
            await asyncio.sleep(self.interval_seconds)
            # Process event
            yield ('output', {'triggered': True})
```

**Problems**:
- Runs forever (resource waste)
- Hard to scale (one process per trigger)
- State loss on crash

### After (New Approach)

```python
from nodetool.workflows.trigger_node import TriggerNode

class NewIntervalTrigger(TriggerNode):
    async def process(self, context):
        if self.is_resuming():
            return await self.process_trigger_resumption(context)
        
        try:
            event = await self.wait_for_trigger_event()
            return {'triggered': True, 'event': event}
        except TriggerInactivityTimeout:
            await self.suspend_for_inactivity({'interval': self.interval_seconds})
```

**Benefits**:
- Auto-suspends after 5 minutes
- Scales to thousands of triggers
- State persisted in event log
- Automatic wake-up

## Configuration

### Inactivity Timeout

**Default**: 300 seconds (5 minutes)

**Per-Node Configuration**:
```python
class MyTrigger(TriggerNode):
    def __init__(self, **data):
        super().__init__(**data)
        self.set_inactivity_timeout(180)  # 3 minutes
```

**Instance Configuration**:
```python
trigger = MyTrigger(id="t1")
trigger.set_inactivity_timeout(600)  # 10 minutes
```

## Best Practices

1. **Set Appropriate Timeouts** - Balance responsiveness vs resource usage
2. **Save Minimal State** - Only save what's needed to resume
3. **Handle Resumption** - Always check `is_resuming()` first
4. **Log Activity** - Use `_update_activity_time()` for custom events
5. **Test Both Paths** - Test suspension and resumption separately
6. **Clean Up Resources** - Release resources before suspension
7. **Document Triggers** - Explain trigger behavior and state

## Testing

### Unit Tests

```bash
pytest tests/workflows/test_trigger_nodes.py -v
```

Coverage:
- Trigger node identification
- Inactivity timeout configuration
- Event sending/receiving
- Auto-suspension on timeout
- Resumption from suspension
- Wake-up service registration

### Integration Tests

Test complete trigger workflow:
1. Start trigger workflow
2. Wait for auto-suspension (5 min)
3. Send trigger event
4. Verify workflow resumes
5. Verify saved state restored
6. Verify workflow completes

## Monitoring

### Suspended Triggers

```python
from nodetool.workflows.trigger_node import TriggerWakeupService

service = TriggerWakeupService.get_instance()
suspended = service.list_suspended_triggers()

for key, info in suspended.items():
    print(f"Workflow: {info['workflow_id']}")
    print(f"Node: {info['node_id']}")
    print(f"Suspended at: {info['suspended_at']}")
    print(f"Metadata: {info['metadata']}")
```

### Projection State

```python
from nodetool.models.run_projection import RunProjection

projection = await RunProjection.get(run_id)
if projection.status == "suspended":
    node_id = projection.metadata.get("suspended_node_id")
    if projection.node_states[node_id].get("suspension_metadata", {}).get("trigger_node"):
        print(f"Trigger node {node_id} suspended due to inactivity")
```

## Future Enhancements

- Dynamic timeout adjustment based on trigger frequency
- Trigger priority levels (wake up high-priority first)
- Trigger event batching (process multiple events together)
- Trigger health monitoring and alerting
- Automatic cleanup of stale suspended triggers
- Trigger event replay on resumption
