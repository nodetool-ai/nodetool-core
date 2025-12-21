# Migration Guide: Event Sourcing → Mutable State Tables

This guide explains how to migrate from the old event sourcing architecture to the new mutable state tables architecture.

## Overview

**What Changed**: The workflow execution system moved from an **event sourcing** model (where events are the source of truth) to a **mutable state tables** model (where database tables are the source of truth, and events are audit-only).

**Why**: This change improves reliability, simplifies recovery logic, and eliminates event replay complexity.

## Breaking Changes

### 1. Event Log is Audit-Only

**Before**:
```python
# Events were source of truth
events = await RunEvent.get_events(run_id)
projection = await RunProjection.rebuild_from_events(events)
# Recovery decisions based on projection from events
```

**After**:
```python
# State tables are source of truth
run_state = await RunState.get(run_id)
node_states = await RunNodeState.get_all_for_run(run_id)
# Recovery decisions based on state tables (NOT events)
```

**Action Required**:
- Replace all `RunEvent.get_events()` calls used for correctness with `RunState.get()` or `RunNodeState.get()`
- Only use `RunEvent` for audit trails, debugging, and observability
- Do NOT make scheduling or recovery decisions based on events

### 2. RunProjection Deprecated

**Before**:
```python
# RunProjection derived from events
projection = await RunProjection.get(run_id)
incomplete_nodes = projection.get_incomplete_nodes()
```

**After**:
```python
# Use RunState and RunNodeState directly
run_state = await RunState.get(run_id)
incomplete_nodes = await RunNodeState.get_incomplete_nodes(run_id)
```

**Action Required**:
- Replace `RunProjection.get()` with `RunState.get()`
- Replace `projection.get_incomplete_nodes()` with `RunNodeState.get_incomplete_nodes()`
- Update any code that reads from `RunProjection`

### 3. New Database Tables Required

**Before**: Only `run_events`, `run_projections`, and `run_leases` tables

**After**: Four new tables added:
- `run_state` - Run-level authoritative state
- `run_node_state` - Per-node authoritative state
- `run_inbox_messages` - Durable inbox
- `trigger_inputs` - Durable trigger events

**Action Required**:
- Run database migrations before deploying
- Migrations are in `src/nodetool/models/migrations/20251228000000_*.sql`
- Migrations run automatically on startup if using the migration system

### 4. Workflow Recovery Changed

**Before**:
```python
# Recovery rebuilt state from events
recovery = WorkflowRecoveryService()
await recovery.resume_workflow(run_id, graph, context)
# Internally: rebuilt projection from events, then scheduled nodes
```

**After**:
```python
# Recovery reads state tables directly
recovery = WorkflowRecoveryService()
await recovery.resume_workflow(run_id, graph, context)
# Internally: reads run_state and run_node_state, then schedules nodes
```

**Action Required**:
- No API changes, but behavior is different
- Recovery is now deterministic (reads database state, not event replay)
- Faster recovery (no event replay needed)

### 5. Trigger Wakeup Service Changed

**Before**:
```python
# In-memory registry (lost on restart)
wakeup_service = TriggerWakeupService()  # Singleton
wakeup_service.register_suspended_trigger(run_id, node_id)
```

**After**:
```python
# Durable database storage (survives restarts)
wakeup_service = TriggerWakeupService()
await wakeup_service.deliver_trigger_input(
    run_id=run_id,
    node_id=node_id,
    input_id="unique-id",
    payload={"event": "data"}
)
```

**Action Required**:
- Update trigger nodes to use `deliver_trigger_input()` instead of in-memory registration
- Trigger inputs now survive process restarts
- Cross-process coordination now works correctly

### 6. Inbox Messages are Durable

**Before**:
```python
# In-memory inbox (lost on crash)
inbox = NodeInbox(node_id)
await inbox.append(message)  # Lost if process crashes
```

**After**:
```python
# Durable inbox (survives crashes)
inbox = DurableInbox(run_id=run_id, node_id=node_id)
await inbox.append(
    handle="input",
    message_id="unique-id",
    payload={"data": 42}
)
```

**Action Required**:
- Use `DurableInbox` for critical message delivery
- Messages now survive crashes and restarts
- Idempotent delivery (duplicate message_id ignored)

## Migration Steps

### Step 1: Update Dependencies

Ensure you have the latest version with the new models:

```bash
# Pull latest code
git pull origin main

# Install dependencies
pip install -e .
```

### Step 2: Run Database Migrations

The new tables will be created automatically on startup:

```python
from nodetool.models.migrations import run_migrations

# Migrations run automatically, but you can verify:
await run_migrations()
```

Or run manually:

```bash
# Apply migrations
python -c "from nodetool.models.migrations import run_migrations; import asyncio; asyncio.run(run_migrations())"
```

### Step 3: Update Code

**Replace Event-Based Logic**:

```python
# OLD - Don't do this anymore
events = await RunEvent.get_events(run_id)
projection = await RunProjection.rebuild_from_events(events)
if projection.status == "suspended":
    # ...

# NEW - Do this instead
run_state = await RunState.get(run_id)
if run_state.status == "suspended":
    # ...
```

**Replace Projection Queries**:

```python
# OLD
projection = await RunProjection.get(run_id)
incomplete = projection.get_incomplete_nodes()

# NEW
run_state = await RunState.get(run_id)
incomplete = await RunNodeState.get_incomplete_nodes(run_id)
```

**Update Trigger Nodes**:

```python
# OLD - In-memory registration
wakeup_service.register_suspended_trigger(run_id, node_id)

# NEW - Durable storage
await wakeup_service.deliver_trigger_input(
    run_id=run_id,
    node_id=node_id,
    input_id=f"trigger-{unique_id}",
    payload=trigger_data
)
```

### Step 4: Test Your Code

Run tests to ensure everything works:

```bash
# Run workflow tests
pytest tests/workflows/ -v

# Run specific migration tests
pytest tests/workflows/test_state_tables_refactor.py -v
```

### Step 5: Deploy

Deploy the new version with migrations:

```bash
# Deploy with automatic migrations
# Migrations run on startup

# Verify tables created
# Check database for: run_state, run_node_state, run_inbox_messages, trigger_inputs
```

## Backward Compatibility

### Existing Workflows

**Status**: Existing running workflows will need to be restarted.

**Reason**: Old workflows don't have entries in the new state tables.

**Action**: 
- Allow existing workflows to complete naturally
- Or restart them with the new system
- No automatic migration of in-flight workflows

### Event Log

**Status**: Existing event log data is preserved.

**Reason**: Events are still written for audit purposes.

**Action**: No action needed, events continue to be written.

### API Compatibility

**Status**: Most APIs are unchanged.

**Exceptions**:
- `WorkflowRecoveryService.resume_workflow()` - Same API, different internal behavior
- `TriggerWakeupService` - New methods, old in-memory registry removed

**Action**: Update code that uses `TriggerWakeupService` directly.

## FAQ

### Q: Can I still query the event log?

**A**: Yes! The event log still exists and events are still written. You can query it for:
- Audit trails
- Debugging
- Timeline visualization
- Observability

But don't use it for correctness decisions (scheduling, recovery, etc).

### Q: What happens to existing runs?

**A**: Existing runs that are not in the new state tables cannot be recovered. They should be restarted or allowed to complete naturally.

### Q: Do I need to change my node implementations?

**A**: No, node implementations remain the same. The changes are in the workflow runtime system.

### Q: What about performance?

**A**: Performance is improved:
- No event replay during recovery (faster)
- Direct database queries (faster)
- No projection rebuild (faster)

### Q: Can I roll back?

**A**: Yes, but you'll need to:
- Remove the new tables (or keep them empty)
- Use the old code (before the refactor)
- Lose any state stored in the new tables

It's recommended to test thoroughly before deploying to production.

### Q: What if event writes fail?

**A**: Event writes are now non-fatal. If an event write fails:
- A warning is logged
- The workflow continues executing
- State tables are still updated (source of truth)

This is intentional - event log failures don't break workflows.

## Support

If you encounter issues during migration:

1. Check the logs for migration errors
2. Verify database tables were created
3. Check that your code doesn't read from `RunEvent` or `RunProjection` for correctness
4. See `RESUMABLE_WORKFLOWS_DESIGN.md` for architecture details
5. See `IMPLEMENTATION_SUMMARY.md` for usage examples

## Summary

**Key Takeaways**:
- ✅ State tables are now source of truth (not events)
- ✅ Event log is audit-only (failures don't break workflows)
- ✅ New database tables required (migrations provided)
- ✅ Recovery is faster and more deterministic
- ✅ Trigger wakeups are durable (cross-process safe)
- ✅ Inbox messages are durable (crash-safe)
- ⚠️ Existing runs need to be restarted
- ⚠️ Update code that reads `RunEvent` or `RunProjection` for correctness
- ⚠️ Update code that uses `TriggerWakeupService` directly

The new architecture is more reliable, simpler, and production-ready!
