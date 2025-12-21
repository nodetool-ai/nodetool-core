# Resumable Workflows Design Document

## Executive Summary

This document describes the implementation of resumable workflows in NodeTool's actor-based execution framework using an append-only event log pattern. The event log serves as the single source of truth for workflow execution state, enabling workflows to resume correctly after crashes, restarts, or interruptions.

## Current Architecture Analysis

### Workflow Execution Model

**Files:**
- `src/nodetool/workflows/workflow_runner.py`: Main orchestrator for workflow execution
- `src/nodetool/workflows/actor.py`: Per-node actor driving individual node execution
- `src/nodetool/workflows/inbox.py`: Message passing between nodes via per-node inboxes
- `src/nodetool/workflows/base_node.py`: Base node class with lifecycle hooks
- `src/nodetool/workflows/processing_context.py`: Execution context and utilities

**Execution Flow:**
1. `WorkflowRunner.run()` initializes the graph and validates nodes
2. One `NodeActor` per node is spawned to drive that node
3. Actors consume inputs from `NodeInbox`, execute the node, and route outputs downstream
4. Nodes have lifecycle: `initialize()` → `pre_process()` → `run()` → `finalize()`
5. End-of-stream (EOS) signals propagate when a node completes

**Message Passing:**
- Each node has a `NodeInbox` with per-handle FIFO buffers
- Messages flow via edges: source node → target node's inbox
- EOS tracking per handle ensures downstream nodes know when upstreams complete
- Backpressure via configurable buffer limits

**Current Persistence:**
- `Job` model: Tracks job metadata (status, error, cost, logs)
- `Workflow` model: Stores workflow definition (graph, nodes, edges)
- No event log or fine-grained execution state tracking
- Status updates via WebSocket messages (ephemeral)

### Actor Messaging Guarantees

**Current Behavior:**
- **Delivery:** At-least-once within a single process run
- **Ordering:** FIFO per edge (messages from A→B maintain order)
- **Retries:** GPU OOM errors trigger retries with exponential backoff (MAX_RETRIES=2)
- **Atomicity:** No atomicity - node execution is not atomic with message sends

**Failure Modes:**
- Process crash → all in-flight state lost
- No recovery mechanism → must restart from beginning
- No deduplication → duplicate processing on retry

### Persistence Boundaries

**Current State:**
1. Job creation: `Job.create()` persists job record
2. Status updates: Job.status updated to "running", "completed", "error"
3. Node updates: Ephemeral `NodeUpdate` messages via WebSocket
4. Output updates: Ephemeral `OutputUpdate` messages, final outputs stored in `Job.result`
5. No intermediate checkpoints or event log

**Critical Gap:** No durable record of:
- Which nodes have started/completed
- Node attempt counts
- Message deliveries
- Dependency satisfaction state
- Trigger consumption cursors

## Event Log Design

### Core Principle

> **The append-only event log is the source of truth for workflow execution. All other state is a projection derived from the log.**

No run-critical decisions may rely solely on in-memory state or mutable DB rows without a corresponding durable event.

### Event Schema

**Table: `run_events`**

```sql
CREATE TABLE IF NOT EXISTS run_events (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    event_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    node_id TEXT,
    payload JSONB NOT NULL,
    
    -- Idempotency and ordering
    UNIQUE(run_id, seq),
    UNIQUE(run_id, node_id, event_type, seq) WHERE node_id IS NOT NULL
);

CREATE INDEX idx_run_events_run_id_seq ON run_events(run_id, seq);
CREATE INDEX idx_run_events_run_id_node_id ON run_events(run_id, node_id);
```

**Fields:**
- `id`: Unique event ID (time-ordered UUID)
- `run_id`: Job/run identifier (FK to jobs.id)
- `seq`: Monotonic sequence number per run (auto-increment)
- `event_type`: Event type string (see below)
- `event_time`: Timestamp when event was recorded
- `node_id`: Optional node identifier for node-specific events
- `payload`: JSON payload with event-specific data

**Idempotency Strategy:**
- Unique constraint on `(run_id, seq)` prevents duplicate sequence numbers
- Event append operations are idempotent: retry of same event with same seq is ignored
- For distributed systems: Use `(run_id, node_id, event_type, seq)` as natural key

### Event Types

#### Run-Level Events

1. **RunCreated**
   - When: Workflow execution starts
   - Payload: `{graph: {...}, params: {...}, user_id: str}`
   - Invariant: First event for a run

2. **RunCancelled**
   - When: Workflow is cancelled (user request or system)
   - Payload: `{reason: str}`
   - Invariant: Terminal event

3. **RunCompleted**
   - When: All nodes complete successfully
   - Payload: `{outputs: {...}, duration_ms: int}`
   - Invariant: Terminal event

4. **RunFailed**
   - When: Workflow fails unrecoverably
   - Payload: `{error: str, node_id: str?}`
   - Invariant: Terminal event

#### Node-Level Events

5. **NodeScheduled**
   - When: Node is ready to execute (dependencies satisfied)
   - Payload: `{node_id: str, node_type: str, attempt: int}`
   - Invariant: Precedes NodeStarted for same attempt

6. **NodeStarted**
   - When: NodeActor begins executing node
   - Payload: `{node_id: str, attempt: int, inputs: {...}}`
   - Invariant: Follows NodeScheduled, precedes NodeCompleted/NodeFailed

7. **NodeCheckpointed**
   - When: Long-running node reaches durable intermediate state
   - Payload: `{node_id: str, attempt: int, checkpoint_data: {...}}`
   - Invariant: Optional, between NodeStarted and NodeCompleted

8. **NodeCompleted**
   - When: Node execution succeeds
   - Payload: `{node_id: str, attempt: int, outputs: {...}, duration_ms: int}`
   - Invariant: Terminal for this attempt

9. **NodeFailed**
   - When: Node execution fails
   - Payload: `{node_id: str, attempt: int, error: str, retryable: bool}`
   - Invariant: May be followed by NodeScheduled (retry) or terminal

#### Trigger Node Events

10. **TriggerRegistered**
    - When: Trigger node initializes and registers for inputs
    - Payload: `{node_id: str, trigger_type: str, config: {...}}`
    - Invariant: Once per trigger node per run

11. **TriggerInputReceived**
    - When: External input arrives for a trigger
    - Payload: `{node_id: str, input_id: str, data: {...}, cursor: str?}`
    - Invariant: Ordered by cursor if applicable

12. **TriggerCursorAdvanced**
    - When: Trigger acknowledges processing of input(s)
    - Payload: `{node_id: str, cursor: str, processed_count: int}`
    - Invariant: Cursor monotonically increases

#### Message Delivery Events (Optional - for at-least-once with dedup)

13. **OutboxEnqueued**
    - When: Node produces output to be sent downstream
    - Payload: `{node_id: str, edge_id: str, message_id: str, data: {...}}`
    - Invariant: Precedes OutboxSent

14. **OutboxSent**
    - When: Output message delivered to target inbox
    - Payload: `{node_id: str, edge_id: str, message_id: str}`
    - Invariant: Follows OutboxEnqueued, idempotent

### Persistence Boundaries

**When to Write Events:**

| Lifecycle Point | Event Type | Rationale |
|-----------------|-----------|-----------|
| WorkflowRunner.run() start | RunCreated | Mark run initiation |
| NodeActor ready to run | NodeScheduled | Durable "will execute" marker |
| NodeActor starts execution | NodeStarted | Begin node attempt |
| Node.run() completes | NodeCompleted | Success boundary |
| Node.run() fails | NodeFailed | Failure boundary |
| Node sends outputs | OutboxEnqueued/Sent | Message delivery tracking |
| Run ends (success) | RunCompleted | Terminal state |
| Run ends (error/cancel) | RunFailed/Cancelled | Terminal state |
| Trigger init | TriggerRegistered | Trigger lifecycle start |
| Trigger input | TriggerInputReceived | Input arrival |
| Trigger ack | TriggerCursorAdvanced | Consumption checkpoint |

**What NOT to Log:**
- Internal node processing steps (only boundaries)
- Progress updates (unless they're checkpoints)
- Ephemeral UI messages (NodeUpdate, EdgeUpdate)
- Every message enqueue/dequeue (only at send boundaries for dedup)

## Projection Model

### RunProjection Table

Materialized view of run state, rebuilt from events.

**Table: `run_projections`**

```sql
CREATE TABLE IF NOT EXISTS run_projections (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    last_event_seq INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    node_states JSONB NOT NULL,
    trigger_cursors JSONB,
    pending_messages JSONB,
    metadata JSONB
);

CREATE INDEX idx_run_projections_status ON run_projections(status);
```

**Fields:**
- `run_id`: Run identifier
- `status`: Current run status (running, completed, failed, cancelled)
- `last_event_seq`: Last event sequence processed into this projection
- `node_states`: JSON map of node_id → state
  ```json
  {
    "node_1": {
      "status": "completed",
      "attempt": 1,
      "started_at": "2025-01-01T00:00:00Z",
      "completed_at": "2025-01-01T00:01:00Z",
      "outputs": {...}
    },
    "node_2": {
      "status": "started",
      "attempt": 2,
      "started_at": "2025-01-01T00:02:00Z"
    }
  }
  ```
- `trigger_cursors`: JSON map of trigger_node_id → cursor
- `pending_messages`: JSON array of unacknowledged outbox messages
- `metadata`: Additional metadata (e.g., graph, params)

### Projection Update Algorithm

**On Event Append:**

```python
async def update_projection(run_id: str, event: RunEvent):
    projection = await get_or_create_projection(run_id)
    
    # Idempotency: skip if already processed
    if event.seq <= projection.last_event_seq:
        return
    
    # Apply event to projection
    if event.event_type == "NodeScheduled":
        projection.node_states[event.node_id] = {
            "status": "scheduled",
            "attempt": event.payload["attempt"]
        }
    elif event.event_type == "NodeStarted":
        projection.node_states[event.node_id]["status"] = "started"
        projection.node_states[event.node_id]["started_at"] = event.event_time
    elif event.event_type == "NodeCompleted":
        projection.node_states[event.node_id]["status"] = "completed"
        projection.node_states[event.node_id]["completed_at"] = event.event_time
        projection.node_states[event.node_id]["outputs"] = event.payload["outputs"]
    elif event.event_type == "TriggerCursorAdvanced":
        projection.trigger_cursors[event.node_id] = event.payload["cursor"]
    # ... handle other event types
    
    projection.last_event_seq = event.seq
    projection.updated_at = datetime.now()
    await projection.save()
```

**Replay from Scratch:**

```python
async def rebuild_projection(run_id: str):
    events = await get_events(run_id, order_by_seq=True)
    projection = RunProjection(run_id=run_id, last_event_seq=0, ...)
    
    for event in events:
        await update_projection_in_memory(projection, event)
    
    await projection.save()
    return projection
```

## Recovery Algorithm

### On Workflow Resume

When resuming a workflow (after crash, restart, or explicit resume request):

1. **Load Projection**
   ```python
   projection = await get_projection(run_id)
   if not projection:
       # First time or projection lost - rebuild from events
       projection = await rebuild_projection(run_id)
   ```

2. **Validate Consistency**
   ```python
   events = await get_events(run_id, seq_gt=projection.last_event_seq)
   if events:
       # New events since projection - replay them
       for event in events:
           await update_projection_in_memory(projection, event)
   ```

3. **Determine Resumption Points**
   ```python
   for node_id, state in projection.node_states.items():
       if state["status"] == "started":
           # Node was running but not completed - re-schedule
           await append_event(NodeScheduled(run_id, node_id, attempt=state["attempt"]+1))
       elif state["status"] == "scheduled":
           # Node was scheduled but never started - re-schedule
           await append_event(NodeScheduled(run_id, node_id, attempt=state["attempt"]))
       elif state["status"] == "completed":
           # Node completed - no action needed
           pass
   ```

4. **Re-register Triggers**
   ```python
   for trigger_id, cursor in projection.trigger_cursors.items():
       trigger_node = graph.find_node(trigger_id)
       await trigger_node.resume_from_cursor(cursor)
   ```

5. **Restart Execution**
   ```python
   # Spawn NodeActors for nodes that need to run
   # They will consume from the event log to determine what to do
   await process_graph(context, graph)
   ```

### Concurrency Safety

**Problem:** Multiple workers trying to resume the same run.

**Solution:** Lease-based locking

```sql
CREATE TABLE IF NOT EXISTS run_leases (
    run_id TEXT PRIMARY KEY,
    worker_id TEXT NOT NULL,
    acquired_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP NOT NULL
);
```

**Algorithm:**

```python
async def acquire_lease(run_id: str, worker_id: str, ttl_seconds: int = 60):
    """Acquire a lease on a run. Returns True if acquired, False if held by another."""
    now = datetime.now()
    expires = now + timedelta(seconds=ttl_seconds)
    
    try:
        # Try to insert new lease
        await db.execute(
            "INSERT INTO run_leases (run_id, worker_id, acquired_at, expires_at) "
            "VALUES (?, ?, ?, ?)",
            (run_id, worker_id, now, expires)
        )
        return True
    except IntegrityError:
        # Lease exists - check if expired
        lease = await db.fetchone(
            "SELECT worker_id, expires_at FROM run_leases WHERE run_id = ?",
            (run_id,)
        )
        if lease and lease["expires_at"] < now:
            # Expired - take over
            await db.execute(
                "UPDATE run_leases SET worker_id = ?, acquired_at = ?, expires_at = ? "
                "WHERE run_id = ? AND expires_at < ?",
                (worker_id, now, expires, run_id, now)
            )
            return True
        return False

async def renew_lease(run_id: str, worker_id: str, ttl_seconds: int = 60):
    """Renew lease held by this worker."""
    expires = datetime.now() + timedelta(seconds=ttl_seconds)
    await db.execute(
        "UPDATE run_leases SET expires_at = ? WHERE run_id = ? AND worker_id = ?",
        (expires, run_id, worker_id)
    )

async def release_lease(run_id: str, worker_id: str):
    """Release lease held by this worker."""
    await db.execute(
        "DELETE FROM run_leases WHERE run_id = ? AND worker_id = ?",
        (run_id, worker_id)
    )
```

## Implementation Steps

### Phase 1: Event Log Foundation
1. Create `RunEvent` model with all event types
2. Implement event append API with idempotency
3. Create database migration for `run_events` table
4. Add indexes for performance
5. Unit tests for event storage and retrieval

### Phase 2: Projection System
1. Create `RunProjection` model
2. Implement projection builder from events
3. Implement idempotent projection updates
4. Create database migration for `run_projections` table
5. Unit tests for projection updates and replay

### Phase 3: Workflow Runner Integration
1. Add event logging to `WorkflowRunner.run()`:
   - RunCreated at start
   - RunCompleted/RunFailed at end
2. Add event logging to node scheduling logic
3. Add lease acquisition/renewal/release
4. Integration tests for event recording

### Phase 4: Actor Integration
1. Add event logging to `NodeActor.run()`:
   - NodeScheduled before execution
   - NodeStarted at beginning
   - NodeCompleted/NodeFailed at end
2. Add outbox event logging to message sends
3. Integration tests for node lifecycle events

### Phase 5: Recovery Implementation
1. Implement `resume_workflow()` function
2. Add projection loading and consistency validation
3. Add resumption point determination logic
4. Add node re-scheduling for incomplete nodes
5. Integration tests for crash recovery scenarios

### Phase 6: Trigger Node Support
1. Design trigger node state persistence
2. Add TriggerRegistered/InputReceived/CursorAdvanced events
3. Implement trigger resume from cursor
4. Integration tests for trigger restart

### Phase 7: Testing & Validation
1. Test crash mid-node recovery
2. Test crash between completion and downstream scheduling
3. Test duplicate message deliveries
4. Test concurrent resume attempts
5. Test trigger restart with cursor
6. Performance testing for event log overhead

## Failure Modes & Recovery

### Scenario 1: Crash Mid-Node Execution

**State:**
- NodeScheduled event exists
- NodeStarted event exists
- No NodeCompleted/NodeFailed event

**Recovery:**
1. Projection shows node in "started" state
2. Increment attempt counter
3. Append new NodeScheduled event with attempt+1
4. Spawn NodeActor to re-execute

**Guarantee:** Node runs at least once, outputs idempotent

### Scenario 2: Crash After NodeCompleted, Before Downstream Scheduled

**State:**
- Node A: NodeCompleted event exists
- Node B (downstream): No NodeScheduled event

**Recovery:**
1. Projection shows Node A completed, Node B not scheduled
2. Check Node B dependencies (all inputs available)
3. Append NodeScheduled event for Node B
4. Spawn NodeActor for Node B

**Guarantee:** Downstream nodes eventually scheduled

### Scenario 3: Duplicate Message Delivery

**State:**
- OutboxEnqueued for message M
- OutboxSent for message M (twice due to retry)

**Recovery:**
1. Target node receives message M twice
2. Check OutboxSent events for message_id
3. If already processed (event exists), skip
4. Otherwise, process and append OutboxSent

**Guarantee:** At-least-once delivery, de-duplicated by target

### Scenario 4: Concurrent Resume Attempts

**State:**
- Two workers try to resume run R

**Recovery:**
1. Worker 1 acquires lease on run R
2. Worker 2 fails to acquire lease
3. Worker 2 backs off or picks different run
4. Worker 1 resumes execution
5. Worker 1 renews lease periodically
6. On completion, Worker 1 releases lease

**Guarantee:** Only one worker processes a run at a time

### Scenario 5: Trigger Node Restart

**State:**
- Trigger was consuming from offset 100
- Process crashed
- New inputs arrived (offsets 101-105)

**Recovery:**
1. Load projection, find trigger_cursors["trigger_1"] = 100
2. Re-register trigger with cursor=100
3. Trigger skips offsets ≤ 100
4. Trigger processes 101-105
5. Append TriggerCursorAdvanced(cursor=105)

**Guarantee:** No duplicate processing, no missed inputs

## Efficiency Considerations

### Event Log Overhead

**Minimize Writes:**
- Only log at durable boundaries (schedule/start/complete/fail)
- Batch append multiple events in single transaction where possible
- Do NOT log every internal step or progress update

**Indexing:**
- Primary index on (run_id, seq) for sequential replay
- Secondary index on (run_id, node_id) for node-specific queries
- Avoid over-indexing

### Projection Caching

**In-Memory Cache:**
- Keep projection in WorkflowRunner memory during execution
- Update in-memory projection on event append
- Flush to DB periodically or at boundaries
- Rebuild from DB on resume

### Event Log Compaction (Future)

**Snapshot Strategy:**
- Periodically create RunProjection snapshot
- Mark events up to snapshot as compactable
- Keep only recent events in hot storage
- Archive old events to cold storage

## Invariants Enforced by Event Log

1. **Event Ordering:** Events for a run are totally ordered by sequence number
2. **Idempotency:** Appending the same event twice (same seq) is safe
3. **Causality:** NodeStarted never precedes NodeScheduled for same attempt
4. **Completeness:** Every run has a RunCreated event
5. **Terminality:** Terminal events (Completed/Failed/Cancelled) are final
6. **Monotonicity:** Trigger cursors never decrease
7. **Message Traceability:** Every sent message has corresponding OutboxEnqueued/Sent events

## Open Questions

1. **Checkpoint Frequency:** How often to checkpoint long-running nodes?
   - Decision: Only at explicit checkpoints, not automatically
   
2. **Event Retention:** How long to keep events after run completion?
   - Decision: 30 days, then archive to cold storage
   
3. **Large Payloads:** How to handle large outputs in events?
   - **Decision**: Multi-tiered storage strategy based on output type and size
   - **AssetRef Types** (ImageRef, VideoRef, AudioRef, etc.): 
     - Use temp storage for in-flight AssetRefs instead of memory URIs
     - Store outputs durably in temp bucket for resumability
     - Log only the temp storage reference metadata (~100 bytes)
     - Promotes to permanent storage if needed after workflow completion
   - **Small Objects** (<1MB): Serialize directly into event payload as JSON
   - **Large Objects** (>1MB): Store in temp storage, log only the storage reference ID
   - **Benefits**:
     - Event log remains compact and fast to query
     - Large media files don't bloat the database
     - In-flight outputs are durable (survive crashes)
     - Recovery can reconstruct outputs by following references
     - Leverages existing temp storage infrastructure
   - **Example**:
     ```json
     {
       "outputs": {
         "image": {"type": "asset_ref", "uri": "temp://bucket/xyz.png", "asset_id": "temp_abc123"},
         "result": {"type": "inline", "value": {"status": "ok", "count": 42}},
         "large_data": {"type": "external_ref", "storage_id": "temp_output_xyz789"}
       }
     }
     ```

4. **Streaming Node Outputs:** How to handle thousands of small streaming messages?
   - **Problem**: Streaming nodes can emit thousands of chunks, causing database write contention
   - **Decision**: Compress streaming outputs at node completion
     - Don't log each streaming chunk individually (causes write contention)
     - Streaming chunks always complete until EOS (end-of-stream)
     - Log only completed nodes, not intermediate chunks
     - At node completion, compress all streamed outputs into one log entry
     - Store compressed output chunks in temp storage if needed
     - Example: Node emits 1000 image chunks → stored as single compressed array reference
   - **Benefits**:
     - Eliminates write contention on database
     - Event log stays compact (one entry per completed node, not per chunk)
     - Recovery still works (re-execute incomplete nodes, completed nodes have full output)
     - Leverages streaming's atomic completion guarantee (EOS)
   - **Implementation Note**: Current implementation logs `outputs={}` for all nodes, which inherently avoids this issue. When output tracking is enabled, must implement compression for streaming nodes.

5. **Distributed Execution:** How to coordinate across multiple machines?
   - Decision: Lease-based locking, events replicated to shared DB

6. **Backfilling:** How to add event log to existing runs?
   - Decision: New runs only, old runs remain as-is

## References

- Martin Fowler: Event Sourcing - https://martinfowler.com/eaaDev/EventSourcing.html
- Implementing Event Sourcing - https://www.eventstore.com/event-sourcing
- Azure Event Sourcing Pattern - https://learn.microsoft.com/en-us/azure/architecture/patterns/event-sourcing
