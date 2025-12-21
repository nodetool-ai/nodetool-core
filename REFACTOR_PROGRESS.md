# Resumable Workflows Architectural Refactor - Implementation Progress

## Overview

This document tracks the implementation of the architectural refactor requested in PR comment #3614840218. The refactor changes the system from **event sourcing** (events as source of truth) to **mutable state tables** (state tables as source of truth, events for audit only).

## Current Status: Phase 1 Complete (Commit c37f560)

### ✅ Completed Work

**Phase 1: New Authoritative State Tables** 
- Created 4 new model classes with full CRUD operations
- Created 4 SQL migrations for table creation
- Integrated models into migration system
- All models import and initialize successfully

**New Models**:
1. `RunState` (`run_state` table) - Run-level authoritative state
   - Status: running | suspended | completed | failed | cancelled | recovering
   - Suspension metadata (node_id, reason, state_json)
   - Completion/failure metadata
   - Optimistic locking via version field

2. `RunNodeState` (`run_node_state` table) - Per-node authoritative state
   - Composite PK: (run_id, node_id)
   - Status: idle | scheduled | running | completed | failed | suspended
   - Attempt tracking
   - Timestamps for all state transitions
   - Failure info (error, retryable)
   - Suspension/resume state
   - Optional outputs storage

3. `RunInboxMessage` (`run_inbox_messages` table) - Durable inbox
   - Unique message_id for idempotency
   - Monotonic msg_seq per (run_id, node_id, handle)
   - Status: pending | claimed | consumed
   - Claim TTL for exactly-once semantics
   - Payload inline or external reference
   - Worker claim tracking

4. `TriggerInput` (`trigger_inputs` table) - Durable trigger events
   - Unique input_id for idempotency
   - Processed flag
   - Optional cursor support
   - Cross-process coordination safe

**Migration Files**:
- `20251228000000_create_run_state.sql`
- `20251228000001_create_run_node_state.sql`
- `20251228000002_create_run_inbox_messages.sql`
- `20251228000003_create_trigger_inputs.sql`

### 📊 Impact

- **+1,088 LOC** added (4 models + 4 migrations)
- **0 breaking changes** yet (old code still works)
- **Database migrations** will run automatically on startup

---

## Remaining Work: Phases 2-8

### Phase 2: WorkflowRunner Integration (~400 LOC changes)

**Goal**: Make WorkflowRunner write to state tables instead of relying on events

**Tasks**:
1. Create RunState at workflow start
   ```python
   run_state = await RunState.create_run(self.job_id)
   ```

2. Update run_state at major boundaries:
   - On RunCreated → `run_state.status = "running"`
   - On completion → `await run_state.mark_completed()`
   - On failure → `await run_state.mark_failed(error)`
   - On suspension → `await run_state.mark_suspended(node_id, reason, state, metadata)`
   - On cancellation → `await run_state.mark_cancelled()`

3. Write run_node_state for all node transitions:
   - Node scheduled → `await node_state.mark_scheduled(attempt)`
   - Node started → `await node_state.mark_running()`
   - Node completed → `await node_state.mark_completed(outputs)`
   - Node failed → `await node_state.mark_failed(error, retryable)`
   - Node suspended → `await node_state.mark_suspended(reason, state)`

4. Make event logging non-fatal:
   ```python
   if self.event_logger:
       try:
           await self.event_logger.log_run_created(...)
       except Exception as e:
           log.warning(f"Event logging failed (non-fatal): {e}")
   ```

5. Remove any RunEvent/RunProjection reads for correctness:
   - Search for `RunEvent.get_events()` → replace with state table reads
   - Search for `RunProjection.get()` → replace with RunState.get()
   - Verify no scheduling decisions based on events

**Files to Modify**:
- `src/nodetool/workflows/workflow_runner.py` (~200 LOC changes)
- `src/nodetool/workflows/actor.py` (~100 LOC changes)
- `src/nodetool/workflows/event_logger.py` (~50 LOC changes - wrap in try/except)
- `src/nodetool/workflows/suspendable_node.py` (~50 LOC changes)

**Testing**:
- Verify run_state is created and updated correctly
- Verify node_state transitions match actual execution
- Verify event logging failures don't break workflow
- Verify suspend/resume uses state tables

---

### Phase 3: Durable Inbox (~600 LOC changes)

**Goal**: Replace in-memory NodeInbox with durable RunInboxMessage storage

**Tasks**:
1. Create DurableInbox wrapper class:
   ```python
   class DurableInbox:
       async def append(self, message_id, payload):
           await RunInboxMessage.append_message(...)
       
       async def get_pending(self, limit=100):
           return await RunInboxMessage.get_pending_messages(...)
       
       async def claim(self, message, worker_id):
           return await message.claim(worker_id)
       
       async def mark_consumed(self, message):
           await message.mark_consumed()
   ```

2. Integrate with NodeInbox:
   - Add `durable_mode` flag to NodeInbox
   - If durable, use DurableInbox instead of in-memory queue
   - Maintain backward compatibility for non-durable mode

3. Update NodeActor to use durable inbox:
   - Generate deterministic message_ids
   - Handle duplicate detection
   - Implement consumption tracking

4. Choose semantics:
   - **Option A (Simpler)**: At-least-once with offsets
     - Just mark messages as consumed
     - Accept that failures may cause duplicate processing
   - **Option B (Complex)**: Exactly-once with claims
     - Claim messages before processing
     - TTL on claims
     - Handle expired claims

5. Add payload reference support:
   - Detect large payloads (>1MB)
   - Store in external storage (Asset system)
   - Store reference in payload_ref field

**Files to Modify**:
- Create `src/nodetool/workflows/durable_inbox.py` (~300 LOC new)
- Modify `src/nodetool/workflows/inbox.py` (~150 LOC changes)
- Modify `src/nodetool/workflows/actor.py` (~150 LOC changes)

**Testing**:
- Test idempotent message delivery
- Test message ordering (seq)
- Test claim/consume cycle
- Test large payload handling
- Test duplicate message_id handling

---

### Phase 4: Trigger Wakeups (~400 LOC changes)

**Goal**: Make trigger inputs durable and remove in-memory registry

**Tasks**:
1. Store trigger inputs durably:
   ```python
   await TriggerInput.add_trigger_input(
       run_id=run_id,
       node_id=node_id,
       input_id=unique_input_id,
       payload=trigger_data,
       cursor=cursor_if_any
   )
   ```

2. Append trigger input as inbox message:
   - When trigger input arrives, also append to run_inbox_messages
   - This wakes up the trigger node

3. Remove in-memory TriggerWakeupService:
   - Delete the singleton registry
   - Replace with database queries

4. Create durable wakeup mechanism:
   ```python
   class TriggerWakeupService:
       async def find_suspended_triggers():
           # Query run_state for suspended runs
           # Query trigger_inputs for pending inputs
           # Match them up
           return runs_to_wake
       
       async def wake_up_trigger(run_id, node_id):
           # Resume the workflow
           pass
   ```

5. Handle multi-server coordination:
   - Use run_leases to prevent concurrent wake-ups
   - Queries work across all servers (no in-memory state)

**Files to Modify**:
- Modify `src/nodetool/workflows/trigger_node.py` (~200 LOC changes)
- Modify `src/nodetool/workflows/workflow_runner.py` (~100 LOC changes)
- Create `src/nodetool/workflows/trigger_wakeup_service.py` (~100 LOC new)

**Testing**:
- Test trigger input storage
- Test trigger wakeup after suspension
- Test idempotent trigger delivery
- Test multi-server trigger coordination
- Test cursor advancement

---

### Phase 5: Recovery Service Refactor (~500 LOC changes)

**Goal**: Rewrite recovery to read from state tables instead of events

**Tasks**:
1. Rewrite `WorkflowRecoveryService.resume_workflow()`:
   ```python
   async def resume_workflow(self, run_id, graph, context):
       # Acquire lease
       lease = await RunLease.acquire(...)
       
       # Read run_state (NOT projection from events)
       run_state = await RunState.get(run_id)
       if not run_state.is_resumable():
           return False, "Not resumable"
       
       # Read node states (NOT from events)
       incomplete = await RunNodeState.get_incomplete_nodes(run_id)
       suspended = await RunNodeState.get_suspended_nodes(run_id)
       
       # Schedule resumption
       for node_state in incomplete:
           await node_state.mark_scheduled(attempt=node_state.attempt + 1)
       
       for node_state in suspended:
           await node_state.mark_resuming(state=node_state.resume_state_json)
           # Set node in graph to resuming mode
           node = graph.find_node(node_state.node_id)
           if node and hasattr(node, '_set_resuming_state'):
               node._set_resuming_state(node_state.resume_state_json)
       
       # Mark run as recovering
       await run_state.mark_recovering()
       
       # Continue execution
       await workflow_runner.run(...)
   ```

2. Remove `RunProjection.rebuild_from_events()` dependency

3. Remove `determine_resumption_points()` event-based logic

4. Keep event logging for audit but don't read it

5. Add crash recovery loop:
   ```python
   async def recovery_worker_loop():
       while True:
           # Find runs stuck in "running" with expired lease
           stuck_runs = await find_stuck_runs()
           
           for run_id in stuck_runs:
               await recovery_service.resume_workflow(run_id, ...)
           
           await asyncio.sleep(30)
   ```

**Files to Modify**:
- Rewrite `src/nodetool/workflows/recovery.py` (~400 LOC changes)
- Create `src/nodetool/workflows/recovery_worker.py` (~100 LOC new)

**Testing**:
- Test recovery from crash (mid-node)
- Test recovery from crash (between nodes)
- Test suspended node resumption
- Test retryable failure handling
- Test lease expiration handling

---

### Phase 6: Event Log Deprecation (~200 LOC changes)

**Goal**: Clearly mark events as audit-only, not correctness

**Tasks**:
1. Update RunEvent docstrings:
   ```python
   """
   Audit-only event log for workflow execution.
   
   IMPORTANT: This event log is for observability and debugging only.
   It is NOT the source of truth for correctness. All scheduling and
   recovery decisions must be based on the mutable state tables
   (run_state, run_node_state) not on events.
   
   Event writes may fail without breaking workflow execution.
   Event sequencing may have gaps or be out-of-order.
   """
   ```

2. Remove strict sequencing:
   - Remove `get_next_seq()` blocking call if possible
   - Use timestamps + auto-increment ID instead
   - Or keep seq but tolerate gaps

3. Make event writes best-effort:
   - Wrap all event writes in try/except
   - Log warnings on failure
   - Don't retry or block

4. Update RunProjection docs:
   ```python
   """
   DEPRECATED: This projection is derived from audit events.
   
   Do not use for correctness decisions. Use run_state and run_node_state instead.
   This may be kept as a cached view for performance, but it's not authoritative.
   """
   ```

5. Consider removing RunProjection entirely:
   - If not used for queries, delete it
   - Or keep it as a denormalized cache but document clearly

**Files to Modify**:
- `src/nodetool/models/run_event.py` (~50 LOC doc changes)
- `src/nodetool/models/run_projection.py` (~50 LOC doc changes)
- `src/nodetool/workflows/event_logger.py` (~100 LOC changes - add try/except everywhere)

**Testing**:
- Test workflow continues when event write fails
- Test event seq can have gaps
- Verify no correctness logic reads events

---

### Phase 7: Testing (~800 LOC new tests)

**Goal**: Comprehensive test coverage for new architecture

**Test Categories**:

1. **State Table Tests** (200 LOC):
   - Test RunState transitions
   - Test RunNodeState transitions
   - Test RunInboxMessage idempotency
   - Test TriggerInput idempotency
   - Test composite keys and queries

2. **Suspend/Resume Tests** (150 LOC):
   - Test suspend writes run_state
   - Test resume reads run_state
   - Test node state restoration
   - Test state JSON serialization

3. **Trigger Wakeup Tests** (150 LOC):
   - Test trigger input storage
   - Test durable wakeup
   - Test cross-process coordination
   - Test cursor advancement
   - Test duplicate trigger handling

4. **Crash Recovery Tests** (200 LOC):
   - Test recovery from mid-node crash
   - Test recovery from between-node crash
   - Test incomplete node detection
   - Test attempt increment
   - Test lease acquisition

5. **Durable Inbox Tests** (150 LOC):
   - Test idempotent message append
   - Test message ordering (seq)
   - Test claim/consume cycle
   - Test expired claims
   - Test large payload refs

6. **Idempotency Tests** (100 LOC):
   - Test duplicate message_id
   - Test duplicate input_id
   - Test repeated resume calls
   - Test concurrent resume attempts

7. **Integration Tests** (150 LOC):
   - Test complete workflow with durable inbox
   - Test workflow with trigger suspension
   - Test workflow crash and recovery
   - Test event logging failures don't break workflow

**Files to Create/Modify**:
- Create `tests/workflows/test_state_tables.py` (~200 LOC)
- Create `tests/workflows/test_durable_inbox.py` (~150 LOC)
- Create `tests/workflows/test_crash_recovery.py` (~200 LOC)
- Create `tests/workflows/test_trigger_wakeup.py` (~150 LOC)
- Modify `tests/workflows/test_resumable_workflows.py` (~100 LOC changes)

---

### Phase 8: Documentation (~1000 LOC changes)

**Goal**: Update all documentation to reflect new architecture

**Tasks**:
1. Rewrite `RESUMABLE_WORKFLOWS_DESIGN.md`:
   - Document mutable state architecture
   - Document event log as audit-only
   - Document durable inbox semantics
   - Document trigger wakeup mechanism
   - Document recovery algorithm
   - Document failure modes

2. Update `IMPLEMENTATION_SUMMARY.md`:
   - Remove event sourcing sections
   - Add state table sections
   - Update usage examples
   - Document migration from old architecture

3. Update `TODO_RESUMABLE_WORKFLOWS.md`:
   - Mark completed items
   - Add new items for post-refactor work
   - Document known limitations

4. Create `MIGRATION_GUIDE.md`:
   - Document breaking changes
   - Provide migration path for existing code
   - Explain backward compatibility (if any)
   - Document data migration strategy

5. Update code comments:
   - Add docstrings to all new methods
   - Update docstrings on modified methods
   - Add architecture comments in key files

**Files to Modify**:
- Rewrite `RESUMABLE_WORKFLOWS_DESIGN.md` (~500 LOC changes)
- Update `IMPLEMENTATION_SUMMARY.md` (~200 LOC changes)
- Update `TODO_RESUMABLE_WORKFLOWS.md` (~100 LOC changes)
- Create `MIGRATION_GUIDE.md` (~200 LOC new)
- Various code files (~200 LOC docstring changes)

---

## Total Estimated Effort

| Phase | Description | LOC Changes | Estimated Time |
|-------|-------------|-------------|----------------|
| 1 ✅ | New state tables | +1,088 | 2 hours ✅ DONE |
| 2 | WorkflowRunner integration | ~400 | 4 hours |
| 3 | Durable inbox | ~600 | 6 hours |
| 4 | Trigger wakeups | ~400 | 4 hours |
| 5 | Recovery refactor | ~500 | 6 hours |
| 6 | Event log deprecation | ~200 | 2 hours |
| 7 | Testing | ~800 | 8 hours |
| 8 | Documentation | ~1,000 | 4 hours |
| **Total** | **All phases** | **~5,000 LOC** | **36 hours** |

**Current Progress**: Phase 1 complete (5% done)
**Remaining**: Phases 2-8 (95% remaining)

---

## Implementation Strategy

### Approach
- **Incremental**: Implement one phase at a time
- **Backward compatible** (where possible): Old code continues to work
- **Test-driven**: Write tests before/during implementation
- **Commit frequently**: One commit per phase or sub-phase

### Risk Mitigation
- Keep old event sourcing code working during transition
- Add feature flags for durable inbox (optional)
- Extensive testing before removing old code
- Document all breaking changes clearly

### Decision Points

**Inbox Semantics** (Phase 3):
- **Recommendation**: Start with at-least-once (simpler)
- Can upgrade to exactly-once later if needed

**Backward Compatibility** (Phase 6):
- **Recommendation**: Break compatibility cleanly
- Old runs require re-execution (don't try to migrate state)
- Document migration path clearly

**RunProjection** (Phase 6):
- **Recommendation**: Keep as denormalized cache, mark deprecated
- Don't delete immediately (might be useful for queries)
- Can remove in future if unused

**Streaming** (Phase 3):
- **Recommendation**: Don't persist every chunk
- Use durable cursor + EOS semantics
- Batch inserts if chunks must be persisted

---

## Next Steps

1. **Complete Phase 2**: Integrate state tables into WorkflowRunner
2. **Test Phase 2**: Verify state transitions work correctly
3. **Commit Phase 2**: Push changes
4. **Continue to Phase 3**: Implement durable inbox

---

## Questions / Clarifications Needed

1. **Inbox Semantics**: Prefer at-least-once or exactly-once?
   - At-least-once is simpler (200 LOC less)
   - Exactly-once requires claim TTL and more complexity

2. **Backward Compatibility**: Should old runs work after refactor?
   - Easier to break compatibility (fresh start)
   - Harder to migrate existing run state

3. **Streaming**: How to handle streaming chunks?
   - Don't persist every chunk (use cursor + EOS)
   - Persist compressed chunks at completion
   - Other approach?

4. **RunProjection**: Keep, deprecate, or delete?
   - Keep as cache (query performance)
   - Delete entirely (simplicity)

5. **Timeline**: Is 36 hours (1 week) acceptable for complete refactor?

---

## References

- PR Comment: #3614840218
- Commit c37f560: Phase 1 (state tables)
- Original implementation: Commits 1-11 (event sourcing)
