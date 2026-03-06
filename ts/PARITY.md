# Workflow Engine Parity: Python vs TypeScript

This document tracks parity gaps between the Python workflow engine (`src/nodetool/workflows/`) and the TypeScript kernel (`ts/packages/kernel/`).

Last updated: 2026-03-06

---

## Table of Contents

- [Critical Gaps](#critical-gaps)
- [Significant Gaps](#significant-gaps)
- [Infrastructure Gaps](#infrastructure-gaps)
- [TypeScript-Only Improvements](#typescript-only-improvements)
- [Detailed Comparison by File](#detailed-comparison-by-file)
  - [WorkflowRunner](#workflowrunner-workflow_runnerpy-vs-runnerts)
  - [NodeActor](#nodeactor-actorpy-vs-actorts)
  - [Graph](#graph-graphpy-vs-graphts)
  - [NodeInbox](#nodeinbox-inboxpy-vs-inboxts)
  - [Channel](#channel-channelpy-vs-channelts)
- [Recommended Fix Priority](#recommended-fix-priority)

---

## Critical Gaps

These will cause incorrect behavior in non-trivial workflows.

### 1. EOS Propagation (actor.ts)

**Python**: After a node completes, `_mark_downstream_eos()` calls `mark_source_done(handle)` on all downstream inboxes and posts `EdgeUpdate` messages with status `"drained"`.

**TypeScript**: No `_markDownstreamEOS()` method exists. After a node completes, downstream inboxes are never told that a source is done. Downstream nodes **hang forever** waiting for input that will never arrive.

**Files**: `actor.py` lines 955, 1318, 1412, 1475 vs `actor.ts` (missing)

---

### 2. Node Initialization

**Python**: Calls `node.initialize(context)` on every node before execution via `initialize_graph()`. Nodes that need setup (model loading, resource allocation, validation) rely on this.

**TypeScript**: Skips initialization entirely. No `initializeGraph()` call in the runner pipeline.

**Files**: `workflow_runner.py` `initialize_graph()` vs `runner.ts` (missing)

---

### 3. Graph Validation Pipeline

**Python** runs a 3-stage pipeline:
1. `_filter_invalid_edges()` — removes edges with missing source/target nodes or invalid handles
2. `validate_graph()` — per-node input validation + `validate_edge_types()` (type compatibility)
3. `initialize_graph()` — calls `node.initialize()` on all nodes

**TypeScript** only calls `graph.validate()` which checks endpoint existence. No type compatibility checking, no invalid edge removal, no node initialization.

**Files**: `workflow_runner.py` lines 1700-1735 vs `runner.ts` `run()` method

---

### 4. on_any Gather Semantics

**Python**: Waits for ALL handles to have at least one value before the first fire (combineLatest semantics). Then fires on each new arriving item.

**TypeScript**: Fires immediately when the first handle has data. This produces different execution results for any node with multiple inputs using `on_any` sync mode.

**Files**: `actor.py` lines 1233-1244 vs `actor.ts` `_gatherOnAny()` lines 269-282

---

### 5. zip_all Stickiness Logic

**Python**: Determines stickiness from edge streaming inheritance (`inherent_streaming` flag computed during `_analyze_streaming()`). A handle is sticky if its upstream edge is NOT inherently streaming.

**TypeScript**: Determines stickiness from the inbox open/closed state. A handle becomes sticky only after all its upstream sources close. This produces different batching behavior.

**Files**: `actor.py` lines 1083-1084, 1167-1171 vs `actor.ts` `_gatherZipAll()` lines 288-345

---

### 6. Control Edge Upstream Deduplication

**Python**: When initializing inboxes, counts **unique controller nodes** for the `__control__` handle:
```python
controller_count = len({e.source for e in control_edges})
inbox.add_upstream("__control__", controller_count)
```

**TypeScript**: Counts **all control edges**, even from the same source node. If one controller has 2 control edges to the same target, TS registers 2 upstreams but only 1 will send EOS, causing the target to hang.

**Files**: `workflow_runner.py` lines 1737-1783 vs `runner.ts` `_initializeInboxes()` lines 273-282

---

### 7. Completion Detection Race Condition

**Python**: After `asyncio.wait()` returns, checks `_check_pending_inbox_work()` for messages still in flight. If found, sleeps `COMPLETION_CHECK_DELAY` (0.01s) and rechecks. This handles the race where a message is enqueued between an actor finishing and the runner checking.

**TypeScript**: Uses bare `Promise.all()` with no post-completion inbox check. May declare the workflow complete while messages are still being delivered.

**Files**: `workflow_runner.py` `process_graph()` completion loop vs `runner.ts` `_processGraph()`

---

### 8. Streaming Analysis

**Python**: `_analyze_streaming()` performs BFS from all `is_streaming_output()` nodes, marking all reachable edges as streaming in `_streaming_edges`. This is used by actors to select execution mode and by stickiness logic.

**TypeScript**: No streaming analysis exists. No `_streaming_edges` dict, no `edge_streams()` query. Actors cannot distinguish streaming vs buffered edges.

**Files**: `workflow_runner.py` lines 552-606 vs `runner.ts` (missing)

---

## Significant Gaps

Missing features that won't crash basic workflows but affect correctness or capabilities.

### 9. Output Node Handling

**Python**: Dedicated `process_output_node()` method with:
- Value normalization via `context.normalize_output_value()`
- Consecutive value deduplication
- `OutputUpdate` messages for real-time client updates
- Tracks outputs in `self.outputs[node.name]` as lists

**TypeScript**: Output collection happens in the actor's `sendOutputs()` callback. No normalization, no dedup, no per-value `OutputUpdate` messages. Clients don't see outputs until the job completes.

**Files**: `workflow_runner.py` `process_output_node()` vs `runner.ts` lines 391-416

---

### 10. Multi-Edge List Type Validation

**Python**: `_classify_list_inputs()` checks `prop.type.is_list_type()` before marking a handle for list aggregation. Only list-typed properties get multi-edge aggregation.

**TypeScript**: `_detectMultiEdgeListInputs()` just counts edges per handle. If count > 1, it marks for aggregation regardless of property type. May try to aggregate into non-list properties causing runtime errors.

**Files**: `workflow_runner.py` lines 608-661 vs `runner.ts` lines 296-314

---

### 11. Controlled Node Lifecycle

**Python** (`_run_controlled_node()`):
- Saves/restores node properties for transient overrides
- Loops on `_get_next_control_event()` until EOS
- Each `RunEvent` triggers full execution (streaming input, streaming output, or buffered)
- Supports `response_future` for bidirectional agent communication
- Metadata tracking (tool_call_id, tool_name, agent_node_id, agent_iteration)
- GPU tracing per control event
- Property validation per event

**TypeScript** (`_runControlled()`):
- Basic property merge with cached inputs
- No property save/restore cycle
- No response futures
- No metadata tracking
- No GPU coordination

**Files**: `actor.py` lines 1414-1595 vs `actor.ts` lines 218-243

---

### 12. Input Dispatch Threading

**Python**: Uses `asyncio.Queue` + async dispatcher task. Thread-safe via `call_soon_threadsafe()`. Safe for WebSocket server pushing events from different threads.

**TypeScript**: Synchronous direct push to inboxes. Not safe for concurrent callers (though Node.js single-threaded nature mitigates this for most cases).

**Files**: `workflow_runner.py` lines 807-875 vs `runner.ts` `pushInputValue()` lines 130-156

---

### 13. Node Finalization

**Python**: Calls `node.finalize(context)` on every node in the `process_graph()` finally block. Closes all inboxes, drains active edges, clears memory caches, runs GC.

**TypeScript**: No finalization. No inbox closing. No edge draining. Possible resource leaks.

**Files**: `workflow_runner.py` `process_graph()` finally block vs `runner.ts` (missing)

---

### 14. Graph.from_dict() Deserialization

**Python**: Full deserialization with:
- Error recovery (`skip_errors` flag)
- Property filtering for nodes with incoming edges
- `allow_undefined_properties` flag
- Multi-pass validation

**TypeScript**: Constructs `Graph` directly from data with no validation during load.

**Files**: `graph.py` lines 98-224 vs `graph.ts` constructor

---

### 15. Edge Counter Updates

**Python**: Emits `EdgeUpdate` messages at multiple lifecycle points:
- `send_messages()` — status `"message_sent"` with counter
- `_dispatch_inputs()` — status `"message_sent"` for streaming inputs
- `_send_EOS()` — status `"completed"`
- `drain_active_edges()` — status `"drained"` for edges with pending work

**TypeScript**: Increments counters in `_incrementEdgeCounter()` but doesn't emit during input dispatch or draining. UI gets incomplete edge activity information.

**Files**: `workflow_runner.py` multiple locations vs `runner.ts` lines 538-550

---

## Infrastructure Gaps

Expected differences due to platform (Node.js vs Python). Worth tracking but not blocking.

### 16. GPU Coordination

Python has a full GPU subsystem: global FIFO lock (`acquire_gpu_lock` / `release_gpu_lock`), CUDA OOM retry logic, `preload_model()` calls, VRAM usage logging, GPU memory cleanup, `GPUIterationTracer`, and `torch_context()`.

Not applicable to Node.js runtime.

### 17. Job Persistence

Python persists job state to database via `Job` model (`mark_completed`, `mark_cancelled`, `mark_suspended`, `mark_failed`). Enables workflow resumption.

TypeScript has no database layer. Jobs are ephemeral.

### 18. OpenTelemetry Tracing

Python integrates OpenTelemetry with per-job tracers, workflow spans, and node execution spans.

TypeScript has no tracing.

### 19. Memory Profiling

Python has `MemoryProfiler` class for VRAM leak debugging with PyTorch profiler integration, snapshots, and Chrome trace export.

Not applicable to Node.js runtime.

### 20. Suspend/Resume

Python handles `WorkflowSuspendedException`: cancels pending tasks, saves node states via `RunNodeState`, marks job as suspended. Can resume later by restoring states.

TypeScript has no suspension support.

### 21. Result Caching

Python integrates with `context.get_cached_result()` / `context.cache_result_async()` for node-level result caching. Conditional: not for streaming-driven nodes.

TypeScript has no caching.

### 22. Asset Auto-Saving

Python has `_auto_save_assets()` that delegates to the asset storage module when `auto_save_asset()` is enabled on a node.

TypeScript has no asset auto-saving.

---

## TypeScript-Only Improvements

Features in TypeScript that Python does not have.

### _controlEdgesRouted Set

TypeScript tracks which control edges have actually routed events in `_controlEdgesRouted: Set<string>`. During EOS propagation, only edges that routed events get `markSourceDone("__control__")`. This prevents closing `__control__` handles that use manual `dispatchControlEvent()` instead of the automatic routing path.

Python always marks `__control__` as done for all control edges.

### Bidirectional Edge Indices

TypeScript pre-builds both `_incomingEdges` and `_outgoingEdges` indices during `Graph` construction. Python only caches outgoing edges by `(source, sourceHandle)` tuple; incoming edge lookups require iteration.

### Fail-Fast on Cycles

TypeScript throws `GraphValidationError` when `topologicalSort()` detects a cycle. Python logs a warning and continues, which can lead to confusing downstream errors.

### Clean RunResult Interface

TypeScript returns a typed `RunResult` with `outputs`, `messages`, `status`, and optional `error`. Python relies on side effects (database updates, message queue posts) and stores outputs in `self.outputs`.

---

## Detailed Comparison by File

### WorkflowRunner (`workflow_runner.py` vs `runner.ts`)

| Method (Python) | TypeScript Equivalent | Parity |
|---|---|---|
| `run()` | `run()` | Partial — missing validation/init pipeline |
| `_run_workflow()` | `run()` (inlined) | Partial |
| `_analyze_streaming()` | Missing | None |
| `_classify_list_inputs()` | `_detectMultiEdgeListInputs()` | Partial — no type check |
| `_classify_control_edges()` | Inline in `_initializeInboxes()` | Partial |
| `_initialize_inboxes()` | `_initializeInboxes()` | Partial — no dedup |
| `_filter_invalid_edges()` | Missing | None |
| `validate_graph()` | `graph.validate()` | Partial — no type checks |
| `initialize_graph()` | Missing | None |
| `send_messages()` | `_sendMessages()` | Partial — no control output extraction |
| `_dispatch_control_event()` | `dispatchControlEvent()` | Partial — no metadata |
| `_dispatch_control_event_to_target()` | `dispatchControlEventToTarget()` | Partial — no metadata |
| `_dispatch_inputs()` | `pushInputValue()` (direct) | Different — no async queue |
| `push_input_value()` | `pushInputValue()` | Partial — not thread-safe |
| `finish_input_stream()` | `finishInputStream()` | Present |
| `process_graph()` | `_processGraph()` | Partial — no race handling |
| `_mark_node_outbound_eos()` | Missing | None |
| `_check_pending_inbox_work()` | Missing | None |
| `drain_active_edges()` | Missing | None |
| `process_output_node()` | Inline in actor callback | Partial |
| `process_with_gpu()` | N/A | N/A (Node.js) |
| `acquire_gpu_lock()` | N/A | N/A (Node.js) |
| `release_gpu_lock()` | N/A | N/A (Node.js) |
| `log_vram_usage()` | N/A | N/A (Node.js) |

### NodeActor (`actor.py` vs `actor.ts`)

| Method (Python) | TypeScript Equivalent | Parity |
|---|---|---|
| `run()` | `run()` | Partial — fewer execution modes |
| `_run_buffered_node()` | `_runBuffered()` | Partial |
| `_run_streaming_input_node()` | Missing | None |
| `_run_output_node()` | Missing | None |
| `_run_controlled_node()` | `_runControlled()` | Partial — vastly simpler |
| `_run_standard_batching()` | `_gatherZipAll()` / `_gatherOnAny()` | Partial — different semantics |
| `_run_with_list_aggregation()` | Missing | None |
| `_gather_initial_inputs()` | `_gatherInputs()` | Partial |
| `_build_control_context()` | Missing | None |
| `_wait_for_control_params()` | Missing | None |
| `_validate_control_params()` | Missing | None |
| `_filter_result()` | Missing | None |
| `_auto_save_assets()` | Missing | None |
| `_mark_downstream_eos()` | Missing | None |
| `_mark_inbound_edges_drained()` | Missing | None |
| `_effective_inbound_handles()` | Missing | None |
| `_is_nonroutable_edge()` | Missing | None |
| `process_node_with_inputs()` | Inline in `_runBuffered()` | Partial |
| `process_streaming_node_with_inputs()` | Inline in `_runBuffered()` | Partial |

### Graph (`graph.py` vs `graph.ts`)

| Method (Python) | TypeScript Equivalent | Parity |
|---|---|---|
| `from_dict()` | Constructor (direct) | Partial — no validation |
| `find_node()` | `findNode()` | Present |
| `find_edges()` | `findOutgoingEdges()` | Different — TS returns all, not by handle |
| `inputs()` / `outputs()` | `inputNodes()` / `outputNodes()` | Different — type-based vs edge-based |
| `get_input_schema()` / `get_output_schema()` | Missing | None |
| `topological_sort()` | `topologicalSort()` | Present — TS throws on cycle |
| `get_control_edges()` | `getControlEdges()` | Present |
| `get_controller_nodes()` | `getControllerNodes()` | Present |
| `get_controlled_nodes()` | `getControlledNodes()` | Present |
| `validate_control_edges()` | `validateControlEdges()` | Present |
| `validate_edge_types()` | Missing | None |
| `_compute_streaming_upstream()` | `_computeStreamingUpstream()` | Present — TS no cache invalidation |
| `has_streaming_upstream()` | `hasStreamingUpstream()` | Present |

### NodeInbox (`inbox.py` vs `inbox.ts`)

| Method (Python) | TypeScript Equivalent | Parity |
|---|---|---|
| `add_upstream()` | `addUpstream()` | Present |
| `put()` | `put()` | Present — different backpressure impl |
| `prepend()` | `prepend()` | Present |
| `iter_input()` | `iterInput()` | Present |
| `iter_input_with_envelope()` | Missing | None |
| `iter_any()` | `iterAny()` | Present |
| `iter_any_with_envelope()` | Missing | None |
| `mark_source_done()` | `markSourceDone()` | Present |
| `is_open()` | `isOpen()` | Present |
| `has_buffered()` | `hasBuffered()` | Present |
| `has_pending_work()` | `hasPendingWork()` | Present |
| `has_any()` | Missing | None |
| `close_all()` | `closeAll()` | Present |
| `_notify_waiters_threadsafe()` | N/A | N/A (single-threaded) |

### Channel (`channel.py` vs `channel.ts`)

| Method (Python) | TypeScript Equivalent | Parity |
|---|---|---|
| `publish()` | `publish()` | Present — different backpressure |
| `subscribe()` | `subscribe()` | Present |
| `close()` | `close()` | Present — less error handling |
| Runtime type validation | `instanceof` check | Weaker in TS |
| `asyncio.Lock` protection | None | N/A (single-threaded) |

---

## Recommended Fix Priority

### Phase 1: Correctness (unblocks most workflows)

1. **EOS propagation** — Add `_markDownstreamEOS()` to actor.ts. After node completes, call `markSourceDone(handle)` on all downstream inboxes for each outgoing edge.
2. **Node initialization** — Add `_initializeGraph()` to runner.ts that calls `node.initialize(context)` on each node before execution.

### Phase 2: Execution Semantics (correct behavior for complex workflows)

3. **on_any gather** — Wait for all handles to have at least one value before first fire.
4. **Control edge upstream dedup** — Deduplicate control edges by source node when counting upstreams.
5. **Completion detection** — Add post-completion inbox check with retry delay.
6. **zip_all stickiness** — Port streaming analysis or align stickiness logic with Python.

### Phase 3: Feature Completeness

7. **Streaming analysis** — Port `_analyze_streaming()` BFS.
8. **Output node handling** — Add `OutputUpdate` messages and value normalization.
9. **Multi-edge list type validation** — Check property type before aggregating.
10. **Controlled node lifecycle** — Port property save/restore, response futures, metadata.
11. **Graph validation** — Port `_filter_invalid_edges()` and `validate_edge_types()`.
12. **Node finalization** — Call `node.finalize()` in finally block, close inboxes, drain edges.

### Phase 4: Production Readiness

13. Graph.from_dict() deserialization with error recovery
14. Job persistence and resume support
15. Edge counter updates at all lifecycle points
16. OpenTelemetry tracing
17. Result caching
