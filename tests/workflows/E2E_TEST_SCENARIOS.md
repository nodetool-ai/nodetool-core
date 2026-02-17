# E2E Workflow Test Scenarios

This document describes comprehensive end-to-end test scenarios for the workflow execution system, covering control edges, actor execution modes, inbox operations, workflow runner lifecycle, and complex topologies.

---

## 1. Control Edge Flow Scenarios

Tests for the async generator control edge system where nodes can control other nodes via `RunEvent` messages.

### 1.1 Basic Control Flow

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| CTRL-001 | Single Controller → Single Target | Controller emits one `RunEvent`; controlled node executes once with transient properties applied | `actor._run_controlled_node()`, `actor._execute_run_event()` | High |
| CTRL-002 | Multiple RunEvents | Controller emits 3 `RunEvent`s via generator; controlled node executes 3 times, once per event | `actor._execute_run_event()` loop | High |
| CTRL-003 | Empty RunEvent | `RunEvent` with `properties={}`; controlled node executes using default property values | `actor._execute_run_event()` with empty props | High |
| CTRL-004 | Transient Property Restoration | Multiple `RunEvent`s with different property values; verify properties are restored to original values between each execution | `node.save_original_properties()`, `node.restore_original_properties()` | High |

**Test Implementation Notes:**
- CTRL-001: Use `SimpleController` → `ThresholdProcessor` with single `RunEvent`
- CTRL-002: Create `MultiTriggerController` that yields 3+ `RunEvent`s
- CTRL-003: Controller yields `RunEvent(properties={})`, verify defaults used
- CTRL-004: Controller yields `RunEvent(properties={"threshold": 0.5})` then `RunEvent(properties={"threshold": 0.9})`; verify intermediate state doesn't leak

### 1.2 Multiple Controllers

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| CTRL-005 | Two Controllers → Same Target | Two controllers emit `RunEvent`s to same controlled node; verify node receives events and merges params in topological order | `actor._get_next_control_event()`, inbox `iter_input("__control__")` | High |
| CTRL-006 | Controller Fan-Out | Single controller controls 3 different target nodes; verify all targets receive the same `RunEvent` | `runner._dispatch_control_event()` loop over edges | High |
| CTRL-007 | Controller Finishes Early | Controller A finishes and signals EOS, Controller B still running; controlled node continues waiting for B's events | `inbox.mark_source_done()` per controller | Medium |
| CTRL-008 | Three Controllers → Same Target | Three controllers with different properties; verify property merge order (later controllers override earlier) | Control edge ordering | Low |

**Test Implementation Notes:**
- CTRL-005: Two `SimpleController` nodes → one `ThresholdProcessor`; each sets different property
- CTRL-006: One `SimpleController` → three `ThresholdProcessor` nodes via control edges
- CTRL-007: Controller A has 1 event then completes, Controller B has 3 events; verify all 3 received

### 1.3 Controller with Mixed Outputs

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| CTRL-009 | Controller Data + Control Output | Controller emits both `__control__` event AND regular data output to different downstream nodes | `runner.send_messages()` routing logic | High |
| CTRL-010 | Streaming Controller | Controller is a streaming-output node (`is_streaming_output=True`) that emits multiple `RunEvent`s over time | `actor._run_streaming_output_batched_node()` with control context | Medium |
| CTRL-011 | Controlled Node with Data Inputs | Controlled node has data edges feeding properties AND a control edge; data arrives first, then `RunEvent` overrides transiently | `actor.process_node_with_inputs()` with control params merge | High |
| CTRL-012 | Controller Outputs After Control | Controller emits control event first, then data output; verify data routing still works | `gen_process` yield order | Medium |

**Test Implementation Notes:**
- CTRL-009: Controller yields `{"__control__": RunEvent(...)}` then `{"result": "data"}`; two different downstream nodes
- CTRL-011: `FloatInput` → data edge → `ThresholdProcessor`, `SimpleController` → control edge → `ThresholdProcessor`

### 1.4 Controlled Node Variants

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| CTRL-013 | Controlled Buffered Node | Controlled node with `is_streaming_input=False, is_streaming_output=False` (default path) | `actor._run_buffered_node()` via `_execute_run_event` | High |
| CTRL-014 | Controlled Streaming Input Node | Controlled node with `is_streaming_input=True`; verify `_run_streaming_input_node` path | `actor._run_streaming_input_node()` via `_execute_run_event` | High |
| CTRL-015 | Controlled Streaming Output Node | Controlled node with `is_streaming_output=True`; verify multiple output emissions per `RunEvent` | `actor._run_streaming_output_batched_node()` via `_execute_run_event` | Medium |
| CTRL-016 | Controlled OutputNode | `OutputNode` controlled by control edge; verify capture works | `actor._run_output_node()` variant | Low |

**Test Implementation Notes:**
- CTRL-014: Create `StreamingInputProcessor` test node with `is_streaming_input() -> True`
- CTRL-015: Create `StreamingOutputProcessor` test node that yields multiple outputs

### 1.5 Error Handling in Control Flow

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| CTRL-017 | Controlled Node Error | Controlled node raises exception during execution; verify error propagates and downstream `EOS` is marked | `actor.run()` exception handler, `_mark_downstream_eos()` | High |
| CTRL-018 | Controller Error Mid-Stream | Controller fails during `gen_process`; verify controlled node receives `EOS` on `__control__` handle | `inbox.mark_source_done()` on controller task completion | High |
| CTRL-019 | Invalid Control Property | `RunEvent` contains property name that doesn't exist on controlled node; verify validation error raised | `actor._validate_control_params()` | High |
| CTRL-020 | Control Property Type Mismatch | `RunEvent` contains value with wrong type for property (e.g., string for int); verify `assign_property` error | `node.assign_property()` validation | High |
| CTRL-021 | Controlled Node Validation Error | Controlled node's property validation fails; error captured, node doesn't execute | Property validation in `assign_property` | Medium |
| CTRL-022 | Error Recovery - Next Event | Controlled node errors on one `RunEvent`, controller sends another; verify controlled node doesn't re-execute after error | Exception propagation stops loop | Medium |

**Test Implementation Notes:**
- CTRL-017: `ThresholdProcessor` raises `ValueError` in `process()`
- CTRL-018: Controller yields one event then raises exception
- CTRL-019: Controller sends `RunEvent(properties={"nonexistent_prop": 42})`
- CTRL-020: Controller sends `RunEvent(properties={"threshold": "not_a_float"})`

### 1.6 Complex Control Topologies

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| CTRL-023 | Chained Control (A→B→C) | Node A controls B via control edge, B controls C via control edge; verify hierarchical execution order | Multiple `_run_controlled_node()` instances | Medium |
| CTRL-024 | Diamond with Control | Two data paths converge on controlled node, plus one control edge; verify data aggregation then control override | Graph validation + execution order | Medium |
| CTRL-025 | Control Edge + Data Edge to Same Property | Data edge sets property `threshold=0.3`, control edge overrides to `threshold=0.8` transiently; after control event, property should be restored | Property assignment order, transient restore | High |
| CTRL-026 | Self-Loop Control (Rejected) | Node cannot control itself; graph validation should reject | `graph.validate_control_edges()` cycle detection | Low |
| CTRL-027 | Control Cycle (Rejected) | A→B→A control cycle; graph validation should reject | Cycle detection in control edges | Low |

**Test Implementation Notes:**
- CTRL-023: `SimpleController(A)` → control → `PassThrough(B)` → control → `ThresholdProcessor(C)`
- CTRL-025: `FloatInput` with `value=0.3` → data edge to `threshold` prop, `SimpleController` → control edge

### 1.7 Control Event Variants

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| CTRL-028 | StopEvent | Controller emits `StopEvent`; controlled node should stop gracefully | `ControlEvent` subclass handling | Low |
| CTRL-029 | Custom Control Event | Custom `ControlEvent` subclass; controlled node handles unknown event type gracefully | Warning log for unknown type | Low |
| CTRL-030 | Legacy Control Output | Controller emits `__control_output__` dict (legacy format); should be wrapped in `RunEvent` for compatibility | `runner.send_messages()` legacy handling | Medium |

**Test Implementation Notes:**
- CTRL-030: Controller returns `{"__control_output__": {"threshold": 0.7}}` instead of `RunEvent`

---

## 2. Actor Execution Mode Scenarios

Tests for the different execution paths in `NodeActor` based on streaming configuration.

### 2.1 Streaming Input/Output Matrix

| ID | Scenario | Input Mode | Output Mode | Description | Code Path | Priority |
|----|----------|------------|-------------|-------------|-----------|----------|
| ACTOR-001 | Buffered Node | False | False | Standard `process()` called once with gathered inputs | `actor._run_buffered_node()` → `process_node_with_inputs()` | High |
| ACTOR-002 | Streaming Input Only | True | False | Node consumes inbox via `iter_input`, emits once | `actor._run_streaming_input_node()` | High |
| ACTOR-003 | Streaming Output Only | False | True | Node called per batch, emits streaming outputs | `actor._run_streaming_output_batched_node()` | High |
| ACTOR-004 | Full Streaming | True | True | Node consumes inbox and emits outputs via `gen_process` | `actor._run_streaming_input_node()` | High |

**Test Implementation Notes:**
- Create test nodes for each combination
- ACTOR-001: Standard node with `process()` method
- ACTOR-002: Node with `is_streaming_input() -> True`, consumes via `node_inputs.stream()`
- ACTOR-003: Node with `is_streaming_output() -> True`, `gen_process()` yields outputs
- ACTOR-004: Node with both streaming flags, full streaming pipeline

### 2.2 Sync Mode Variants

| ID | Scenario | Sync Mode | Description | Code Path | Priority |
|----|----------|-----------|-------------|-----------|----------|
| ACTOR-005 | on_any Mode | "on_any" | Node fires when any input arrives (after initial batch), then on each subsequent input | `actor._run_standard_batching()` on_any branch | High |
| ACTOR-006 | zip_all Mode | "zip_all" | Node waits for aligned inputs across all handles, emits combined batches | `actor._run_standard_batching()` zip_all branch | High |
| ACTOR-007 | zip_all with Sticky Inputs | "zip_all" | Non-streaming source inputs are "sticky" - value reused across batches | `is_sticky` dict, `sticky_values` reuse | High |
| ACTOR-008 | on_any Initial Fire | "on_any" | Node with 3 input handles fires once first value from each handle arrives | `pending_handles` countdown logic | Medium |
| ACTOR-009 | on_any Subsequent Fires | "on_any" | After initial fire, node fires on each new input to any handle | `initial_fired` flag logic | Medium |
| ACTOR-010 | Sticky Value Update | "zip_all" | New value on sticky handle updates the sticky value for future batches | `sticky_values[handle] = buffers[h].popleft()` | Medium |

**Test Implementation Notes:**
- ACTOR-005: Three `FloatInput` nodes → one processor with `get_sync_mode() -> "on_any"`
- ACTOR-006: Three `FloatInput` nodes → one processor with `get_sync_mode() -> "zip_all"`
- ACTOR-007: Mix streaming and non-streaming inputs to same node

### 2.3 List Aggregation

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| ACTOR-011 | Multi-Edge to List[T] | 3 data edges feed same `list[int]` property; all values from all edges collected into single list | `actor._run_with_list_aggregation()`, `list_buffers` accumulation | High |
| ACTOR-012 | Single Edge to List[T] | One data edge feeds `list[int]` property; standard streaming behavior, NOT aggregated | `actor._run_standard_batching()` | Medium |
| ACTOR-013 | Mixed List and Non-List | One `list[int]` property with 2 edges (aggregated), one `int` property with 1 edge (standard) | Combined paths | High |
| ACTOR-014 | List with Nested Lists | Input is `list[list[int]]`; should flatten or preserve nested structure | List extension logic | Low |
| ACTOR-015 | Empty List Aggregation | No inputs arrive on list handle; result is empty list `[]` | Buffer initialization | Medium |

**Test Implementation Notes:**
- ACTOR-011: Create `ListSumProcessor` with `values: list[int]` property, 3 edges from different `IntInput` nodes
- ACTOR-013: Node with `items: list[int]` (2 edges) and `multiplier: int` (1 edge)

### 2.4 Non-Routable Upstreams

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| ACTOR-016 | Only Non-Routable Upstreams | Node fed only by non-routable outputs (e.g., Agent dynamic-only); skips execution, marks downstream EOS | `_only_nonroutable_upstreams()`, early return | Medium |
| ACTOR-017 | Mixed Routable/Non-Routable | One routable + one non-routable upstream; executes normally with routable inputs only | `_effective_inbound_handles()` filtering | Medium |
| ACTOR-018 | Agent Suppresses Output | Agent node's dynamic-only outputs not counted as routable | `should_route_output()` returns False | Low |

---

## 3. Inbox Scenarios

Tests for the `NodeInbox` class handling input buffering, EOS tracking, and backpressure.

### 3.1 Basic Operations

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| INBOX-001 | Single Handle Put/Get | Put 5 items to one handle, consume in FIFO order via `iter_input` | `inbox.put()`, `iter_input()` | High |
| INBOX-002 | Multi-Handle Arrival Order | Put items to handles A, B, A, C in sequence; `iter_any` yields in exact arrival order | `inbox.iter_any()`, `_arrival` queue | High |
| INBOX-003 | EOS Detection Single | Single handle with one upstream producer; producer signals done, consumer receives all items then EOS | `inbox.mark_source_done()`, `is_open()` check | High |
| INBOX-004 | Iter Any with EOS | `iter_any` terminates when all handles reach EOS | `any_open` check in `iter_any` | High |
| INBOX-005 | Iter Input with EOS | `iter_input(handle)` terminates when that specific handle reaches EOS | `open_counts[handle] == 0` check | High |

### 3.2 Multi-Upstream EOS

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| INBOX-006 | Two Producers Same Handle | Two edges to same handle (`add_upstream(count=2)`); handle only done when BOTH signal EOS | `add_upstream()`, `mark_source_done()` count decrement | High |
| INBOX-007 | Mixed EOS Timing | Producer A signals done early, Producer B continues sending; handle remains open until B done | `open_counts` tracking | Medium |
| INBOX-008 | All Handles EOS | 3 handles each with producers; `iter_any` terminates only when ALL complete | `any_open` aggregate check | Medium |
| INBOX-009 | Producer Done with Buffered Items | Producer signals done but items still in buffer; consumer drains buffer before seeing EOS | Drain loop before EOS check | High |

**Test Implementation Notes:**
- INBOX-006: Create inbox, `add_upstream("h", 2)`, put items, call `mark_source_done("h")` twice
- INBOX-007: Two async tasks: A puts 1 item + signals done, B puts 3 items over time + signals done

### 3.3 Backpressure

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| INBOX-010 | Buffer Limit Enforced | `buffer_limit=2`; third `put()` blocks until consumer drains at least one item | `inbox.put()` wait loop on `len(buffer) >= limit` | High |
| INBOX-011 | Backpressure Release | Producer blocked on full buffer; consumer drains item; producer unblocks | `cond.notify_all()` on consume | High |
| INBOX-012 | Multiple Blocked Producers | Two producers blocked on same full buffer; consumer drains; both race to put | Multiple waiters on condition | Medium |
| INBOX-013 | Backpressure Per Handle | `buffer_limit=2`; handle A full blocks A, handle B can still accept | Per-handle buffer limits | Low |
| INBOX-014 | No Limit (Unlimited) | `buffer_limit=None`; producer never blocks regardless of buffer size | Skip backpressure check | Low |

**Test Implementation Notes:**
- INBOX-010: Create `NodeInbox(buffer_limit=2)`, put 3 items in separate tasks, verify third blocks
- INBOX-011: Use `asyncio.wait_for()` with timeout to verify blocking/unblocking

### 3.4 Edge Cases

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| INBOX-015 | Empty Inbox (No Upstreams) | Inbox with no upstreams registered; `iter_any` returns immediately (no items, no wait) | `any_open` is False on start | Medium |
| INBOX-016 | Close All | `close_all()` called; all blocked consumers wake and terminate | `_closed=True`, `notify_all()` | Medium |
| INBOX-017 | Try Pop Non-Blocking | `try_pop_any()` returns available items without blocking | `try_pop_any()`, `_arrival` check | Low |
| INBOX-018 | Try Pop Empty | `try_pop_any()` on empty inbox returns `None` | Empty `_arrival` check | Low |
| INBOX-019 | Put After Close | `put()` after `close_all()` is no-op | `_closed` check in `put()` | Low |
| INBOX-020 | Mark Done Already Zero | `mark_source_done()` on handle with `open_count=0` stays at 0 (no negative) | `max(new_val, 0)` | Low |

### 3.5 Message Envelope

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| INBOX-021 | Metadata Propagation | Put item with metadata; consume via `iter_input_with_envelope()`; verify metadata present | `MessageEnvelope.metadata` | Medium |
| INBOX-022 | Timestamp Auto-Generated | `MessageEnvelope.timestamp` is auto-generated on put | `datetime.now(UTC)` default | Low |
| INBOX-023 | Event ID Unique | Each `MessageEnvelope.event_id` is unique | `uuid.uuid4()` default | Low |
| INBOX-024 | Backward Compat Unwrap | `iter_input()` yields unwrapped data, not envelope | `envelope.data` extraction | High |

---

## 4. Workflow Runner Scenarios

Tests for the `WorkflowRunner` orchestration layer.

### 4.1 Graph Lifecycle

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| RUNNER-001 | Valid Graph Execution | Well-formed graph with 5 nodes executes all nodes to completion | `runner.run()` happy path | High |
| RUNNER-002 | Graph Validation Failure | Invalid graph (missing required input, no edges to required prop); fails validation with error | `runner.validate_graph()` | High |
| RUNNER-003 | Node Initialization | Each node's `initialize()` called exactly once before execution | `runner.initialize_graph()` | Medium |
| RUNNER-004 | Node Finalization | Each node's `finalize()` called exactly once in finally block, even on error | `runner.run()` finally loop | High |
| RUNNER-005 | Init Error Propagates | Node raises in `initialize()`; error captured, graph execution halted | Exception in `initialize_graph()` | Medium |
| RUNNER-006 | Finalize Error Logged | Node raises in `finalize()`; error logged but doesn't prevent other finalizations | Try/except in finalize loop | Low |

### 4.2 InputNode Handling

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| RUNNER-007 | Static InputNode | Non-streaming `InputNode` emits one value then signals EOS | `push_input_value()` + `finish_input_stream()` | High |
| RUNNER-008 | Streaming InputNode | Streaming `InputNode` receives multiple values via `push_input_value()` over time | Input dispatcher loop | Medium |
| RUNNER-009 | Default Value Used | `InputNode` has `value=42` but no params provided; default is pushed | Default value push logic in `run()` | Medium |
| RUNNER-010 | Param Overrides Default | `InputNode` has `value=10`, params provide `{"input": 20}`; 20 is used | Params assignment before default check | Medium |
| RUNNER-011 | Duplicate InputNode Names | Two `InputNode`s with same `name="foo"`; raises `ValueError` | Duplicate name validation | High |
| RUNNER-012 | Missing InputNode Name | `InputNode` with empty `name=""`; raises `ValueError` | Empty name validation | High |
| RUNNER-013 | Extra Param Ignored | Params contain key not matching any `InputNode`; warning logged, param ignored | `key not in input_nodes` warning | Low |
| RUNNER-014 | Streaming Input Empty Default | Streaming `InputNode` with empty default; empty NOT pushed (waits for real input) | `is_empty()` and `is_streaming_output()` check | Medium |

### 4.3 OutputNode Handling

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| RUNNER-015 | Single OutputNode | One `OutputNode` captures final result in `runner.outputs` | `runner.process_output_node()` | High |
| RUNNER-016 | Streaming to OutputNode | Streaming node feeds `OutputNode`; multiple values captured in order | `actor._run_output_node()` iter loop | Medium |
| RUNNER-017 | Multiple OutputNodes | Two `OutputNode`s with different names; both captured separately | `runner.outputs` dict by name | Medium |
| RUNNER-018 | OutputNode Value Update | Same value received twice; only stored once (dedup) | `outputs[name][-1] != value` check | Low |

### 4.4 Streaming Analysis

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| RUNNER-019 | Streaming Propagation | Node with `is_streaming_output=True` marks all downstream edges as streaming | `runner._analyze_streaming()` BFS walk | High |
| RUNNER-020 | Control Edge Excluded | Control edges don't participate in streaming propagation | Skip `edge_type == "control"` | High |
| RUNNER-021 | No Streaming Sources | All nodes return `is_streaming_output=False`; all edges marked non-streaming | Default `False` in `_streaming_edges` | Low |
| RUNNER-022 | Multiple Streaming Sources | Two streaming sources; edges from both marked, convergence point streaming | Multiple BFS seeds | Medium |

### 4.5 List Input Classification

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| RUNNER-023 | Multi-Edge List Detected | Two edges to same `list[T]` handle; detected for aggregation | `runner._classify_list_inputs()` | High |
| RUNNER-024 | Single Edge List Not Aggregated | One edge to `list[T]` handle; NOT marked for aggregation (streaming behavior) | `len(edges) > 1` check | Medium |
| RUNNER-025 | Non-List Multi-Edge Rejected | Two edges to non-list property; validation should fail (caught elsewhere) | `not prop.type.is_list_type()` skip | Low |

### 4.6 Edge Filtering

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| RUNNER-026 | Invalid Source Node Removed | Edge references non-existent source node; edge silently removed | `runner._filter_invalid_edges()` source check | Medium |
| RUNNER-027 | Invalid Target Node Removed | Edge references non-existent target node; edge silently removed | Target node check | Medium |
| RUNNER-028 | Invalid Source Handle Removed | Edge's `sourceHandle` not a declared output of source; edge removed | `find_output_instance()` check | Medium |
| RUNNER-029 | Invalid Target Handle Removed | Edge's `targetHandle` not a declared property AND node not dynamic; edge removed | `find_property()` + `is_dynamic()` check | Medium |
| RUNNER-030 | Dynamic Node Accepts Any | Dynamic node accepts edge to undeclared handle; edge kept | `is_dynamic()` returns True | Medium |
| RUNNER-031 | Control Edges Pass Through | Control edges skip source/target handle validation (uses special handles) | `edge_type == "control"` early continue | High |

### 4.7 Error Handling

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| RUNNER-032 | Node Execution Error | Node raises exception during execution; error captured, job marked "error", `JobUpdate` posted | Exception handler in `run()` | High |
| RUNNER-033 | Job Status Progression | Job transitions: "running" → "completed" with `JobUpdate` messages | `JobUpdate` status field | Medium |
| RUNNER-034 | Cancellation | `asyncio.CancelledError` raised; status="cancelled", `JobUpdate` posted | `CancelledError` handler | High |
| RUNNER-035 | Workflow Suspension | `WorkflowSuspendedException` raised; status="suspended", node state saved | Suspension handler, `RunNodeState.mark_suspended()` | Medium |
| RUNNER-036 | OOM Error Message | CUDA OOM error gets special message formatting | `is_cuda_oom_exception()` check | Low |
| RUNNER-037 | Error in Finally | Error during finalization doesn't prevent job status update | Finally runs after status set | Low |

### 4.8 Caching

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| RUNNER-038 | Cache Hit | Same node with same inputs executed twice; second returns cached result | `context.get_cached_result()` | High |
| RUNNER-039 | Cache Miss | First execution; result cached for future runs | `context.cache_result_async()` | High |
| RUNNER-040 | Cache Disabled | `disable_caching=True`; no caching occurs | Skip cache check when disabled | Medium |
| RUNNER-041 | Streaming Upstream Skips Cache | Node with streaming upstream NOT cached (invalidates cache key) | `driven_by_stream` check | High |
| RUNNER-042 | Cache Key Includes Props | Cache key includes property values; different props = cache miss | Cache key generation | Medium |

### 4.9 Input Dispatcher

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| RUNNER-043 | Push Dispatches to Edges | `push_input_value()` dispatches to all edges from that `InputNode` | `_dispatch_inputs()` edge loop | High |
| RUNNER-044 | Finish Marks Edges Done | `finish_input_stream()` marks all target handles as done | `mark_source_done()` per edge | High |
| RUNNER-045 | Shutdown Stops Dispatcher | `{"op": "shutdown"}` event stops dispatcher loop | Shutdown check in `_dispatch_inputs()` | Medium |
| RUNNER-046 | Thread-Safe Enqueue | `_enqueue_input_event()` works from different thread | `call_soon_threadsafe()` | Medium |
| RUNNER-047 | Invalid Event Dropped | Event missing required fields; error logged, event dropped | Field validation | Low |

### 4.10 Job Model Tracking

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| RUNNER-048 | Job Created | Job model created if not exists | `Job.get()` then `Job.create()` | Low |
| RUNNER-049 | Job Marked Completed | On success, `job_model.mark_completed()` called | Status update in success path | Low |
| RUNNER-050 | Job Marked Failed | On error, `job_model.mark_failed()` called | Status update in error path | Low |
| RUNNER-051 | Job Marked Cancelled | On cancellation, `job_model.mark_cancelled()` called | Status update in cancel path | Low |

---

## 5. GPU Coordination Scenarios

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| GPU-001 | GPU Node Acquires Lock | Node with `requires_gpu=True` acquires global `gpu_lock` | `acquire_gpu_lock()` | High |
| GPU-002 | GPU Lock FIFO | Two GPU nodes; second waits for first to release lock | Lock ordering, `_gpu_lock_holder` tracking | Medium |
| GPU-003 | GPU Node on CPU Device | GPU node but `device="cpu"`; raises `RuntimeError` | Device check before lock | High |
| GPU-004 | GPU Error Releases Lock | GPU node raises exception; lock released in `finally` block | `release_gpu_lock()` in finally | High |
| GPU-005 | GPU Lock Timeout | GPU lock held too long; timeout error raised | `GPU_LOCK_TIMEOUT` check | Low |
| GPU-006 | GPU Status Tracking | `get_gpu_lock_status()` returns holder info | `_gpu_lock_holder` dict | Low |
| GPU-007 | Waiting Status Update | Node waiting for GPU lock sends "waiting" status update | `await node.send_update(context, status="waiting")` | Low |

---

## 6. Complex Workflow Topologies

### 6.1 Graph Shapes

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| COMPLEX-001 | Diamond Dependency | A → B, A → C, B → D, C → D; verify D waits for both B and C | Topological sort, inbox waits | High |
| COMPLEX-002 | Long Linear Chain | 10-node linear chain A→B→C→...→J; verify execution order | Sequential dependency resolution | Medium |
| COMPLEX-003 | Wide Fan-Out | 1 source → 10 targets; verify all 10 receive data and execute | Edge dispatch to all targets | Medium |
| COMPLEX-004 | Wide Fan-In | 10 sources → 1 target; verify target waits for all 10 | Multi-upstream inbox, `zip_all` or `on_any` | Medium |
| COMPLEX-005 | Tree Structure | Root → 3 children → 9 grandchildren; verify breadth-first-ish execution | Topological levels | Low |
| COMPLEX-006 | Disconnected Components | Two separate subgraphs with no edges between; both execute independently | Independent task spawning | Medium |

### 6.2 Error Propagation

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| COMPLEX-007 | Error in Fan-Out | Source → 3 targets, target 2 errors; targets 1 and 3 complete, target 2 error | Independent error per branch | Medium |
| COMPLEX-008 | Error Blocks Downstream | A → B → C, B errors; C never executes (no input) | Inbox never receives, task completes | High |
| COMPLEX-009 | Error with Side Paths | Main path errors, side path completes independently | Parallel task execution | Medium |

### 6.3 Streaming + Static Mix

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| COMPLEX-010 | Streaming and Static Inputs | Node receives streaming input A and static input B; executes per A item with sticky B | `zip_all` with sticky values | High |
| COMPLEX-011 | Two Streaming Sources | Node receives from two streaming sources; emits on aligned pairs | `zip_all` buffer alignment | Medium |
| COMPLEX-012 | Streaming Source + Multiple Static | One streaming + two static inputs; all statics sticky | Multiple sticky handles | Medium |

---

## 7. Node I/O Scenarios

### 7.1 NodeInputs Wrapper

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| IO-001 | First Returns Value | `inputs.first("handle")` returns first available value | `iter_input` take first | High |
| IO-002 | First Returns Default | `inputs.first("handle", default=42)` returns 42 if EOS with no items | Default return on empty | Medium |
| IO-003 | Stream Yields All | `inputs.stream("handle")` yields all items in order | `iter_input` loop | High |
| IO-004 | Any Cross-Handle | `inputs.any()` yields `(handle, item)` in arrival order | `iter_any` wrapper | High |
| IO-005 | Has Buffered Check | `inputs.has_buffered("handle")` reflects buffer state | `inbox.has_buffered()` | Low |
| IO-006 | Has Stream Check | `inputs.has_stream("handle")` reflects open upstream | `inbox.is_open()` | Low |
| IO-007 | Envelope Access | `inputs.first_with_envelope()` returns full envelope | `iter_input_with_envelope` | Medium |

### 7.2 NodeOutputs Wrapper

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| IO-008 | Emit Validates Slot | `outputs.emit("invalid_slot", value)` raises `ValueError` | `find_output_instance()` check | High |
| IO-009 | Emit Routes to Inbox | `outputs.emit("output", value)` puts to target inbox | `runner.send_messages()` dispatch | High |
| IO-010 | Emit Default Slot | `outputs.default(value)` emits to "output" or single slot | Default slot selection | Medium |
| IO-011 | Emit Collects | Multiple emits to same slot; `collected()` returns last value | `_collected` dict update | High |
| IO-012 | Capture Only Mode | `capture_only=True`; emits collected but not routed | Skip `send_messages` call | Medium |
| IO-013 | Complete Marks EOS | `outputs.complete("slot")` marks downstream handle done | `inbox.mark_source_done()` | Medium |
| IO-014 | Non-Routable Suppressed | `should_route_output() == False`; emit doesn't route | Early return in `emit()` | Medium |
| IO-015 | Metadata Propagates | `emit(slot, value, metadata={...})` attaches metadata to message | `inbox.put(handle, value, metadata)` | Medium |
| IO-016 | Auto-Save Asset | Node with `auto_save_asset() == True`; asset saved on emit | `auto_save_assets()` call | Low |

---

## 8. Legacy Compatibility

| ID | Scenario | Description | Code Path | Priority |
|----|----------|-------------|-----------|----------|
| LEGACY-001 | ControlAgent __control_output__ | Legacy `ControlAgent` outputs `__control_output__` dict; wrapped in `RunEvent` | `runner.send_messages()` legacy handling | High |
| LEGACY-002 | Non-Agent Controller Allowed | Graph validation allows non-Agent nodes as controllers | `graph.validate_control_edges()` no type restriction | High |
| LEGACY-003 | Edge Type Defaults to Data | Edge without `edge_type` field defaults to `"data"` | `Edge.edge_type` default | Medium |

---

## Summary Statistics

| Category | Scenarios | High Priority |
|----------|-----------|---------------|
| 1. Control Edge Flow | 30 | 16 |
| 2. Actor Execution Modes | 18 | 9 |
| 3. Inbox Operations | 24 | 10 |
| 4. Workflow Runner | 51 | 20 |
| 5. GPU Coordination | 7 | 3 |
| 6. Complex Topologies | 12 | 4 |
| 7. Node I/O | 16 | 6 |
| 8. Legacy Compatibility | 3 | 2 |
| **Total** | **161** | **70** |

---

## Implementation Priority

### Phase 1: Core Control Edge Flow (High Priority)
1. CTRL-001, CTRL-002, CTRL-003, CTRL-004 - Basic control flow
2. CTRL-006, CTRL-009 - Fan-out and mixed outputs
3. CTRL-011, CTRL-017, CTRL-019, CTRL-020 - Error handling
4. CTRL-025 - Data + control to same property

### Phase 2: Actor Modes & Inbox (High Priority)
1. ACTOR-001, ACTOR-005, ACTOR-006 - Sync modes
2. ACTOR-007, ACTOR-011, ACTOR-013 - List aggregation
3. INBOX-001, INBOX-002, INBOX-003, INBOX-006 - Basic operations
4. INBOX-010, INBOX-011 - Backpressure

### Phase 3: Workflow Runner (High Priority)
1. RUNNER-001, RUNNER-002 - Graph lifecycle
2. RUNNER-007, RUNNER-011, RUNNER-012 - InputNode handling
3. RUNNER-015, RUNNER-019, RUNNER-020 - Output and streaming
4. RUNNER-032, RUNNER-034, RUNNER-038, RUNNER-041 - Error and cache

### Phase 4: Complex Scenarios (Medium Priority)
1. COMPLEX-001, COMPLEX-008 - Diamond and error propagation
2. COMPLEX-010, COMPLEX-011 - Mixed streaming
3. CTRL-023 - Chained control

---

## Test Helper Nodes Reference

Existing test nodes in `nodetool.workflows.test_helper`:
- `StringInput`, `FloatInput`, `IntInput` - Input nodes
- `StringOutput` - Output node
- `ThresholdProcessor` - Node with configurable threshold for control testing
- `SimpleController` - Controller that yields `RunEvent`
- `IntAccumulator` - Tracks execution count for multi-trigger testing
- `FormatText` - Simple string formatting

Additional test nodes needed:
- `MultiTriggerController` - Yields multiple `RunEvent`s
- `StreamingInputProcessor` - `is_streaming_input() -> True`
- `StreamingOutputProcessor` - `is_streaming_output() -> True`
- `ListSumProcessor` - `list[int]` property for aggregation testing
- `ErrorProcessor` - Raises exception on demand
