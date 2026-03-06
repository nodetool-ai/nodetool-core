# Kernel Tasks — `packages/kernel`

Parity gaps between `src/nodetool/workflows/` (Python) and `ts/packages/kernel/` (TypeScript).
Regression tests live in `ts/packages/kernel/tests/`.

**Status:** 🔴 open · 🟡 in progress · 🟢 done · ⚪ N/A

---

## Already fixed

| ID | Gap | Status |
|----|-----|--------|
| K-1 | EOS propagation — `_sendEOS()` calls `markSourceDone` after each actor | 🟢 |
| K-2 | Node initialization — `_initializeGraph()` calls `executor.initialize()` | 🟢 |
| K-4 | on_any gather semantics — combineLatest, wait for all handles first | 🟢 |
| K-6 | Control edge upstream dedup — count unique source nodes | 🟢 |
| K-7 | Completion detection race — `_checkPendingInboxWork()` + delay | 🟢 |
| K-8 | Streaming analysis — `_analyzeStreaming()` BFS, `edgeStreams()` | 🟢 |
| K-13 | Node finalization in `finally` block | 🟢 |
| K-3b | `_filterInvalidEdges()` removes edges with missing nodes | 🟢 |
| K-3d | `validateEdgeTypes()` checks source→target type compat | 🟢 |
| K-5 | Pre-computed stickyHandles wired from runner to actor | 🟢 |

Regression tests for open gaps: `tests/parity-graph-validation.test.ts`, `tests/parity-stickiness-gaps.test.ts`, `tests/parity-output-edge-gaps.test.ts`, `tests/parity-actor-lifecycle-gaps.test.ts`

---

## Phase 1 — Correctness

### T-K-3 · Graph validation pipeline
**Status:** 🔴 open
**Regression test:** `tests/parity-graph-validation.test.ts` (9 todos exist)
**Python:** `workflow_runner.py` runs `_filter_invalid_edges()` then `validate_edge_types()` before execution.
**Gap:** TS only calls `graph.validate()` (endpoint existence). No type compatibility checking, no invalid-edge removal.

Implementation tasks:
- [ ] **T-K-3a TEST** — Write failing test: graph with edge to missing node is silently filtered (not thrown). Currently TS throws; Python filters.
- [ ] **T-K-3b IMPL** — Add `_filterInvalidEdges()` to `runner.ts`. Remove edges whose `source` or `target` don't exist in `graph.nodes`. Call before `graph.validate()`.
- [ ] **T-K-3c TEST** — Write failing test: edge between incompatible types raises `GraphValidationError` during `validate_edge_types()`.
- [ ] **T-K-3d IMPL** — Add `validateEdgeTypes()` to `graph.ts`. Match `sourceHandle` type against `targetHandle` type from node descriptors. Raise on mismatch.

---

### T-K-5 · zip_all stickiness uses streaming analysis
**Status:** 🔴 open
**Regression test:** `tests/parity-stickiness-gaps.test.ts` (4 tests, all pass but document concurrent scenario gap in comments)
**Python:** A handle is sticky if its upstream edge is NOT inherently streaming (from `_analyze_streaming()`). Stickiness is set at graph-init time, not at EOS time.
**Gap:** TS `_gatherZipAll()` treats a handle as sticky only after all upstreams close. In a concurrent scenario where a non-streaming source hasn't closed yet but a streaming source is active, TS blocks incorrectly.

Implementation tasks:
- [ ] **T-K-5a TEST** — Write concurrent timing test: non-streaming A → C (zip_all), streaming B → C. Assert C fires on each B item reusing A's value *before* A closes. Use `Promise.race` / manual inbox ordering.
- [ ] **T-K-5b IMPL** — Wire `runner.edgeStreams(edge)` into `NodeActor._gatherZipAll()`. Pre-compute sticky handles at actor construction time: a handle is sticky iff its upstream edge is not streaming. Pass this map into the actor.
- [ ] **T-K-5c IMPL** — Modify `_gatherZipAll()` to use the pre-computed sticky map instead of checking `isOpen()` to decide stickiness.

---

## Phase 2 — Feature completeness

### T-K-9 · OutputUpdate messages per value
**Status:** 🔴 open
**Regression test:** `tests/parity-output-edge-gaps.test.ts` (documents gap, currently 0 `output_update` messages emitted)
**Python:** `process_output_node()` emits `OutputUpdate` per produced value with normalization and consecutive dedup.
**Gap:** TS collects outputs in `result.outputs` only after job completes. No `output_update` messages emitted.

Implementation tasks:
- [ ] **T-K-9a TEST** — Write test: run pipeline ending at output node, assert `result.messages` contains at least one message with `type === "output_update"` including `node_id`, `output_name`, `value`.
- [ ] **T-K-9b IMPL** — Add `_processOutputNode()` to `runner.ts`. When an output node produces a value, emit `output_update` message. Track `_lastOutputValues` per node for consecutive dedup.
- [ ] **T-K-9c IMPL** — Add value normalization hook. For now, pass through; later tie to `context.normalize_output_value()`.

---

### T-K-10 · Multi-edge list type validation
**Status:** 🔴 open
**Regression test:** `tests/parity-stickiness-gaps.test.ts` (documents current behavior — aggregates regardless of type)
**Python:** `_classify_list_inputs()` checks `prop.type.is_list_type()` before marking a handle for list aggregation.
**Gap:** TS `_detectMultiEdgeListInputs()` marks for aggregation purely based on edge count > 1.

Implementation tasks:
- [ ] **T-K-10a TEST** — Write test: two edges to a non-list-typed handle. Assert aggregation is NOT applied (values are not wrapped in array).
- [ ] **T-K-10b IMPL** — Extend `NodeDescriptor` protocol type to include property type info (or pass node class metadata). Update `_detectMultiEdgeListInputs()` to check `prop.type.is_list_type` before marking.

---

### T-K-11 · Controlled node lifecycle
**Status:** 🔴 open
**Regression test:** `tests/parity-actor-lifecycle-gaps.test.ts` (2 todos: response_future, metadata)
**Python:** `_run_controlled_node()` saves/restores node properties, supports `response_future` for bidirectional comms, tracks `tool_call_id`/`tool_name`/`agent_node_id`/`agent_iteration` metadata.
**Gap:** TS `_runControlled()` does basic property merge only. No save/restore, no futures, no metadata.

Implementation tasks:
- [ ] **T-K-11a TEST** — Write test: mutating node properties during execution should not bleed into next control event (property snapshot/restore).
- [ ] **T-K-11b IMPL** — Add property snapshot before each control event execution; restore after. Use `structuredClone(node.properties)`.
- [ ] **T-K-11c TEST** — Write test: control event with `tool_call_id` field in metadata; assert `NodeUpdate` message contains matching `tool_call_id`.
- [ ] **T-K-11d IMPL** — Extract metadata fields (`tool_call_id`, `tool_name`, `agent_node_id`, `agent_iteration`) from `ControlEvent.properties` and attach to emitted `NodeUpdate` messages.
- [ ] **T-K-11e TEST** — Write todo test for `response_future` bidirectional communication pattern.
- [ ] **T-K-11f IMPL** — Design and implement `response_future` support (promise-based callback from controlled node back to controller).

---

### T-K-14 · Graph.fromDict() with error recovery
**Status:** 🔴 open
**Regression test:** `tests/parity-graph-validation.test.ts` (6 todos for fromDict API)
**Python:** `Graph.from_dict(skip_errors=True)` drops unrecognized node types and their edges. `allow_undefined_properties` flag ignores unknown fields.
**Gap:** TS constructs `Graph` directly from data, no validation during load.

Implementation tasks:
- [ ] **T-K-14a TEST** — Write failing test: `Graph.fromDict({ nodes, edges }, { skipErrors: true })` with an unrecognized node type. Assert graph constructed with that node dropped and its edges removed.
- [ ] **T-K-14b TEST** — Write failing test: `Graph.fromDict` with `skipErrors: false` on unrecognized type throws.
- [ ] **T-K-14c IMPL** — Add `Graph.fromDict(data, opts?)` static factory. Options: `skipErrors`, `allowUndefinedProperties`. Filter invalid nodes/edges when `skipErrors=true`.
- [ ] **T-K-14d TEST** — Write failing test: node with incoming edge has its static property value ignored (edge value takes precedence at load time).
- [ ] **T-K-14e IMPL** — During `fromDict`, strip static property values from node descriptors for handles that have an incoming edge.

---

### T-K-15 · Edge counter updates at all lifecycle points
**Status:** 🔴 open
**Regression test:** `tests/parity-output-edge-gaps.test.ts` (documents gaps: no "drained", no "active" on dispatch)
**Python:** Emits `EdgeUpdate` at send_messages ("message_sent"), dispatch ("message_sent"), EOS ("completed"), drain ("drained").
**Gap:** TS emits "active" in `_sendMessages` and "completed" in `_sendEOS`, but not during `_dispatchInputs` and not "drained".

Implementation tasks:
- [ ] **T-K-15a TEST** — Write test: params-dispatched edges should emit `edge_update` with `status: "active"` during `_dispatchInputs`. Currently missing.
- [ ] **T-K-15b IMPL** — Call `_incrementEdgeCounter()` for each edge during `_dispatchInputs` delivery.
- [ ] **T-K-15c TEST** — Write test: after all actors complete, input→target edges emit `edge_update` with `status: "drained"`.
- [ ] **T-K-15d IMPL** — Add `_drainActiveEdges()` called in post-completion cleanup. For edges with pending work, emit `edge_update` status "drained".

---

## Phase 3 — Infrastructure (optional / low priority)

### T-K-12 · Input dispatch async queue
**Status:** ⚪ low priority
**Python:** `asyncio.Queue` + dispatcher task for thread-safe input delivery.
**Gap:** TS does synchronous direct push; safe for Node.js single-threaded model.
No action needed unless TS moves to worker threads.

### T-K-16 · GPU coordination
**Status:** ⚪ N/A — not applicable to Node.js runtime.

### T-K-17 · Job persistence
**Status:** ⚪ deferred — tracked in [tasks-models.md](tasks-models.md) (Job model methods).

### T-K-18 · OpenTelemetry tracing
**Status:** ⚪ deferred — nice to have, not blocking.

### T-K-19 · Memory profiling
**Status:** ⚪ N/A — PyTorch specific.

### T-K-20 · Suspend/resume
**Status:** ⚪ deferred — complex; blocked on Job persistence.

### T-K-21 · Result caching
**Status:** ⚪ deferred — depends on storage layer.

### T-K-22 · Asset auto-saving
**Status:** ⚪ deferred — depends on storage layer.
