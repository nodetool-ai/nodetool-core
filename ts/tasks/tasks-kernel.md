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
| K-3 | Graph validation — `_filterInvalidEdges()` + `validateEdgeTypes()` | 🟢 |
| K-4 | on_any gather semantics — combineLatest, wait for all handles first | 🟢 |
| K-5 | Pre-computed stickyHandles wired from runner to actor | 🟢 |
| K-6 | Control edge upstream dedup — count unique source nodes | 🟢 |
| K-7 | Completion detection race — `_checkPendingInboxWork()` + delay | 🟢 |
| K-8 | Streaming analysis — `_analyzeStreaming()` BFS, `edgeStreams()` | 🟢 |
| K-9 | OutputUpdate messages per value | 🟢 |
| K-11 | Controlled node lifecycle (sendControlEvent with response promise) | 🟢 |
| K-13 | Node finalization in `finally` block | 🟢 |
| K-14 | Graph.fromDict() with basic input validation | 🟢 |

---

## Open gaps

### T-K-10 · Multi-edge list type validation
**Status:** 🟢 done — `NodeDescriptor.propertyTypes` added; `_detectMultiEdgeListInputs()` checks `TypeMetadata.isListType()` before marking.

---

### T-K-15 · Edge counter updates at all lifecycle points
**Status:** 🟢 done — `_incrementEdgeCounter()` in `_dispatchInputs()`, `_drainActiveEdges()` in post-completion cleanup.

---

## Deferred / N/A

| ID | Description | Status |
|----|-------------|--------|
| K-12 | Input dispatch async queue | ⚪ N/A (Node.js single-threaded) |
| K-16 | GPU coordination | ⚪ N/A |
| K-17 | Job persistence | ⚪ deferred (tracked in T-WS-18) |
| K-18 | OpenTelemetry tracing | ⚪ deferred |
| K-19 | Memory profiling | ⚪ N/A (PyTorch) |
| K-20 | Suspend/resume | ⚪ deferred |
| K-21 | Result caching | ⚪ deferred |
| K-22 | Asset auto-saving | ⚪ deferred |
