**Rewrite Objective**
Full TypeScript rewrite of the workflow runtime so that scheduling, execution, control edges, streaming, state/recovery, and job lifecycle no longer depend on Python runtime code.

---

**Implementation Progress**

Status legend: ✅ Done | 🚧 Partial | ⬜ Not started

| Phase | Description | Status | Notes |
|-------|-------------|--------|-------|
| 0 | Spec Freeze & Parity Definition | 🚧 | Spec derived from Python source; golden traces deferred |
| 1 | TS Foundations & Protocol Package | ✅ | Workspace, protocol types (18 message types), contract tests (15 passing) |
| 2 | Graph & Inbox Kernel | ✅ | Graph model, validation, topological sort, NodeInbox with backpressure (33 tests) |
| 3 | Actor Runtime Core | ✅ | Buffered, streaming-output, controlled modes; on_any & zip_all sync modes (6 tests) |
| 4 | WorkflowRunner Orchestration | ✅ | Graph init, inbox setup, input dispatch, actor spawn, EOS routing, edge counters (7 tests) |
| 5 | Node SDK & Registry | ✅ | BaseNode with lifecycle hooks & streaming; NodeRegistry; 11 example nodes; 34 tests |
| 6 | Processing Context, Assets, Caching | 🚧 | ProcessingContext, MemoryCache, asset-safe sanitizeForClient, in-memory + file storage adapters, workspace path resolution (23 tests); S3 adapter + full asset normalization pending |
| 7 | Job Execution + State/Recovery | ⬜ | Deferred (out of scope – pure execution engine goal) |
| 8 | Dual-Run Shadow & Diff Harness | ⬜ | |
| 9 | Canary & Cutover | ⬜ | |
| 10 | Python Decommission | ⬜ | |

**Test Summary**: 118 tests across 11 test files, all passing.

**Packages implemented** (under `ts/`):
- `@nodetool/protocol` – Message types, graph types, control events
- `@nodetool/kernel` – Graph, NodeInbox, NodeActor, WorkflowRunner
- `@nodetool/runtime` – ProcessingContext, MemoryCache, output sanitization, storage adapters
- `@nodetool/node-sdk` – BaseNode, NodeRegistry, 11 example/test nodes

**Remaining work for completed phases**:
- Phase 2: MessagePack serialization compatibility tests (deferred)
- Phase 3: GPU lock / memory cleanup (not applicable in TS; defer to job-exec)
- Phase 4: Sub-graph / GroupNode execution (deferred)
- Phase 5: Typed property validation, more node ports from Python ecosystem
- Phase 6: S3 storage adapter, full asset normalization/materialization modes

---

**Baseline To Replace**
Current Python components to mirror behavior from:
- [workflow_runner.py](/Users/mg/workspace/nodetool-core/src/nodetool/workflows/workflow_runner.py)
- [actor.py](/Users/mg/workspace/nodetool-core/src/nodetool/workflows/actor.py)
- [inbox.py](/Users/mg/workspace/nodetool-core/src/nodetool/workflows/inbox.py)
- [base_node.py](/Users/mg/workspace/nodetool-core/src/nodetool/workflows/base_node.py)
- [processing_context.py](/Users/mg/workspace/nodetool-core/src/nodetool/workflows/processing_context.py)
- [run_workflow.py](/Users/mg/workspace/nodetool-core/src/nodetool/workflows/run_workflow.py)
- [types.py](/Users/mg/workspace/nodetool-core/src/nodetool/workflows/types.py)
- [test_e2e_runner_scenarios.py](/Users/mg/workspace/nodetool-core/tests/workflows/test_e2e_runner_scenarios.py)
- [test_e2e_actor_scenarios.py](/Users/mg/workspace/nodetool-core/tests/workflows/test_e2e_actor_scenarios.py)
- [docs/websocket-protocol.md](/Users/mg/workspace/nodetool-core/docs/websocket-protocol.md)

**Assumptions**
- “Full rewrite” means TS is the production source of truth for runner behavior.
- Temporary Python bridge is allowed only during migration and removed at end.
- External API/WebSocket message contracts stay backward compatible.
- Existing workflow JSON graph format remains compatible.

---

**Target Architecture**
1. `workflow-kernel-ts`
- Graph loading/validation.
- Actor runtime.
- Node inbox and backpressure.
- Control event routing.
- Streaming propagation and sync modes.

2. `workflow-node-sdk-ts`
- `BaseNode` equivalent.
- Typed property metadata and validation.
- Node lifecycle hooks: `initialize`, `preProcess`, `run`, `finalize`.
- Streaming input/output helpers.

3. `workflow-runtime-ts`
- Processing context.
- Message queue and emission.
- Asset normalization and output shaping.
- Caching interfaces.
- Job/status integration.

4. `workflow-job-exec-ts`
- In-process execution.
- Worker thread execution.
- Subprocess execution.
- Docker execution compatibility.

5. `workflow-protocol-ts`
- Shared message types: `JobUpdate`, `NodeUpdate`, `EdgeUpdate`, `OutputUpdate`, etc.
- Graph/edge/node contracts.
- Serialization compatibility for JSON and MessagePack.

6. `workflow-state-ts`
- Job state persistence.
- Node state table integration.
- Resume/recovery orchestration.
- Lease/heartbeat logic.

7. `workflow-parity-harness`
- Dual-run comparison framework.
- Golden trace playback.
- Differential assertions for message stream parity.

---

**Behavioral Parity Contract**
1. Execution model
- One actor task per node.
- No centralized scheduler loop.
- Output nodes update collected outputs and emit `OutputUpdate`.

2. Input semantics
- `on_any` and `zip_all` parity.
- Sticky input behavior for non-inherently-streaming sources in zip mode.
- Multi-edge list aggregation behavior for `list[T]` targets.

3. Streaming semantics
- Streaming propagation across data edges.
- Control edges excluded from streaming propagation.
- EOS propagation on all downstream edges for completion/error/cancel paths.

4. Control semantics
- `__control__` event channel support.
- Targeted and broadcast control dispatch.
- Legacy `__control_output__` compatibility during transition.

5. Backpressure and ordering
- Per-handle FIFO.
- Arrival-order multiplexing for `iter_any`.
- Per-handle buffer limits and producer blocking behavior.

6. Lifecycle and status
- Job transitions: running/completed/failed/cancelled/suspended.
- Node transitions: running/completed/error/suspended.
- Cleanup behavior and drained edge notifications.

---

**Execution Plan (Detailed Phases)**

1. Phase 0: Spec Freeze and Parity Definition (2 weeks)
- Create explicit TS rewrite spec from observed Python behavior and tests.
- Convert critical scenarios into language-agnostic parity cases from workflow tests.
- Define strict compatibility matrix for:
  - actor behavior
  - inbox semantics
  - control edges
  - job lifecycle
  - message ordering
- Deliverables:
  - `docs/workflow-ts-spec.md`
  - `docs/workflow-ts-parity-matrix.md`
  - Golden message traces for high-priority scenarios
- Exit criteria:
  - Approved spec with no “undefined behavior” gaps for core paths.

2. Phase 1: TS Foundations and Protocol Package (2 weeks)
- Add TS workspace and strict build/test tooling.
- Create `workflow-protocol-ts` with type definitions mapped from Python models.
- Implement JSON/MessagePack encode/decode compatibility tests against current protocol docs.
- Add contract tests for message discriminator `type`.
- Deliverables:
  - `package.json`/workspace setup
  - protocol package with schema tests
- Exit criteria:
  - Protocol types compile strict.
  - Snapshot compatibility with current WebSocket payloads.

3. Phase 2: Graph and Inbox Kernel (3 weeks)
- Implement graph model and edge/node lookup primitives.
- Implement graph validation equivalent to Python runner safeguards.
- Implement `NodeInbox` with:
  - per-handle buffers
  - upstream counts
  - EOS
  - `iterInput` and `iterAny`
  - metadata envelope support
  - backpressure limits
- Build deterministic tests mirroring inbox scenario tests.
- Deliverables:
  - `Graph`, `Edge`, `NodeInbox` core modules
  - inbox test suite at parity with Python behavior
- Exit criteria:
  - All inbox/backpressure/eos ordering tests pass.

4. Phase 3: Actor Runtime Core (4 weeks)
- Implement `NodeActor` execution matrix:
  - buffered mode
  - streaming input mode
  - streaming output batched mode
  - controlled mode
- Implement sync modes:
  - `on_any`
  - `zip_all` including sticky logic
- Implement list aggregation path for multi-edge list inputs.
- Implement downstream EOS and drained edge handling.
- Deliverables:
  - TS actor runtime package
  - actor-mode parity tests mapped from E2E scenarios
- Exit criteria:
  - Parity tests pass for core actor scenarios.

5. Phase 4: WorkflowRunner Orchestration (4 weeks)
- Implement TS `WorkflowRunner` orchestration:
  - graph init/validation
  - inbox initialization
  - input dispatcher
  - actor spawn/await
  - completion detection race guard
  - send_messages routing and edge counters
- Implement control event dispatch and legacy compatibility mode.
- Implement runner outputs collection and output node behavior.
- Deliverables:
  - TS runner API equivalent to `run_workflow` consumption pattern
  - runner integration tests
- Exit criteria:
  - Runner/actor/inbox integrated with message parity for targeted scenarios.

6. Phase 5: Node SDK and Registry (5 weeks)
- Implement TS `BaseNode` equivalent:
  - property metadata
  - assignment and validation
  - outputs declaration
  - lifecycle hooks
  - `sync_mode`
  - streaming flags
- Implement node registry and resolution.
- Replace dynamic Python import behavior with explicit registration manifest.
- Port test-helper nodes first for parity harness.
- Deliverables:
  - TS node SDK
  - node registry with static manifests
  - ported test helper nodes
- Exit criteria:
  - All parity scenarios can run entirely in TS using helper nodes.

7. Phase 6: Processing Context, Assets, and Caching (4 weeks)
- Implement TS `ProcessingContext` equivalent:
  - message queue APIs
  - cache get/set
  - output normalization
  - client-safe asset sanitization
- Build adapter interfaces for storage, API client, and workspace resolution.
- Ensure `OutputUpdate` and final result payload parity for assets.
- Deliverables:
  - context/runtime package
  - asset normalization module
  - integration tests for output materialization
- Exit criteria:
  - Output payloads match expected contract and client behavior.

8. Phase 7: Job Execution + State/Recovery (deferred – pure execution engine goal)
- This phase is intentionally out of scope for the pure execution engine.
- State persistence, recovery, and lease/heartbeat logic belong in a separate
  infrastructure layer and are decoupled from the core runtime.
- The pure TS execution engine (Phases 1–6) can run standalone without any
  job-state backing store.

9. Phase 8: Dual-Run Shadow and Diff Harness (3 weeks)
- Run TS and Python engines side by side for selected workloads.
- Compare message streams, statuses, and outputs with tolerance rules.
- Add drift dashboard and failure categorization:
  - protocol drift
  - ordering drift
  - output drift
  - timing-only drift
- Deliverables:
  - automated shadow runner
  - diff reports in CI
- Exit criteria:
  - Drift below agreed threshold for staged traffic.

10. Phase 9: Canary and Cutover (3 weeks)
- Feature flag execution engine selection.
- Canary by workflow category and user cohort.
- Rollout gates:
  - error rate
  - completion rate
  - median/p95 runtime
  - output parity score
- Remove Python runtime dependency from prod path after stable window.
- Deliverables:
  - cutover runbook
  - rollback runbook
  - deprecation plan for Python runner
- Exit criteria:
  - TS runner default for all traffic with rollback confidence.

11. Phase 10: Python Decommission (2 weeks)
- Remove obsolete Python runner paths and migration shims.
- Archive parity fixtures and maintain TS contract tests.
- Update docs and operational tooling.
- Deliverables:
  - cleaned codebase
  - updated docs
- Exit criteria:
  - No production dependency on Python runner.

---

**Testing Strategy**
1. Unit tests
- Inbox, sync modes, control dispatch, list aggregation, graph validation, property assignment.

2. Scenario tests
- Direct ports of high-priority workflows from existing E2E runner/actor tests.

3. Contract tests
- WebSocket payload schema and field-level compatibility.
- Binary and MessagePack compatibility.

4. Differential tests
- Same graph+inputs executed on Python and TS.
- Assert message sequence equivalence with explicit tolerance policy.

5. Performance tests
- Throughput under fan-out/fan-in.
- Backpressure behavior under load.
- Memory stability under streaming workloads.

6. Resilience tests
- Forced cancellation mid-run.
- Controller failure mid-stream.
- Recovery after process restart.
- Lease contention and stale ownership.

---

**Migration Controls**
1. Feature flags
- `runner.engine = python|ts`
- `runner.shadow = on|off`
- `runner.control_legacy = on|off`

2. Rollback
- Instant fallback to Python engine flag.
- Preserve message contract so clients need no rollback changes.

3. Data migration
- Keep existing DB tables first.
- Introduce additive fields only.
- No destructive schema change until full cutover.

---

**Risks and Mitigations**
1. Behavior drift in edge cases
- Mitigation: phase 0 spec freeze + differential harness + golden traces.

2. Dynamic node loading mismatch
- Mitigation: explicit registration manifest and node compatibility audit.

3. State/recovery regressions
- Mitigation: dedicated recovery phase before canary; fail-safe fallback.

4. Performance regressions under streaming
- Mitigation: benchmark gates in CI and rollout SLO gates.

5. Scope explosion from node ecosystem
- Mitigation: decouple kernel rewrite from node porting; prioritize by usage.

---

**Definition of Done**
1. TS runner handles 100% production workflow traffic.
2. Python runner removed from default codepath.
3. Message/API compatibility maintained.
4. Recovery, suspension, cancellation, and streaming semantics validated.
5. Operational SLOs meet or exceed current baseline for 30 consecutive days.
