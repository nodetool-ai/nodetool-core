# Workflow TS Parity Matrix

| Area | Python Baseline | TS Status | Evidence |
|---|---|---|---|
| Protocol message types | `workflows/types.py` | Implemented | `@nodetool/protocol` tests |
| Graph validation/topology | `workflows/graph.py` | Implemented | `kernel/tests/graph.test.ts` |
| Inbox + backpressure | `workflows/inbox.py` | Implemented | `kernel/tests/inbox.test.ts` |
| Actor execution modes | `workflows/actor.py` | Implemented | `kernel/tests/actor.test.ts` |
| Runner orchestration | `workflows/workflow_runner.py` | Implemented | `kernel/tests/runner.test.ts` |
| Node SDK + registry | `workflows/base_node.py` | Implemented | `node-sdk` test suite |
| Output sanitization | `sanitize_memory_uris_for_client` | Implemented | `runtime/tests/context.test.ts` |
| Asset normalization modes | `processing_context.normalize_output_value` | Partial | `runtime/tests/context.test.ts` |
| Storage adapters | asset persistence path | Implemented core adapters | runtime storage adapter tests |
| Workspace path resolution | `io/path_utils.resolve_workspace_path` | Implemented | runtime workspace tests |
| State + recovery | run state tables/lease | Deferred | Out of pure engine scope |
| Dual-run harness | shadow + diff | Partial | `@nodetool/parity-harness` |
| Canary + cutover | prod rollout controls | Not started in code | Runbooks added |
| Python decommission | removal of Python runtime path | Not started | Plan doc added |

## Drift Categories
- `protocol_drift`
- `ordering_drift`
- `output_drift`
- `timing_only_drift`

## Exit Bar for Runtime Parity
- No protocol drift on selected canary workflows.
- Output drift only for approved tolerated fields.
- Ordering drift absent for deterministic scenarios.
