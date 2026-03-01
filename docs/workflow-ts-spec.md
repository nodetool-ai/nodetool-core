# Workflow TS Spec

## Scope
- TS runtime behavior for graph execution, actors, inbox semantics, and output message contracts.
- Compatibility target: existing Python workflow message and graph format contracts.

## Core Execution Contract
- One actor task per executable node.
- Source nodes (no inbound data edges) dispatch params directly.
- `WorkflowRunner.run()` returns final status, outputs, and emitted messages.

## Input and Sync Semantics
- Supported sync modes: `on_any`, `zip_all`.
- `zip_all` uses sticky values for non-streaming sources.
- Multi-edge list inputs aggregate values in arrival order.

## Streaming and EOS
- Data edges propagate streaming values.
- Control edges are excluded from data streaming.
- Downstream EOS is propagated on completion/error/cancel.

## Control Edges
- `__control__` channel supported for targeted/broadcast control events.
- Legacy compatibility path remains available for `__control_output__` during migration.

## Output and Client Safety
- `output_update` emitted from output nodes.
- Asset-like payloads support normalization modes:
  - `python`
  - `raw`
  - `data_uri`
  - `storage_url`
  - `workspace`
- Client sanitization rewrites `memory://` only for asset-shaped payloads.

## Storage Adapters
- `InMemoryStorageAdapter`
- `FileStorageAdapter`
- `S3StorageAdapter` (SDK-agnostic via injected client interface)

## Workspace Path Resolution
- Supports `/workspace/...`, `workspace/...`, absolute, and relative paths.
- Enforces traversal protection; resolved paths must remain within workspace root.

## Out of Scope
- Job state persistence, leases, and recovery state tables (separate infra layer).
- Production traffic routing and rollout gates.
