/**
 * @nodetool/models – Database models and query utilities.
 *
 * Public API surface for the models package. Re-exports everything
 * consumers need to define, query and persist data models.
 */

// ── Condition Builder ────────────────────────────────────────────────
export {
  Operator,
  LogicalOperator,
  Variable,
  Condition,
  ConditionGroup,
  Field,
  ConditionBuilder,
  field,
} from "./condition-builder.js";
export type { ConditionValue } from "./condition-builder.js";

// ── Database Adapter ─────────────────────────────────────────────────
export type {
  DatabaseAdapter,
  TableSchema,
  FieldDef,
  IndexDef,
  Row,
} from "./database-adapter.js";

// ── In-memory Adapter ────────────────────────────────────────────────
export { MemoryAdapter, MemoryAdapterFactory } from "./memory-adapter.js";

// ── SQLite Adapter ──────────────────────────────────────────────────
export { SQLiteAdapter, SQLiteAdapterFactory } from "./sqlite-adapter.js";

// ── Base Model ───────────────────────────────────────────────────────
export {
  DBModel,
  ModelObserver,
  ModelChangeEvent,
  createTimeOrderedUuid,
  computeEtag,
  setGlobalAdapterResolver,
  getGlobalAdapterResolver,
} from "./base-model.js";
export type {
  ModelClass,
  ModelObserverCallback,
  IndexSpec,
  AdapterResolver,
} from "./base-model.js";

// ── Domain Models ────────────────────────────────────────────────────
export { Job } from "./job.js";
export type { JobStatus } from "./job.js";

export { Workflow } from "./workflow.js";
export type { AccessLevel, WorkflowGraph } from "./workflow.js";

export { Asset } from "./asset.js";

export { Message } from "./message.js";

export { Thread } from "./thread.js";

export { Secret } from "./secret.js";

export { Workspace } from "./workspace.js";

export { OAuthCredential } from "./oauth-credential.js";

export { RunNodeState } from "./run-node-state.js";
export type { NodeStatus } from "./run-node-state.js";

export { RunEvent } from "./run-event.js";
export type { EventType } from "./run-event.js";

export { RunLease } from "./run-lease.js";

export { Prediction } from "./prediction.js";
export type {
  AggregateResult,
  ProviderAggregateResult,
  ModelAggregateResult,
} from "./prediction.js";
