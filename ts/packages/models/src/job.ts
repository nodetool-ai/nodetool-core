/**
 * Job model – tracks workflow execution state.
 *
 * Port of Python's `nodetool.models.job`.
 */

import type { TableSchema } from "./database-adapter.js";
import type { Row } from "./database-adapter.js";
import {
  DBModel,
  createTimeOrderedUuid,
  type IndexSpec,
  type ModelClass,
} from "./base-model.js";
import { field } from "./condition-builder.js";

// ── Types ────────────────────────────────────────────────────────────

export type JobStatus =
  | "scheduled"
  | "running"
  | "suspended"
  | "paused"
  | "completed"
  | "failed"
  | "cancelled"
  | "recovering";

// ── Schema ───────────────────────────────────────────────────────────

const JOB_SCHEMA: TableSchema = {
  table_name: "jobs",
  primary_key: "id",
  columns: {
    id: { type: "string" },
    user_id: { type: "string" },
    workflow_id: { type: "string" },
    status: { type: "string" },
    name: { type: "string", optional: true },
    graph: { type: "json", optional: true },
    params: { type: "json", optional: true },
    worker_id: { type: "string", optional: true },
    heartbeat_at: { type: "datetime", optional: true },
    started_at: { type: "datetime", optional: true },
    finished_at: { type: "datetime", optional: true },
    error: { type: "string", optional: true },
    retry_count: { type: "number" },
    version: { type: "number" },
    suspended_node_id: { type: "string", optional: true },
    suspension_state_json: { type: "json", optional: true },
    created_at: { type: "datetime" },
    updated_at: { type: "datetime" },
  },
};

const JOB_INDEXES: IndexSpec[] = [
  { name: "idx_jobs_status", columns: ["status"], unique: false },
  { name: "idx_jobs_updated_at", columns: ["updated_at"], unique: false },
  { name: "idx_jobs_worker_id", columns: ["worker_id"], unique: false },
  { name: "idx_jobs_heartbeat_at", columns: ["heartbeat_at"], unique: false },
];

// ── Model ────────────────────────────────────────────────────────────

export class Job extends DBModel {
  static override schema = JOB_SCHEMA;
  static override indexes = JOB_INDEXES;

  declare id: string;
  declare user_id: string;
  declare workflow_id: string;
  declare status: JobStatus;
  declare name: string;
  declare graph: Record<string, unknown> | null;
  declare params: Record<string, unknown> | null;
  declare worker_id: string | null;
  declare heartbeat_at: string | null;
  declare started_at: string | null;
  declare finished_at: string | null;
  declare error: string | null;
  declare retry_count: number;
  declare version: number;
  declare suspended_node_id: string | null;
  declare suspension_state_json: Record<string, unknown> | null;
  declare created_at: string;
  declare updated_at: string;

  constructor(data: Row) {
    super(data);
    const now = new Date().toISOString();
    this.id ??= createTimeOrderedUuid();
    this.status ??= "scheduled";
    this.retry_count ??= 0;
    this.version ??= 1;
    this.created_at ??= now;
    this.updated_at ??= now;
    this.graph ??= null;
    this.params ??= null;
    this.worker_id ??= null;
    this.heartbeat_at ??= null;
    this.started_at ??= null;
    this.finished_at ??= null;
    this.error ??= null;
    this.suspended_node_id ??= null;
    this.suspension_state_json ??= null;
    this.name ??= "";
  }

  override beforeSave(): void {
    this.updated_at = new Date().toISOString();
  }

  // ── State transitions ────────────────────────────────────────────

  markRunning(workerId?: string): void {
    this.status = "running";
    this.started_at = new Date().toISOString();
    if (workerId) this.worker_id = workerId;
  }

  markCompleted(): void {
    this.status = "completed";
    this.finished_at = new Date().toISOString();
  }

  markFailed(error: string): void {
    this.status = "failed";
    this.error = error;
    this.finished_at = new Date().toISOString();
  }

  markCancelled(): void {
    this.status = "cancelled";
    this.finished_at = new Date().toISOString();
  }

  markSuspended(nodeId: string, state?: Record<string, unknown>): void {
    this.status = "suspended";
    this.suspended_node_id = nodeId;
    if (state) this.suspension_state_json = state;
  }

  markResumed(): void {
    this.status = "running";
    this.suspended_node_id = null;
    this.suspension_state_json = null;
  }

  markPaused(): void {
    this.status = "paused";
  }

  markRecovering(): void {
    this.status = "recovering";
  }

  updateHeartbeat(): void {
    this.heartbeat_at = new Date().toISOString();
  }

  incrementRetry(): void {
    this.retry_count += 1;
  }

  isStale(thresholdMs: number): boolean {
    if (!this.heartbeat_at) return true;
    const elapsed = Date.now() - new Date(this.heartbeat_at).getTime();
    return elapsed > thresholdMs;
  }

  // ── Static queries ───────────────────────────────────────────────

  static async paginate(
    userId: string,
    opts: {
      cursor?: string;
      limit?: number;
      status?: JobStatus;
    } = {},
  ): Promise<[Job[], string]> {
    const { limit = 50, status } = opts;
    let cond = field("user_id").equals(userId);
    if (status) cond = cond.and(field("status").equals(status));

    return (Job as unknown as ModelClass<Job>).query({
      condition: cond,
      orderBy: "updated_at",
      reverse: true,
      limit,
    });
  }
}
