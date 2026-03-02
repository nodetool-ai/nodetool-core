/**
 * Workflow model – stores DAG-based workflow definitions.
 *
 * Port of Python's `nodetool.models.workflow`.
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

export type AccessLevel = "private" | "public";

export interface WorkflowGraph {
  nodes: Record<string, unknown>[];
  edges: Record<string, unknown>[];
}

// ── Schema ───────────────────────────────────────────────────────────

const WORKFLOW_SCHEMA: TableSchema = {
  table_name: "workflows",
  primary_key: "id",
  columns: {
    id: { type: "string" },
    user_id: { type: "string" },
    name: { type: "string" },
    description: { type: "string", optional: true },
    tags: { type: "json", optional: true },
    graph: { type: "json" },
    settings: { type: "json", optional: true },
    access: { type: "string" },
    created_at: { type: "datetime" },
    updated_at: { type: "datetime" },
  },
};

const WORKFLOW_INDEXES: IndexSpec[] = [
  { name: "idx_workflows_user_id", columns: ["user_id"], unique: false },
  { name: "idx_workflows_access", columns: ["access"], unique: false },
];

// ── Model ────────────────────────────────────────────────────────────

export class Workflow extends DBModel {
  static override schema = WORKFLOW_SCHEMA;
  static override indexes = WORKFLOW_INDEXES;

  declare id: string;
  declare user_id: string;
  declare name: string;
  declare description: string;
  declare tags: string[];
  declare graph: WorkflowGraph;
  declare settings: Record<string, unknown> | null;
  declare access: AccessLevel;
  declare created_at: string;
  declare updated_at: string;

  constructor(data: Row) {
    super(data);
    const now = new Date().toISOString();
    this.id ??= createTimeOrderedUuid();
    this.name ??= "";
    this.description ??= "";
    this.tags ??= [];
    this.graph ??= { nodes: [], edges: [] };
    this.settings ??= null;
    this.access ??= "private";
    this.created_at ??= now;
    this.updated_at ??= now;
  }

  override beforeSave(): void {
    this.updated_at = new Date().toISOString();
  }

  // ── Static queries ───────────────────────────────────────────────

  /** Find a workflow by id, respecting ownership or public access. */
  static async find(
    userId: string,
    workflowId: string,
  ): Promise<Workflow | null> {
    const wf = await (Workflow as unknown as ModelClass<Workflow>).get(
      workflowId,
    );
    if (!wf) return null;
    if (wf.user_id === userId || wf.access === "public") return wf;
    return null;
  }

  /** Paginate workflows for a user (includes their own + public). */
  static async paginate(
    userId: string,
    opts: {
      limit?: number;
      access?: AccessLevel;
    } = {},
  ): Promise<[Workflow[], string]> {
    const { limit = 50, access } = opts;
    let cond = field("user_id").equals(userId);
    if (access) cond = cond.and(field("access").equals(access));

    return (Workflow as unknown as ModelClass<Workflow>).query({
      condition: cond,
      orderBy: "updated_at",
      reverse: true,
      limit,
    });
  }
}
