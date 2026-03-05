/**
 * Asset model – digital asset storage with folder hierarchy.
 *
 * Port of Python's `nodetool.models.asset`.
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

// ── Schema ───────────────────────────────────────────────────────────

const ASSET_SCHEMA: TableSchema = {
  table_name: "assets",
  primary_key: "id",
  columns: {
    id: { type: "string" },
    user_id: { type: "string" },
    parent_id: { type: "string", optional: true },
    file_id: { type: "string", optional: true },
    name: { type: "string" },
    content_type: { type: "string" },
    size: { type: "number", optional: true },
    duration: { type: "number", optional: true },
    metadata: { type: "json", optional: true },
    workflow_id: { type: "string", optional: true },
    node_id: { type: "string", optional: true },
    job_id: { type: "string", optional: true },
    created_at: { type: "datetime" },
    updated_at: { type: "datetime" },
  },
};

const ASSET_INDEXES: IndexSpec[] = [
  {
    name: "idx_assets_user_parent",
    columns: ["user_id", "parent_id"],
    unique: false,
  },
];

// ── Model ────────────────────────────────────────────────────────────

export class Asset extends DBModel {
  static override schema = ASSET_SCHEMA;
  static override indexes = ASSET_INDEXES;

  declare id: string;
  declare user_id: string;
  declare parent_id: string | null;
  declare file_id: string | null;
  declare name: string;
  declare content_type: string;
  declare size: number | null;
  declare duration: number | null;
  declare metadata: Record<string, unknown> | null;
  declare workflow_id: string | null;
  declare node_id: string | null;
  declare job_id: string | null;
  declare created_at: string;
  declare updated_at: string;

  constructor(data: Row) {
    super(data);
    const now = new Date().toISOString();
    this.id ??= createTimeOrderedUuid();
    this.name ??= "";
    this.content_type ??= "application/octet-stream";
    this.parent_id ??= null;
    this.file_id ??= null;
    this.size ??= null;
    this.duration ??= null;
    this.metadata ??= null;
    this.workflow_id ??= null;
    this.node_id ??= null;
    this.job_id ??= null;
    this.created_at ??= now;
    this.updated_at ??= now;
  }

  override beforeSave(): void {
    this.updated_at = new Date().toISOString();
  }

  // ── Computed properties ──────────────────────────────────────────

  get isFolder(): boolean {
    return this.content_type === "folder";
  }

  get fileExtension(): string {
    const dot = this.name.lastIndexOf(".");
    return dot >= 0 ? this.name.slice(dot + 1).toLowerCase() : "";
  }

  get hasThumbnail(): boolean {
    const type = this.content_type;
    return type.startsWith("image/") || type.startsWith("video/");
  }

  // ── Static queries ───────────────────────────────────────────────

  /** Find an asset by id, scoped to the user. */
  static async find(
    userId: string,
    assetId: string,
  ): Promise<Asset | null> {
    const asset = (await (Asset as unknown as ModelClass<Asset>).get(assetId)) as Asset | null;
    if (!asset || asset.user_id !== userId) return null;
    return asset;
  }

  /** List assets in a folder. */
  static async paginate(
    userId: string,
    opts: {
      parentId?: string | null;
      contentType?: string;
      workflowId?: string;
      nodeId?: string;
      jobId?: string;
      limit?: number;
    } = {},
  ): Promise<[Asset[], string]> {
    const { parentId, contentType, workflowId, nodeId, jobId, limit = 50 } = opts;
    let cond = field("user_id").equals(userId);
    if (parentId !== undefined) {
      cond = cond.and(field("parent_id").equals(parentId));
    }
    if (contentType) {
      cond = cond.and(field("content_type").equals(contentType));
    }
    if (workflowId) {
      cond = cond.and(field("workflow_id").equals(workflowId));
    }
    if (nodeId) {
      cond = cond.and(field("node_id").equals(nodeId));
    }
    if (jobId) {
      cond = cond.and(field("job_id").equals(jobId));
    }

    return (Asset as unknown as ModelClass<Asset>).query({
      condition: cond,
      orderBy: "created_at",
      reverse: true,
      limit,
    });
  }

  /** Get children of a folder. */
  static async getChildren(
    userId: string,
    parentId: string,
    limit = 100,
  ): Promise<Asset[]> {
    const [assets] = await Asset.paginate(userId, { parentId, limit });
    return assets;
  }
}
