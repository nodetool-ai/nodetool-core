/**
 * Message model – conversation messages with tool call support.
 *
 * Port of Python's `nodetool.models.message`.
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

const MESSAGE_SCHEMA: TableSchema = {
  table_name: "messages",
  primary_key: "id",
  columns: {
    id: { type: "string" },
    user_id: { type: "string" },
    thread_id: { type: "string" },
    role: { type: "string" },
    content: { type: "string", optional: true },
    tool_calls: { type: "json", optional: true },
    input_files: { type: "json", optional: true },
    output_files: { type: "json", optional: true },
    provider: { type: "string", optional: true },
    model: { type: "string", optional: true },
    cost: { type: "number", optional: true },
    workflow_id: { type: "string", optional: true },
    created_at: { type: "datetime" },
    updated_at: { type: "datetime" },
  },
};

const MESSAGE_INDEXES: IndexSpec[] = [
  {
    name: "idx_messages_thread_id",
    columns: ["thread_id"],
    unique: false,
  },
];

// ── Model ────────────────────────────────────────────────────────────

export class Message extends DBModel {
  static override schema = MESSAGE_SCHEMA;
  static override indexes = MESSAGE_INDEXES;

  declare id: string;
  declare user_id: string;
  declare thread_id: string;
  declare role: string;
  declare content: string | null;
  declare tool_calls: unknown[] | null;
  declare input_files: unknown[] | null;
  declare output_files: unknown[] | null;
  declare provider: string | null;
  declare model: string | null;
  declare cost: number | null;
  declare workflow_id: string | null;
  declare created_at: string;
  declare updated_at: string;

  constructor(data: Row) {
    super(data);
    const now = new Date().toISOString();
    this.id ??= createTimeOrderedUuid();
    this.role ??= "user";
    this.content ??= null;
    this.tool_calls ??= null;
    this.input_files ??= null;
    this.output_files ??= null;
    this.provider ??= null;
    this.model ??= null;
    this.cost ??= null;
    this.workflow_id ??= null;
    this.created_at ??= now;
    this.updated_at ??= now;
  }

  override beforeSave(): void {
    this.updated_at = new Date().toISOString();
  }

  // ── Static queries ───────────────────────────────────────────────

  /** Paginate messages in a thread. */
  static async paginate(
    threadId: string,
    opts: { limit?: number } = {},
  ): Promise<[Message[], string]> {
    const { limit = 50 } = opts;
    const cond = field("thread_id").equals(threadId);

    return (Message as unknown as ModelClass<Message>).query({
      condition: cond,
      orderBy: "created_at",
      limit,
    });
  }
}
