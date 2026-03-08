/**
 * Protocol-level parity checker.
 *
 * Compares Python WebSocket/streaming message schemas (exported as JSON by
 * ``scripts/export_parity_snapshot.py messages``) against the static
 * TypeScript manifest derived from ``protocol/src/messages.ts``.
 *
 * TypeScript interfaces are NOT runtime-inspectable, so we maintain a
 * hand-curated ``TS_MESSAGE_MANIFEST`` that mirrors every exported interface
 * that belongs to the ``ProcessingMessage`` union.
 *
 * Type mapping rules (Python field type → canonical string):
 *   string / str          → "string"
 *   number / int / float  → "number"
 *   boolean / bool        → "boolean"
 *   object / dict / any   → "json"
 *   arrays / Uint8Array   → "json"
 *   string | null / opt   → "string"  (required=false handles the null)
 */

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface MessageFieldDef {
  type: string;
  required: boolean;
}

export interface MessageSchema {
  /** The literal value of the "type" discriminator field, e.g. "job_update" */
  typeDiscriminator: string;
  fields: Record<string, MessageFieldDef>;
}

export type MessageSchemaMap = Record<string, MessageSchema>;

export interface ProtocolDrift {
  /** Python class name, e.g. "JobUpdate" */
  message: string;
  field: string;
  severity: "error" | "warning" | "info";
  message_text: string;
  python?: unknown;
  ts?: unknown;
}

export interface ProtocolParityReport {
  pass: boolean;
  drifts: ProtocolDrift[];
  summary: {
    messagesChecked: number;
    fieldsChecked: number;
    errors: number;
    warnings: number;
  };
}

// ---------------------------------------------------------------------------
// Static manifest derived from protocol/src/messages.ts
// Covers every member of the ProcessingMessage union.
// ---------------------------------------------------------------------------

export const TS_MESSAGE_MANIFEST: MessageSchemaMap = {
  // ── JobUpdate ────────────────────────────────────────────────────────────
  JobUpdate: {
    typeDiscriminator: "job_update",
    fields: {
      type: { type: "string", required: true },
      status: { type: "string", required: true },
      job_id: { type: "string", required: false },
      workflow_id: { type: "string", required: false },
      message: { type: "string", required: false },
      result: { type: "json", required: false },
      error: { type: "string", required: false },
      traceback: { type: "string", required: false },
      run_state: { type: "json", required: false },
      duration: { type: "number", required: false },
    },
  },

  // ── NodeUpdate ───────────────────────────────────────────────────────────
  NodeUpdate: {
    typeDiscriminator: "node_update",
    fields: {
      type: { type: "string", required: true },
      node_id: { type: "string", required: true },
      node_name: { type: "string", required: true },
      node_type: { type: "string", required: true },
      status: { type: "string", required: true },
      error: { type: "string", required: false },
      result: { type: "json", required: false },
      properties: { type: "json", required: false },
      workflow_id: { type: "string", required: false },
    },
  },

  // ── NodeProgress ─────────────────────────────────────────────────────────
  NodeProgress: {
    typeDiscriminator: "node_progress",
    fields: {
      type: { type: "string", required: true },
      node_id: { type: "string", required: true },
      progress: { type: "number", required: true },
      total: { type: "number", required: true },
      chunk: { type: "string", required: false },
      workflow_id: { type: "string", required: false },
    },
  },

  // ── EdgeUpdate ───────────────────────────────────────────────────────────
  EdgeUpdate: {
    typeDiscriminator: "edge_update",
    fields: {
      type: { type: "string", required: true },
      workflow_id: { type: "string", required: true },
      edge_id: { type: "string", required: true },
      status: { type: "string", required: true },
      counter: { type: "number", required: false },
    },
  },

  // ── OutputUpdate ─────────────────────────────────────────────────────────
  OutputUpdate: {
    typeDiscriminator: "output_update",
    fields: {
      type: { type: "string", required: true },
      node_id: { type: "string", required: true },
      node_name: { type: "string", required: true },
      output_name: { type: "string", required: true },
      value: { type: "json", required: true },
      output_type: { type: "string", required: true },
      metadata: { type: "json", required: true },
      workflow_id: { type: "string", required: false },
    },
  },

  // ── PreviewUpdate ────────────────────────────────────────────────────────
  PreviewUpdate: {
    typeDiscriminator: "preview_update",
    fields: {
      type: { type: "string", required: true },
      node_id: { type: "string", required: true },
      value: { type: "json", required: true },
    },
  },

  // ── SaveUpdate ───────────────────────────────────────────────────────────
  SaveUpdate: {
    typeDiscriminator: "save_update",
    fields: {
      type: { type: "string", required: true },
      node_id: { type: "string", required: true },
      name: { type: "string", required: true },
      value: { type: "json", required: true },
      output_type: { type: "string", required: true },
      metadata: { type: "json", required: true },
    },
  },

  // ── BinaryUpdate ─────────────────────────────────────────────────────────
  BinaryUpdate: {
    typeDiscriminator: "binary_update",
    fields: {
      type: { type: "string", required: true },
      node_id: { type: "string", required: true },
      output_name: { type: "string", required: true },
      binary: { type: "json", required: true },
    },
  },

  // ── LogUpdate ────────────────────────────────────────────────────────────
  LogUpdate: {
    typeDiscriminator: "log_update",
    fields: {
      type: { type: "string", required: true },
      node_id: { type: "string", required: true },
      node_name: { type: "string", required: true },
      content: { type: "string", required: true },
      severity: { type: "string", required: true },
      workflow_id: { type: "string", required: false },
    },
  },

  // ── Notification ─────────────────────────────────────────────────────────
  Notification: {
    typeDiscriminator: "notification",
    fields: {
      type: { type: "string", required: true },
      node_id: { type: "string", required: true },
      content: { type: "string", required: true },
      severity: { type: "string", required: true },
      workflow_id: { type: "string", required: false },
    },
  },

  // ── Error (Python name) / ErrorMessage (TS name) ─────────────────────────
  Error: {
    typeDiscriminator: "error",
    fields: {
      type: { type: "string", required: true },
      message: { type: "string", required: true },
      thread_id: { type: "string", required: false },
      workflow_id: { type: "string", required: false },
    },
  },

  // ── ToolCallUpdate ───────────────────────────────────────────────────────
  ToolCallUpdate: {
    typeDiscriminator: "tool_call_update",
    fields: {
      type: { type: "string", required: true },
      node_id: { type: "string", required: false },
      thread_id: { type: "string", required: false },
      workflow_id: { type: "string", required: false },
      tool_call_id: { type: "string", required: false },
      name: { type: "string", required: true },
      args: { type: "json", required: true },
      message: { type: "string", required: false },
      step_id: { type: "string", required: false },
      agent_execution_id: { type: "string", required: false },
    },
  },

  // ── ToolResultUpdate ─────────────────────────────────────────────────────
  ToolResultUpdate: {
    typeDiscriminator: "tool_result_update",
    fields: {
      type: { type: "string", required: true },
      node_id: { type: "string", required: true },
      thread_id: { type: "string", required: false },
      workflow_id: { type: "string", required: false },
      result: { type: "json", required: true },
    },
  },

  // ── TaskUpdate ───────────────────────────────────────────────────────────
  TaskUpdate: {
    typeDiscriminator: "task_update",
    fields: {
      type: { type: "string", required: true },
      node_id: { type: "string", required: false },
      thread_id: { type: "string", required: false },
      workflow_id: { type: "string", required: false },
      task: { type: "json", required: true },
      step: { type: "json", required: false },
      event: { type: "json", required: true },
    },
  },

  // ── StepResult ───────────────────────────────────────────────────────────
  StepResult: {
    typeDiscriminator: "step_result",
    fields: {
      type: { type: "string", required: true },
      step: { type: "json", required: true },
      result: { type: "json", required: true },
      error: { type: "string", required: false },
      is_task_result: { type: "boolean", required: false },
      thread_id: { type: "string", required: false },
      workflow_id: { type: "string", required: false },
    },
  },

  // ── PlanningUpdate ───────────────────────────────────────────────────────
  PlanningUpdate: {
    typeDiscriminator: "planning_update",
    fields: {
      type: { type: "string", required: true },
      node_id: { type: "string", required: false },
      thread_id: { type: "string", required: false },
      workflow_id: { type: "string", required: false },
      phase: { type: "string", required: true },
      status: { type: "string", required: true },
      content: { type: "string", required: false },
    },
  },

  // ── Chunk ────────────────────────────────────────────────────────────────
  Chunk: {
    typeDiscriminator: "chunk",
    fields: {
      type: { type: "string", required: true },
      node_id: { type: "string", required: false },
      thread_id: { type: "string", required: false },
      workflow_id: { type: "string", required: false },
      content_type: { type: "string", required: false },
      content: { type: "string", required: false },
      content_metadata: { type: "json", required: false },
      done: { type: "boolean", required: false },
      thinking: { type: "boolean", required: false },
    },
  },

  // ── Prediction ───────────────────────────────────────────────────────────
  Prediction: {
    typeDiscriminator: "prediction",
    fields: {
      type: { type: "string", required: true },
      id: { type: "string", required: true },
      user_id: { type: "string", required: true },
      node_id: { type: "string", required: true },
      workflow_id: { type: "string", required: false },
      provider: { type: "string", required: false },
      model: { type: "string", required: false },
      version: { type: "string", required: false },
      node_type: { type: "string", required: false },
      status: { type: "string", required: true },
      params: { type: "json", required: false },
      data: { type: "json", required: false },
      cost: { type: "number", required: false },
      logs: { type: "string", required: false },
      error: { type: "string", required: false },
      duration: { type: "number", required: false },
      created_at: { type: "string", required: false },
      started_at: { type: "string", required: false },
      completed_at: { type: "string", required: false },
    },
  },
};

// ---------------------------------------------------------------------------
// Comparison logic
// ---------------------------------------------------------------------------

/**
 * Compare Python message schemas (from the export script) against the static
 * TypeScript manifest.
 *
 * Matching is done by ``type_discriminator`` ↔ ``typeDiscriminator``.
 */
export function compareProtocolMessages(
  pythonSchemas: MessageSchemaMap,
  tsManifest: MessageSchemaMap,
): ProtocolParityReport {
  const drifts: ProtocolDrift[] = [];
  let fieldsChecked = 0;

  // Build a lookup from typeDiscriminator → [tsName, tsSchema]
  const tsByDiscriminator = new Map<string, [string, MessageSchema]>();
  for (const [name, schema] of Object.entries(tsManifest)) {
    tsByDiscriminator.set(schema.typeDiscriminator, [name, schema]);
  }

  // Build a lookup from typeDiscriminator for Python side
  const pyByDiscriminator = new Map<string, [string, MessageSchema]>();
  for (const [name, schema] of Object.entries(pythonSchemas)) {
    pyByDiscriminator.set(schema.typeDiscriminator, [name, schema]);
  }

  // Check each Python message has a TS counterpart
  for (const [pyName, pySchema] of Object.entries(pythonSchemas)) {
    const tsEntry = tsByDiscriminator.get(pySchema.typeDiscriminator);
    if (!tsEntry) {
      drifts.push({
        message: pyName,
        field: "(message)",
        severity: "error",
        message_text: `Python message "${pyName}" (type "${pySchema.typeDiscriminator}") has no TypeScript counterpart`,
      });
      continue;
    }

    const [tsName, tsSchema] = tsEntry;
    const pyFields = pySchema.fields;
    const tsFields = tsSchema.fields;

    // For each Python field, check it exists in TS with compatible type
    for (const [fieldName, pyDef] of Object.entries(pyFields)) {
      fieldsChecked++;

      if (!(fieldName in tsFields)) {
        // Missing field: severity depends on whether Python considers it required
        const severity = pyDef.required ? "error" : "warning";
        drifts.push({
          message: pyName,
          field: fieldName,
          severity,
          message_text: pyDef.required
            ? `Required Python field "${fieldName}" is missing in TypeScript`
            : `Optional Python field "${fieldName}" is missing in TypeScript`,
          python: pyDef,
        });
        continue;
      }

      const tsDef = tsFields[fieldName];

      // Type mismatch
      if (pyDef.type !== tsDef.type) {
        drifts.push({
          message: pyName,
          field: fieldName,
          severity: "warning",
          message_text: `Type mismatch for field "${fieldName}"`,
          python: pyDef.type,
          ts: tsDef.type,
        });
      }
    }

    // Fields in TS but not in Python → info
    for (const fieldName of Object.keys(tsFields)) {
      if (!(fieldName in pyFields)) {
        drifts.push({
          message: `${pyName}/${tsName}`,
          field: fieldName,
          severity: "info",
          message_text: `Field "${fieldName}" exists in TypeScript but missing in Python`,
          ts: tsFields[fieldName],
        });
      }
    }
  }

  // Python messages not represented on TS side (already covered above as errors).
  // Additionally report TS messages with no Python counterpart as info.
  for (const [tsName, tsSchema] of Object.entries(tsManifest)) {
    if (!pyByDiscriminator.has(tsSchema.typeDiscriminator)) {
      drifts.push({
        message: tsName,
        field: "(message)",
        severity: "info",
        message_text: `TypeScript message "${tsName}" (type "${tsSchema.typeDiscriminator}") has no Python counterpart`,
        ts: tsSchema.typeDiscriminator,
      });
    }
  }

  const errors = drifts.filter((d) => d.severity === "error").length;
  const warnings = drifts.filter((d) => d.severity === "warning").length;

  return {
    pass: errors === 0,
    drifts,
    summary: {
      messagesChecked: Object.keys(pythonSchemas).length,
      fieldsChecked,
      errors,
      warnings,
    },
  };
}
