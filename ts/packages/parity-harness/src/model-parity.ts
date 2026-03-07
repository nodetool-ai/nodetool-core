/**
 * Model-level parity checker.
 *
 * Compares Python model schemas (exported as JSON by
 * ``scripts/export_parity_snapshot.py``) against TypeScript model schemas
 * so we can detect drift at the DB-model level.
 *
 * Each schema is expressed in a normalised format:
 *
 * ```ts
 * interface ModelSchema {
 *   table_name: string;
 *   primary_key: string;            // default "id"
 *   columns: Record<string, { type: string; optional: boolean }>;
 *   indexes: { name: string; columns: string[]; unique: boolean }[];
 * }
 * ```
 */

// ── Public types ─────────────────────────────────────────────────────

export interface ColumnDef {
  type: string;
  optional: boolean;
}

export interface IndexDef {
  name: string;
  columns: string[];
  unique: boolean;
}

export interface ModelSchema {
  table_name: string;
  primary_key: string;
  columns: Record<string, ColumnDef>;
  indexes: IndexDef[];
}

export type ModelSchemaMap = Record<string, ModelSchema>;

export type DriftSeverity = "error" | "warning" | "info";

export interface ModelDrift {
  model: string;
  field: string;
  severity: DriftSeverity;
  message: string;
  python?: unknown;
  ts?: unknown;
}

export interface ModelParityReport {
  pass: boolean;
  drifts: ModelDrift[];
  summary: {
    modelsChecked: number;
    columnsChecked: number;
    errors: number;
    warnings: number;
  };
}

// ── Comparison logic ─────────────────────────────────────────────────

export function compareModelSchemas(
  pythonSchemas: ModelSchemaMap,
  tsSchemas: ModelSchemaMap,
): ModelParityReport {
  const drifts: ModelDrift[] = [];
  let columnsChecked = 0;

  // Build a lookup by table_name for TS schemas
  const tsByTable = new Map<string, [string, ModelSchema]>();
  for (const [name, schema] of Object.entries(tsSchemas)) {
    tsByTable.set(schema.table_name, [name, schema]);
  }

  // Build a lookup by table_name for Python schemas
  const pyByTable = new Map<string, [string, ModelSchema]>();
  for (const [name, schema] of Object.entries(pythonSchemas)) {
    pyByTable.set(schema.table_name, [name, schema]);
  }

  // Check each Python model has a TS counterpart
  for (const [pyName, pySchema] of Object.entries(pythonSchemas)) {
    const tsEntry = tsByTable.get(pySchema.table_name);
    if (!tsEntry) {
      drifts.push({
        model: pyName,
        field: "(model)",
        severity: "error",
        message: `Python model "${pyName}" (table "${pySchema.table_name}") has no TypeScript counterpart`,
      });
      continue;
    }

    const [tsName, tsSchema] = tsEntry;

    // Compare primary keys
    if (pySchema.primary_key !== tsSchema.primary_key) {
      drifts.push({
        model: pyName,
        field: "primary_key",
        severity: "error",
        message: `Primary key mismatch`,
        python: pySchema.primary_key,
        ts: tsSchema.primary_key,
      });
    }

    // Compare columns
    const pyColumns = new Set(Object.keys(pySchema.columns));
    const tsColumns = new Set(Object.keys(tsSchema.columns));

    for (const col of pyColumns) {
      columnsChecked++;
      if (!tsColumns.has(col)) {
        drifts.push({
          model: pyName,
          field: col,
          severity: "warning",
          message: `Column "${col}" exists in Python but missing in TypeScript`,
        });
        continue;
      }

      const pyCol = pySchema.columns[col];
      const tsCol = tsSchema.columns[col];

      // Type comparison
      if (pyCol.type !== tsCol.type) {
        drifts.push({
          model: pyName,
          field: col,
          severity: "warning",
          message: `Type mismatch for column "${col}"`,
          python: pyCol.type,
          ts: tsCol.type,
        });
      }

      // Optionality comparison
      if (pyCol.optional !== tsCol.optional) {
        drifts.push({
          model: pyName,
          field: col,
          severity: "info",
          message: `Optionality mismatch for column "${col}"`,
          python: pyCol.optional,
          ts: tsCol.optional,
        });
      }
    }

    // Check for TS columns not in Python
    for (const col of tsColumns) {
      if (!pyColumns.has(col)) {
        drifts.push({
          model: `${pyName}/${tsName}`,
          field: col,
          severity: "info",
          message: `Column "${col}" exists in TypeScript but missing in Python`,
        });
      }
    }

    // Compare indexes by column sets
    const pyIndexSets = pySchema.indexes.map((idx) => idx.columns.sort().join(","));
    const tsIndexSets = tsSchema.indexes.map((idx) => idx.columns.sort().join(","));

    for (const pyIdx of pySchema.indexes) {
      const pyKey = pyIdx.columns.sort().join(",");
      if (!tsIndexSets.includes(pyKey)) {
        drifts.push({
          model: pyName,
          field: `index(${pyIdx.columns.join(",")})`,
          severity: "warning",
          message: `Index on [${pyIdx.columns.join(", ")}] exists in Python but missing in TypeScript`,
          python: pyIdx,
        });
      }
    }

    for (const tsIdx of tsSchema.indexes) {
      const tsKey = tsIdx.columns.sort().join(",");
      if (!pyIndexSets.includes(tsKey)) {
        drifts.push({
          model: `${pyName}/${tsName}`,
          field: `index(${tsIdx.columns.join(",")})`,
          severity: "info",
          message: `Index on [${tsIdx.columns.join(", ")}] exists in TypeScript but missing in Python`,
          ts: tsIdx,
        });
      }
    }
  }

  // Check for TS models not in Python
  for (const [tsName, tsSchema] of Object.entries(tsSchemas)) {
    if (!pyByTable.has(tsSchema.table_name)) {
      drifts.push({
        model: tsName,
        field: "(model)",
        severity: "info",
        message: `TypeScript model "${tsName}" (table "${tsSchema.table_name}") has no Python counterpart`,
      });
    }
  }

  const errors = drifts.filter((d) => d.severity === "error").length;
  const warnings = drifts.filter((d) => d.severity === "warning").length;

  return {
    pass: errors === 0,
    drifts,
    summary: {
      modelsChecked: Object.keys(pythonSchemas).length,
      columnsChecked,
      errors,
      warnings,
    },
  };
}
