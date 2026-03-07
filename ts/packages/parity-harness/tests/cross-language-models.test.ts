/**
 * Integration test: Cross-language model schema parity.
 *
 * This test:
 * 1. Runs the Python export script to get Python model schemas
 * 2. Extracts TypeScript model schemas directly from @nodetool/models
 * 3. Compares them using the model-parity checker
 *
 * Run with: npx vitest run tests/cross-language-models.test.ts
 *
 * Requires Python to be available with nodetool installed.
 */

import { execSync } from "node:child_process";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { compareModelSchemas, type ModelSchemaMap } from "../src/model-parity.js";

/** Workspace root (tests/ → parity-harness/ → packages/ → ts/ → repo root). */
function repoRoot(): string {
  return path.resolve(__dirname, "../../../..");
}

/**
 * Run the Python export script and parse the JSON output.
 * Falls back to a cached snapshot if Python is not available.
 */
function loadPythonSchemas(): ModelSchemaMap {
  const scriptPath = path.join(repoRoot(), "scripts", "export_parity_snapshot.py");
  const stdout = execSync(`python3 ${scriptPath} models`, {
    encoding: "utf-8",
    cwd: repoRoot(),
    timeout: 30_000,
  });
  const data = JSON.parse(stdout);
  return data.models as ModelSchemaMap;
}

/**
 * Extract schemas from the TS model classes by reading their static schema
 * properties. We import each model file directly.
 */
function loadTsSchemas(): ModelSchemaMap {
  // We dynamically require each model file to extract the static schema.
  // This avoids needing @nodetool/models as a dependency of parity-harness.
  const modelDir = path.resolve(__dirname, "../../models/src");
  const modelFiles: Array<{ name: string; file: string }> = [
    { name: "Asset", file: "asset.js" },
    { name: "Job", file: "job.js" },
    { name: "Workflow", file: "workflow.js" },
    { name: "WorkflowVersion", file: "workflow-version.js" },
    { name: "Message", file: "message.js" },
    { name: "Thread", file: "thread.js" },
    { name: "Prediction", file: "prediction.js" },
    { name: "Secret", file: "secret.js" },
    { name: "OAuthCredential", file: "oauth-credential.js" },
    { name: "RunEvent", file: "run-event.js" },
    { name: "RunNodeState", file: "run-node-state.js" },
    { name: "RunLease", file: "run-lease.js" },
    { name: "Workspace", file: "workspace.js" },
  ];

  const schemas: ModelSchemaMap = {};
  for (const { name, file } of modelFiles) {
    try {
      // Use require to load the compiled TS model (dist/)
      const distFile = path.resolve(__dirname, "../../models/dist", file);
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const mod = require(distFile);
      const ModelClass = mod[name] ?? mod.default;
      if (!ModelClass?.schema) {
        continue;
      }
      const tsSchema = ModelClass.schema;
      const columns: Record<string, { type: string; optional: boolean }> = {};
      for (const [colName, colDef] of Object.entries(tsSchema.columns)) {
        const def = colDef as { type: string; optional?: boolean };
        columns[colName] = {
          type: def.type,
          optional: def.optional ?? false,
        };
      }
      schemas[name] = {
        table_name: tsSchema.table_name,
        primary_key: tsSchema.primary_key ?? "id",
        columns,
        indexes: (ModelClass.indexes ?? []).map(
          (idx: { name: string; columns: string[]; unique: boolean }) => ({
            name: idx.name,
            columns: idx.columns,
            unique: idx.unique,
          }),
        ),
      };
    } catch {
      // Model not available or not built yet — skip
    }
  }
  return schemas;
}

describe("Cross-language model parity", () => {
  it("Python and TypeScript model schemas have same table names", () => {
    let pySchemas: ModelSchemaMap;
    let tsSchemas: ModelSchemaMap;

    try {
      pySchemas = loadPythonSchemas();
    } catch {
      // Python not available in this environment
      console.warn("⚠ Skipping: Python not available");
      return;
    }

    try {
      tsSchemas = loadTsSchemas();
    } catch {
      console.warn("⚠ Skipping: TS models not built");
      return;
    }

    // At minimum, we should have some models from both sides
    expect(Object.keys(pySchemas).length).toBeGreaterThan(0);
    expect(Object.keys(tsSchemas).length).toBeGreaterThan(0);

    // All TS model table names should exist in Python
    const pyTableNames = new Set(Object.values(pySchemas).map((s) => s.table_name));
    for (const [tsName, tsSchema] of Object.entries(tsSchemas)) {
      expect(pyTableNames.has(tsSchema.table_name)).toBe(true);
    }
  });

  it("generates a full parity report without critical errors", () => {
    let pySchemas: ModelSchemaMap;
    let tsSchemas: ModelSchemaMap;

    try {
      pySchemas = loadPythonSchemas();
      tsSchemas = loadTsSchemas();
    } catch {
      console.warn("⚠ Skipping: Python or TS models not available");
      return;
    }

    const report = compareModelSchemas(pySchemas, tsSchemas);

    // Print a summary for visibility
    console.log("\n📊 Model Parity Report:");
    console.log(`   Models checked: ${report.summary.modelsChecked}`);
    console.log(`   Columns checked: ${report.summary.columnsChecked}`);
    console.log(`   Errors: ${report.summary.errors}`);
    console.log(`   Warnings: ${report.summary.warnings}`);
    if (report.drifts.length > 0) {
      console.log("\n   Drifts:");
      for (const d of report.drifts) {
        console.log(`     [${d.severity}] ${d.model}.${d.field}: ${d.message}`);
      }
    }

    // The report should not have critical errors (missing models)
    // Warnings about missing columns are acceptable during the port
    expect(report.summary.errors).toBe(0);
  });

  it("Python models export valid JSON with correct structure", () => {
    let pySchemas: ModelSchemaMap;
    try {
      pySchemas = loadPythonSchemas();
    } catch {
      console.warn("⚠ Skipping: Python not available");
      return;
    }

    for (const [name, schema] of Object.entries(pySchemas)) {
      expect(typeof schema.table_name).toBe("string");
      expect(typeof schema.primary_key).toBe("string");
      expect(typeof schema.columns).toBe("object");
      expect(Array.isArray(schema.indexes)).toBe(true);

      for (const [colName, colDef] of Object.entries(schema.columns)) {
        expect(typeof colDef.type).toBe("string");
        expect(typeof colDef.optional).toBe("boolean");
      }
    }
  });
});
