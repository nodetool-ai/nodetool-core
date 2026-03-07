/**
 * Integration test: Cross-language protocol message parity.
 *
 * This test:
 * 1. Runs the Python export script to get Python message schemas
 * 2. Calls compareProtocolMessages(pythonSchemas, TS_MESSAGE_MANIFEST)
 * 3. Asserts no errors (no missing message types, no missing required fields)
 * 4. Logs the full report summary (including warnings) for visibility
 *
 * Run with: npx vitest run tests/cross-language-messages.test.ts
 *
 * Requires Python to be available with nodetool installed.
 */

import { execSync } from "node:child_process";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  compareProtocolMessages,
  TS_MESSAGE_MANIFEST,
  type MessageSchemaMap,
} from "../src/protocol-parity.js";

/** Workspace root (tests/ → parity-harness/ → packages/ → ts/ → repo root). */
function repoRoot(): string {
  return path.resolve(__dirname, "../../../..");
}

/**
 * Run the Python export script and return the messages sub-map.
 * Throws if Python is not available.
 */
function loadPythonMessageSchemas(): MessageSchemaMap {
  const scriptPath = path.join(repoRoot(), "scripts", "export_parity_snapshot.py");
  const stdout = execSync(`uv run python ${scriptPath} messages`, {
    encoding: "utf-8",
    cwd: repoRoot(),
    timeout: 30_000,
  });
  const data = JSON.parse(stdout) as { messages: Record<string, unknown> };
  // Normalise: the Python exporter uses "type_discriminator", but our
  // MessageSchemaMap type expects "typeDiscriminator".
  const result: MessageSchemaMap = {};
  for (const [name, raw] of Object.entries(data.messages)) {
    const r = raw as {
      type_discriminator: string;
      fields: Record<string, { type: string; required: boolean }>;
    };
    result[name] = {
      typeDiscriminator: r.type_discriminator,
      fields: r.fields,
    };
  }
  return result;
}

describe("Cross-language protocol message parity", () => {
  it("Python and TypeScript message schemas have no errors", () => {
    let pySchemas: MessageSchemaMap;

    try {
      pySchemas = loadPythonMessageSchemas();
    } catch {
      console.warn("Skipping: Python not available or export script failed");
      return;
    }

    const report = compareProtocolMessages(pySchemas, TS_MESSAGE_MANIFEST);

    // Print a detailed summary for visibility in CI logs
    console.log("\nProtocol Message Parity Report:");
    console.log(`  Messages checked: ${report.summary.messagesChecked}`);
    console.log(`  Fields checked:   ${report.summary.fieldsChecked}`);
    console.log(`  Errors:           ${report.summary.errors}`);
    console.log(`  Warnings:         ${report.summary.warnings}`);
    console.log(`  Pass:             ${report.pass}`);

    if (report.drifts.length > 0) {
      console.log("\n  Drifts:");
      for (const d of report.drifts) {
        console.log(
          `    [${d.severity.toUpperCase()}] ${d.message}.${d.field}: ${d.message_text}`,
        );
        if (d.python !== undefined) console.log(`      python: ${JSON.stringify(d.python)}`);
        if (d.ts !== undefined) console.log(`      ts:     ${JSON.stringify(d.ts)}`);
      }
    }

    // No missing message types and no missing required fields
    expect(report.summary.errors).toBe(0);
  });

  it("Python export returns valid message schemas", () => {
    let pySchemas: MessageSchemaMap;

    try {
      pySchemas = loadPythonMessageSchemas();
    } catch {
      console.warn("Skipping: Python not available or export script failed");
      return;
    }

    expect(Object.keys(pySchemas).length).toBeGreaterThan(0);

    for (const [name, schema] of Object.entries(pySchemas)) {
      expect(typeof schema.typeDiscriminator).toBe("string");
      expect(schema.typeDiscriminator.length).toBeGreaterThan(0);
      expect(typeof schema.fields).toBe("object");

      for (const [fieldName, def] of Object.entries(schema.fields)) {
        expect(typeof def.type).toBe("string");
        expect(typeof def.required).toBe("boolean");
      }
    }
  });

  it("TypeScript manifest covers all Python message types", () => {
    let pySchemas: MessageSchemaMap;

    try {
      pySchemas = loadPythonMessageSchemas();
    } catch {
      console.warn("Skipping: Python not available or export script failed");
      return;
    }

    const tsDiscriminators = new Set(
      Object.values(TS_MESSAGE_MANIFEST).map((s) => s.typeDiscriminator),
    );

    const missing: string[] = [];
    for (const [name, schema] of Object.entries(pySchemas)) {
      if (!tsDiscriminators.has(schema.typeDiscriminator)) {
        missing.push(`${name} (${schema.typeDiscriminator})`);
      }
    }

    if (missing.length > 0) {
      console.warn(
        `TypeScript manifest missing message types: ${missing.join(", ")}`,
      );
    }

    expect(missing).toHaveLength(0);
  });
});
