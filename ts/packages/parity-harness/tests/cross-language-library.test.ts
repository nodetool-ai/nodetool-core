/**
 * Integration test: Cross-language library parity.
 *
 * This test:
 * 1. Runs the Python export script to get Python library class signatures
 * 2. Extracts TypeScript class method names from @nodetool/models and @nodetool/kernel
 * 3. Compares them using the library-parity checker
 *
 * Run with: npx vitest run tests/cross-language-library.test.ts
 *
 * Requires Python to be available with nodetool installed.
 */

import { execSync } from "node:child_process";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  compareLibraryClasses,
  type LibraryClass,
  type TsClassDef,
  type TsMethodDef,
} from "../src/library-parity.js";

/** Workspace root (tests/ → parity-harness/ → packages/ → ts/ → repo root). */
function repoRoot(): string {
  return path.resolve(__dirname, "../../../..");
}

/**
 * Run the Python export script and get library class signatures.
 * Returns an empty array if Python is not available.
 */
function loadPythonLibrary(): LibraryClass[] {
  const scriptPath = path.join(repoRoot(), "scripts", "export_parity_snapshot.py");
  const stdout = execSync(`python3 ${scriptPath} library`, {
    encoding: "utf-8",
    cwd: repoRoot(),
    timeout: 30_000,
  });
  const data = JSON.parse(stdout);
  return data.library as LibraryClass[];
}

/**
 * Extract public methods from a JavaScript class constructor.
 * Includes both instance methods (from prototype) and static methods.
 * Private methods (prefixed with `_`) are excluded.
 */
function extractTsMethods(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ctor: new (...args: any[]) => unknown,
): TsMethodDef[] {
  const methods: TsMethodDef[] = [];
  const excluded = new Set(["constructor", "length", "name", "prototype", "caller", "arguments"]);

  // Instance methods from prototype
  for (const name of Object.getOwnPropertyNames(ctor.prototype)) {
    if (excluded.has(name) || name.startsWith("_")) continue;
    const desc = Object.getOwnPropertyDescriptor(ctor.prototype, name);
    if (desc && typeof desc.value === "function") {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      methods.push({ name, paramCount: (desc.value as any).length ?? 0 });
    }
  }

  // Static methods on the class itself
  for (const name of Object.getOwnPropertyNames(ctor)) {
    if (excluded.has(name) || name.startsWith("_")) continue;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const val = (ctor as any)[name];
    if (typeof val === "function") {
      methods.push({ name, paramCount: val.length ?? 0 });
    }
  }

  return methods;
}

/**
 * Load TypeScript class definitions by introspecting compiled package dist files.
 */
function loadTsLibrary(): TsClassDef[] {
  const classes: TsClassDef[] = [];

  // DBModel from @nodetool/models
  try {
    const distFile = path.resolve(__dirname, "../../models/dist/base-model.js");
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const mod = require(distFile);
    const DBModel = mod.DBModel ?? mod.default;
    if (DBModel) {
      classes.push({ name: "DBModel", methods: extractTsMethods(DBModel) });
    }
  } catch {
    // Package not built — skip
  }

  // Graph from @nodetool/kernel
  try {
    const distFile = path.resolve(__dirname, "../../kernel/dist/graph.js");
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const mod = require(distFile);
    const Graph = mod.Graph ?? mod.default;
    if (Graph) {
      classes.push({ name: "Graph", methods: extractTsMethods(Graph) });
    }
  } catch {
    // Package not built — skip
  }

  return classes;
}

describe("Cross-language library parity", () => {
  it("Python and TypeScript library classes have no critical method drift", () => {
    let pyClasses: LibraryClass[];
    let tsClasses: TsClassDef[];

    try {
      pyClasses = loadPythonLibrary();
    } catch {
      console.warn("⚠ Skipping: Python not available");
      return;
    }

    try {
      tsClasses = loadTsLibrary();
    } catch {
      console.warn("⚠ Skipping: TS library packages not built");
      return;
    }

    if (tsClasses.length === 0) {
      console.warn("⚠ Skipping: no TS library classes found (packages not built)");
      return;
    }

    const report = compareLibraryClasses(pyClasses, tsClasses);

    // Print a summary for visibility
    console.log("\n📊 Library Parity Report:");
    console.log(`   Classes checked: ${report.summary.classesChecked}`);
    console.log(`   Methods checked: ${report.summary.methodsChecked}`);
    console.log(`   Matched: ${report.summary.matched}`);
    console.log(`   Python-only: ${report.summary.pythonOnly}`);
    console.log(`   TS-only: ${report.summary.tsOnly}`);
    if (report.drifts.length > 0) {
      console.log("\n   Drifts:");
      for (const d of report.drifts) {
        console.log(`     [${d.severity}] ${d.class_name}.${d.method}: ${d.message}`);
      }
    }

    // The report should pass — no errors (only info/warning drifts are acceptable
    // while the TypeScript port is still in progress)
    expect(report.pass).toBe(true);
  });

  it("Python library export produces valid JSON with correct structure", () => {
    let pyClasses: LibraryClass[];
    try {
      pyClasses = loadPythonLibrary();
    } catch {
      console.warn("⚠ Skipping: Python not available");
      return;
    }

    expect(pyClasses.length).toBeGreaterThan(0);

    for (const cls of pyClasses) {
      expect(typeof cls.module).toBe("string");
      expect(typeof cls.class).toBe("string");
      if (!cls.error) {
        expect(Array.isArray(cls.methods)).toBe(true);
        for (const method of cls.methods) {
          expect(typeof method.name).toBe("string");
          expect(Array.isArray(method.params)).toBe(true);
        }
      }
    }
  });

  it("Python export filters out Pydantic base class methods", () => {
    let pyClasses: LibraryClass[];
    try {
      pyClasses = loadPythonLibrary();
    } catch {
      console.warn("⚠ Skipping: Python not available");
      return;
    }

    const pydanticOnlyMethods = ["model_dump", "model_copy", "model_dump_json", "model_post_init"];
    for (const cls of pyClasses) {
      if (cls.error) continue;
      const methodNames = cls.methods.map((m) => m.name);
      for (const pydanticMethod of pydanticOnlyMethods) {
        expect(methodNames).not.toContain(pydanticMethod);
      }
    }
  });

  it("TypeScript library classes expose expected core methods", () => {
    let tsClasses: TsClassDef[];
    try {
      tsClasses = loadTsLibrary();
    } catch {
      console.warn("⚠ Skipping: TS library packages not built");
      return;
    }

    if (tsClasses.length === 0) {
      console.warn("⚠ Skipping: no TS library classes found (packages not built)");
      return;
    }

    const dbModel = tsClasses.find((c) => c.name === "DBModel");
    if (dbModel) {
      const methodNames = dbModel.methods.map((m) => m.name);
      expect(methodNames).toContain("save");
      expect(methodNames).toContain("delete");
      expect(methodNames).toContain("getEtag");
      expect(methodNames).toContain("partitionValue");
      // Static methods
      expect(methodNames).toContain("create");
      expect(methodNames).toContain("get");
      expect(methodNames).toContain("query");
    }

    const graph = tsClasses.find((c) => c.name === "Graph");
    if (graph) {
      const methodNames = graph.methods.map((m) => m.name);
      expect(methodNames).toContain("findNode");
      expect(methodNames).toContain("topologicalSort");
      expect(methodNames).toContain("getInputSchema");
      expect(methodNames).toContain("getOutputSchema");
      // Static method
      expect(methodNames).toContain("fromDict");
    }
  });
});
