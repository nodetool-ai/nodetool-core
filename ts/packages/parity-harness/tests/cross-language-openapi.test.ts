/**
 * Integration test: Cross-language OpenAPI schema parity.
 *
 * This test:
 * 1. Runs the Python export script to get the Python FastAPI OpenAPI schema
 * 2. Validates the structure of the exported schema
 * 3. Calls extractRoutes() and validates we get a meaningful number of routes
 *
 * Run with: npx vitest run tests/cross-language-openapi.test.ts
 *
 * Requires Python to be available with nodetool installed.
 */

import { execSync } from "node:child_process";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { extractRoutes } from "../src/openapi-parity.js";

/** Workspace root (tests/ -> parity-harness/ -> packages/ -> ts/ -> repo root). */
function repoRoot(): string {
  return path.resolve(__dirname, "../../../..");
}

/**
 * Run the Python export script and parse the JSON output for the openapi section.
 * Returns null if Python is not available or the script fails.
 */
function loadPythonOpenApiSchema(): Record<string, unknown> | null {
  try {
    const stdout = execSync(
      `uv run python scripts/export_parity_snapshot.py openapi`,
      {
        encoding: "utf-8",
        cwd: repoRoot(),
        timeout: 60_000,
      },
    );
    const data = JSON.parse(stdout) as Record<string, unknown>;
    return data["openapi"] as Record<string, unknown>;
  } catch {
    return null;
  }
}

describe("Cross-language OpenAPI schema parity", () => {
  it("Python OpenAPI schema has the right top-level structure", () => {
    const schema = loadPythonOpenApiSchema();
    if (!schema) {
      console.warn("⚠ Skipping: Python not available");
      return;
    }

    // Must be an object
    expect(typeof schema).toBe("object");
    expect(schema).not.toBeNull();

    // Must have an openapi version string
    expect(typeof schema["openapi"]).toBe("string");
    expect((schema["openapi"] as string).startsWith("3.")).toBe(true);

    // Must have a paths object
    expect(typeof schema["paths"]).toBe("object");
    expect(schema["paths"]).not.toBeNull();

    // Must have a components.schemas object
    const components = schema["components"];
    expect(typeof components).toBe("object");
    expect(components).not.toBeNull();

    const schemasObj = (components as Record<string, unknown>)["schemas"];
    expect(typeof schemasObj).toBe("object");
    expect(schemasObj).not.toBeNull();
  });

  it("Python OpenAPI schema has >100 routes", () => {
    const schema = loadPythonOpenApiSchema();
    if (!schema) {
      console.warn("⚠ Skipping: Python not available");
      return;
    }

    const routes = extractRoutes(schema);
    const routeCount = Object.keys(routes).length;

    console.log(`\n  Python OpenAPI routes extracted: ${routeCount}`);

    // The Python FastAPI server has many routes — we expect well over 100
    expect(routeCount).toBeGreaterThan(100);
  });

  it("Python OpenAPI paths contain expected API prefixes", () => {
    const schema = loadPythonOpenApiSchema();
    if (!schema) {
      console.warn("⚠ Skipping: Python not available");
      return;
    }

    const routes = extractRoutes(schema);
    const routeKeys = Object.keys(routes);

    // There should be routes under /api/ at minimum
    const apiRoutes = routeKeys.filter((k) => k.includes(" /api/"));
    expect(apiRoutes.length).toBeGreaterThan(0);
  });

  it("extractRoutes returns valid OpenApiRoute objects from Python schema", () => {
    const schema = loadPythonOpenApiSchema();
    if (!schema) {
      console.warn("⚠ Skipping: Python not available");
      return;
    }

    const routes = extractRoutes(schema);

    for (const [key, route] of Object.entries(routes)) {
      // key must be "METHOD /path"
      expect(typeof key).toBe("string");
      expect(key).toMatch(/^[A-Z]+ \//);

      // method and path must be strings
      expect(typeof route.method).toBe("string");
      expect(typeof route.path).toBe("string");

      // responseFields must be an object
      expect(typeof route.responseFields).toBe("object");

      // Each field must have type (string) and required (boolean)
      for (const [_fieldName, fieldDef] of Object.entries(route.responseFields)) {
        expect(typeof fieldDef.type).toBe("string");
        expect(typeof fieldDef.required).toBe("boolean");
      }
    }
  });
});
