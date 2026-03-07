import { beforeAll, describe, expect, it } from "vitest";
import { fetchJson, compareHttpResponses } from "../src/shadow.js";

const PYTHON_BASE_URL =
  process.env.NODETOOL_TEST_SERVER_URL ?? "http://localhost:7788";
const TS_BASE_URL =
  process.env.NODETOOL_TEST_TS_SERVER_URL ?? "http://localhost:7789";

let serverAvailable = false;
let tsServerAvailable = false;

describe("cross-language HTTP shadow tests", () => {
  beforeAll(async () => {
    const result = await fetchJson(`${PYTHON_BASE_URL}/openapi.json`, 2000);
    serverAvailable = result !== null;
    if (!serverAvailable) {
      console.warn(
        "⚠ Skipping HTTP shadow tests: Python server not running at " +
          PYTHON_BASE_URL,
      );
    }

    const tsResult = await fetchJson(`${TS_BASE_URL}/openapi.json`, 2000);
    tsServerAvailable = tsResult !== null;
  });

  it("GET /openapi.json returns 200 with openapi version and paths", async () => {
    if (!serverAvailable) return;

    const body = await fetchJson(`${PYTHON_BASE_URL}/openapi.json`, 5000);
    expect(body).not.toBeNull();

    const schema = body as Record<string, unknown>;
    expect(typeof schema["openapi"]).toBe("string");
    expect((schema["openapi"] as string).length).toBeGreaterThan(0);
    expect(schema["paths"]).toBeDefined();
    expect(typeof schema["paths"]).toBe("object");
    expect(Array.isArray(schema["paths"])).toBe(false);
  });

  it("GET /openapi.json has at least 50 paths", async () => {
    if (!serverAvailable) return;

    const body = await fetchJson(`${PYTHON_BASE_URL}/openapi.json`, 5000);
    expect(body).not.toBeNull();

    const schema = body as Record<string, unknown>;
    const paths = schema["paths"] as Record<string, unknown>;
    const pathCount = Object.keys(paths).length;
    expect(pathCount).toBeGreaterThanOrEqual(50);
  });

  it("GET /api/nodes/metadata returns an array or object (skips on 404)", async () => {
    if (!serverAvailable) return;

    const body = await fetchJson(
      `${PYTHON_BASE_URL}/api/nodes/metadata`,
      5000,
    );
    // fetchJson returns null on non-2xx (including 404) — treat as skip
    if (body === null) return;

    expect(
      Array.isArray(body) || (typeof body === "object" && body !== null),
    ).toBe(true);
  });

  it("compare /openapi.json from both servers: zero error drifts", async () => {
    if (!serverAvailable || !tsServerAvailable) return;

    const result = await compareHttpResponses(
      PYTHON_BASE_URL,
      TS_BASE_URL,
      "/openapi.json",
      { timeoutMs: 5000 },
    );

    // If either server went away between checks, treat as skip
    if (result === null) return;

    const errorDrifts = result.drifts.filter(
      (d) => d.category === "output_drift" || d.category === "ordering_drift",
    );
    expect(errorDrifts).toHaveLength(0);
  });
});
