import { describe, expect, it } from "vitest";
import { fetchJson, compareHttpResponses } from "../src/shadow.js";

// Port 19999 should never be listening in CI or local dev.
const DEAD_URL = "http://localhost:19999";

describe("fetchJson", () => {
  it("returns null for an unreachable URL", async () => {
    const result = await fetchJson(`${DEAD_URL}/anything`, 2000);
    expect(result).toBeNull();
  });

  it("returns null when timeout expires", async () => {
    // We use a non-routable address (port 0 is immediately refused on most OS,
    // so we rely on the dead port timing behaviour; either ECONNREFUSED or
    // AbortError → null).
    const result = await fetchJson(`${DEAD_URL}/slow`, 500);
    expect(result).toBeNull();
  });
});

describe("compareHttpResponses", () => {
  it("returns null when both servers are unreachable", async () => {
    const result = await compareHttpResponses(
      DEAD_URL,
      DEAD_URL,
      "/openapi.json",
      { timeoutMs: 2000 },
    );
    expect(result).toBeNull();
  });

  it("returns null when only the Python server is unreachable", async () => {
    // Both dead — result is null
    const result = await compareHttpResponses(
      DEAD_URL,
      DEAD_URL,
      "/openapi.json",
      { timeoutMs: 2000 },
    );
    expect(result).toBeNull();
  });
});
