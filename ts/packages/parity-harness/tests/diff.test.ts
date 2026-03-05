import { describe, expect, it } from "vitest";
import { diffMessageStreams } from "../src/diff.js";

describe("diffMessageStreams", () => {
  it("returns no drift for equivalent streams", () => {
    const python = [
      { type: "job_update", status: "running", duration: 1.0 },
      { type: "output_update", value: { x: 1 } },
    ];
    const ts = [
      { type: "job_update", status: "running", duration: 2.0 },
      { type: "output_update", value: { x: 1 } },
    ];

    expect(diffMessageStreams(python, ts)).toEqual([]);
  });

  it("detects protocol drift on type mismatch", () => {
    const drifts = diffMessageStreams(
      [{ type: "job_update", status: "running" }],
      [{ type: "node_update", status: "running" }]
    );
    expect(drifts).toHaveLength(1);
    expect(drifts[0].category).toBe("protocol_drift");
  });

  it("detects output drift on output payload mismatch", () => {
    const drifts = diffMessageStreams(
      [{ type: "output_update", value: { result: 1 } }],
      [{ type: "output_update", value: { result: 2 } }]
    );
    expect(drifts).toHaveLength(1);
    expect(drifts[0].category).toBe("output_drift");
  });

  it("detects ordering drift for count mismatch", () => {
    const drifts = diffMessageStreams([{ type: "job_update" }], []);
    expect(drifts[0].category).toBe("ordering_drift");
  });

  it("detects timing_only_drift when only timing fields differ", () => {
    const drifts = diffMessageStreams(
      [{ type: "node_progress", elapsed_ms: 100, data: "x" }],
      [{ type: "node_progress", elapsed_ms: 500, data: "x" }]
    );
    expect(drifts).toHaveLength(1);
    expect(drifts[0].category).toBe("timing_only_drift");
    expect(drifts[0].message).toBe("Only timing fields differ");
  });

  it("detects ordering_drift when non-timing fields differ on non-output types", () => {
    const drifts = diffMessageStreams(
      [{ type: "node_progress", data: "alpha" }],
      [{ type: "node_progress", data: "beta" }]
    );
    expect(drifts).toHaveLength(1);
    expect(drifts[0].category).toBe("ordering_drift");
    expect(drifts[0].message).toBe("Message payload/order differs");
  });

  it("handles non-object messages for type comparison", () => {
    const drifts = diffMessageStreams(["plain_string"], ["plain_string"]);
    // Both non-objects, pType and tType are both undefined, so types match.
    // Then equalIgnoringFields compares them - they're equal strings.
    expect(drifts).toHaveLength(0);
  });

  it("handles non-object vs object for type comparison", () => {
    const drifts = diffMessageStreams(
      ["plain_string"],
      [{ type: "job_update" }]
    );
    // pType is undefined, tType is "job_update" => protocol_drift
    expect(drifts).toHaveLength(1);
    expect(drifts[0].category).toBe("protocol_drift");
  });

  it("handles null values in messages", () => {
    const drifts = diffMessageStreams([null], [null]);
    expect(drifts).toHaveLength(0);
  });

  it("applies custom ignoreFields option", () => {
    const drifts = diffMessageStreams(
      [{ type: "output_update", value: 1, custom: "a" }],
      [{ type: "output_update", value: 1, custom: "b" }],
      { ignoreFields: ["custom", "duration", "timestamp", "ts"] }
    );
    expect(drifts).toHaveLength(0);
  });

  it("handles job_update with payload match (no drift)", () => {
    const drifts = diffMessageStreams(
      [{ type: "job_update", status: "done", result: 42 }],
      [{ type: "job_update", status: "done", result: 42 }]
    );
    expect(drifts).toHaveLength(0);
  });

  it("stableValue handles nested arrays and objects", () => {
    const drifts = diffMessageStreams(
      [{ type: "output_update", value: { nested: [{ a: 1, b: 2 }] } }],
      [{ type: "output_update", value: { nested: [{ b: 2, a: 1 }] } }]
    );
    // Keys are sorted by stableValue, so these should match
    expect(drifts).toHaveLength(0);
  });

  it("detects output drift for job_update payload mismatch", () => {
    const drifts = diffMessageStreams(
      [{ type: "job_update", status: "done", result: 1 }],
      [{ type: "job_update", status: "done", result: 2 }]
    );
    expect(drifts).toHaveLength(1);
    expect(drifts[0].category).toBe("output_drift");
  });

  it("handles started_at and finished_at as timing-only differences", () => {
    const drifts = diffMessageStreams(
      [{ type: "node_progress", started_at: "2024-01-01", finished_at: "2024-01-02", data: "x" }],
      [{ type: "node_progress", started_at: "2025-01-01", finished_at: "2025-01-02", data: "x" }]
    );
    expect(drifts).toHaveLength(1);
    expect(drifts[0].category).toBe("timing_only_drift");
  });
});
