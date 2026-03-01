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
});
