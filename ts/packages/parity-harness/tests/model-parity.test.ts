import { describe, expect, it } from "vitest";
import {
  compareModelSchemas,
  type ModelSchema,
  type ModelSchemaMap,
} from "../src/model-parity.js";

// ── Unit tests for the comparison logic ──────────────────────────────

describe("compareModelSchemas", () => {
  it("returns pass=true when schemas are identical", () => {
    const schema: ModelSchema = {
      table_name: "items",
      primary_key: "id",
      columns: {
        id: { type: "string", optional: false },
        name: { type: "string", optional: false },
        count: { type: "number", optional: true },
      },
      indexes: [{ name: "idx_name", columns: ["name"], unique: false }],
    };

    const py: ModelSchemaMap = { Item: schema };
    const ts: ModelSchemaMap = { Item: { ...schema } };

    const report = compareModelSchemas(py, ts);
    expect(report.pass).toBe(true);
    expect(report.drifts).toHaveLength(0);
    expect(report.summary.errors).toBe(0);
  });

  it("detects missing TS model", () => {
    const py: ModelSchemaMap = {
      Item: {
        table_name: "items",
        primary_key: "id",
        columns: { id: { type: "string", optional: false } },
        indexes: [],
      },
    };
    const ts: ModelSchemaMap = {};

    const report = compareModelSchemas(py, ts);
    expect(report.pass).toBe(false);
    expect(report.drifts).toHaveLength(1);
    expect(report.drifts[0].severity).toBe("error");
    expect(report.drifts[0].message).toContain("no TypeScript counterpart");
  });

  it("detects missing Python model (info only)", () => {
    const py: ModelSchemaMap = {};
    const ts: ModelSchemaMap = {
      Extra: {
        table_name: "extras",
        primary_key: "id",
        columns: { id: { type: "string", optional: false } },
        indexes: [],
      },
    };

    const report = compareModelSchemas(py, ts);
    expect(report.pass).toBe(true); // TS-only models are info, not errors
    expect(report.drifts).toHaveLength(1);
    expect(report.drifts[0].severity).toBe("info");
  });

  it("detects primary key mismatch", () => {
    const py: ModelSchemaMap = {
      Item: {
        table_name: "items",
        primary_key: "id",
        columns: { id: { type: "string", optional: false } },
        indexes: [],
      },
    };
    const ts: ModelSchemaMap = {
      Item: {
        table_name: "items",
        primary_key: "item_id",
        columns: { item_id: { type: "string", optional: false } },
        indexes: [],
      },
    };

    const report = compareModelSchemas(py, ts);
    expect(report.drifts.some((d) => d.field === "primary_key")).toBe(true);
  });

  it("detects missing column in TS", () => {
    const py: ModelSchemaMap = {
      Item: {
        table_name: "items",
        primary_key: "id",
        columns: {
          id: { type: "string", optional: false },
          extra: { type: "string", optional: false },
        },
        indexes: [],
      },
    };
    const ts: ModelSchemaMap = {
      Item: {
        table_name: "items",
        primary_key: "id",
        columns: { id: { type: "string", optional: false } },
        indexes: [],
      },
    };

    const report = compareModelSchemas(py, ts);
    const colDrift = report.drifts.find((d) => d.field === "extra");
    expect(colDrift).toBeDefined();
    expect(colDrift!.severity).toBe("warning");
  });

  it("detects type mismatch", () => {
    const py: ModelSchemaMap = {
      Item: {
        table_name: "items",
        primary_key: "id",
        columns: {
          id: { type: "string", optional: false },
          count: { type: "number", optional: false },
        },
        indexes: [],
      },
    };
    const ts: ModelSchemaMap = {
      Item: {
        table_name: "items",
        primary_key: "id",
        columns: {
          id: { type: "string", optional: false },
          count: { type: "string", optional: false },
        },
        indexes: [],
      },
    };

    const report = compareModelSchemas(py, ts);
    const typeDrift = report.drifts.find(
      (d) => d.field === "count" && d.message.includes("Type mismatch"),
    );
    expect(typeDrift).toBeDefined();
  });

  it("detects missing index in TS", () => {
    const py: ModelSchemaMap = {
      Item: {
        table_name: "items",
        primary_key: "id",
        columns: { id: { type: "string", optional: false } },
        indexes: [{ name: "idx_name", columns: ["name"], unique: false }],
      },
    };
    const ts: ModelSchemaMap = {
      Item: {
        table_name: "items",
        primary_key: "id",
        columns: { id: { type: "string", optional: false } },
        indexes: [],
      },
    };

    const report = compareModelSchemas(py, ts);
    expect(report.drifts.some((d) => d.field.includes("index"))).toBe(true);
  });

  it("matches models by table_name, not class name", () => {
    const py: ModelSchemaMap = {
      MyAsset: {
        table_name: "shared_table",
        primary_key: "id",
        columns: { id: { type: "string", optional: false } },
        indexes: [],
      },
    };
    const ts: ModelSchemaMap = {
      Asset: {
        table_name: "shared_table",
        primary_key: "id",
        columns: { id: { type: "string", optional: false } },
        indexes: [],
      },
    };

    const report = compareModelSchemas(py, ts);
    expect(report.pass).toBe(true);
    expect(report.drifts).toHaveLength(0);
  });

  it("summary counts are correct", () => {
    const py: ModelSchemaMap = {
      A: {
        table_name: "a",
        primary_key: "id",
        columns: {
          id: { type: "string", optional: false },
          x: { type: "number", optional: false },
        },
        indexes: [],
      },
      B: {
        table_name: "b",
        primary_key: "id",
        columns: { id: { type: "string", optional: false } },
        indexes: [],
      },
    };
    const ts: ModelSchemaMap = {
      A: {
        table_name: "a",
        primary_key: "id",
        columns: {
          id: { type: "string", optional: false },
          x: { type: "number", optional: false },
        },
        indexes: [],
      },
      B: {
        table_name: "b",
        primary_key: "id",
        columns: { id: { type: "string", optional: false } },
        indexes: [],
      },
    };

    const report = compareModelSchemas(py, ts);
    expect(report.summary.modelsChecked).toBe(2);
    expect(report.summary.columnsChecked).toBe(3); // id + x in A, id in B
  });
});
