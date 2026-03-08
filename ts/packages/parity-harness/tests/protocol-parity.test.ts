/**
 * Unit tests for protocol-parity.ts
 *
 * Tests the compareProtocolMessages() function and validates the static
 * TS_MESSAGE_MANIFEST derived from protocol/src/messages.ts.
 */

import { describe, expect, it } from "vitest";
import {
  compareProtocolMessages,
  TS_MESSAGE_MANIFEST,
  type MessageSchema,
  type MessageSchemaMap,
} from "../src/protocol-parity.js";

// ---------------------------------------------------------------------------
// Helper: build a minimal MessageSchemaMap for testing
// ---------------------------------------------------------------------------

function makeSchema(
  typeDiscriminator: string,
  fields: MessageSchema["fields"],
): MessageSchema {
  return { typeDiscriminator, fields };
}

// ---------------------------------------------------------------------------
// compareProtocolMessages – unit tests
// ---------------------------------------------------------------------------

describe("compareProtocolMessages", () => {
  it("returns pass=true when schemas are identical", () => {
    const schema = makeSchema("foo_update", {
      type: { type: "string", required: true },
      name: { type: "string", required: true },
      count: { type: "number", required: false },
    });
    const py: MessageSchemaMap = { FooUpdate: schema };
    const ts: MessageSchemaMap = { FooUpdate: { ...schema } };

    const report = compareProtocolMessages(py, ts);
    expect(report.pass).toBe(true);
    expect(report.summary.errors).toBe(0);
    expect(report.summary.warnings).toBe(0);
  });

  it("reports error when Python message has no TS counterpart", () => {
    const py: MessageSchemaMap = {
      MissingMessage: makeSchema("missing_msg", {
        type: { type: "string", required: true },
        data: { type: "string", required: true },
      }),
    };
    const ts: MessageSchemaMap = {};

    const report = compareProtocolMessages(py, ts);
    expect(report.pass).toBe(false);
    expect(report.summary.errors).toBeGreaterThan(0);

    const errDrift = report.drifts.find((d) => d.severity === "error");
    expect(errDrift).toBeDefined();
    expect(errDrift?.field).toBe("(message)");
    expect(errDrift?.message_text).toContain("missing_msg");
  });

  it("reports error when required Python field is missing in TS", () => {
    const pySchema = makeSchema("bar_update", {
      type: { type: "string", required: true },
      required_field: { type: "string", required: true },
    });
    const tsSchema = makeSchema("bar_update", {
      type: { type: "string", required: true },
      // required_field intentionally absent
    });

    const report = compareProtocolMessages(
      { BarUpdate: pySchema },
      { BarUpdate: tsSchema },
    );

    expect(report.pass).toBe(false);
    expect(report.summary.errors).toBeGreaterThan(0);

    const errDrift = report.drifts.find(
      (d) => d.severity === "error" && d.field === "required_field",
    );
    expect(errDrift).toBeDefined();
    expect(errDrift?.message_text).toContain("Required");
  });

  it("reports warning when optional Python field is missing in TS", () => {
    const pySchema = makeSchema("baz_update", {
      type: { type: "string", required: true },
      optional_field: { type: "string", required: false },
    });
    const tsSchema = makeSchema("baz_update", {
      type: { type: "string", required: true },
      // optional_field intentionally absent
    });

    const report = compareProtocolMessages(
      { BazUpdate: pySchema },
      { BazUpdate: tsSchema },
    );

    // No errors — missing optional fields are warnings only
    expect(report.summary.errors).toBe(0);

    const warnDrift = report.drifts.find(
      (d) => d.severity === "warning" && d.field === "optional_field",
    );
    expect(warnDrift).toBeDefined();
    expect(warnDrift?.message_text).toContain("Optional");
  });

  it("reports warning for type mismatches", () => {
    const pySchema = makeSchema("qux_update", {
      type: { type: "string", required: true },
      value: { type: "number", required: true },
    });
    const tsSchema = makeSchema("qux_update", {
      type: { type: "string", required: true },
      value: { type: "string", required: true }, // wrong type
    });

    const report = compareProtocolMessages(
      { QuxUpdate: pySchema },
      { QuxUpdate: tsSchema },
    );

    const warnDrift = report.drifts.find(
      (d) => d.severity === "warning" && d.field === "value",
    );
    expect(warnDrift).toBeDefined();
    expect(warnDrift?.message_text).toContain("Type mismatch");
  });

  it("reports info for TS-only fields", () => {
    const pySchema = makeSchema("extra_update", {
      type: { type: "string", required: true },
    });
    const tsSchema = makeSchema("extra_update", {
      type: { type: "string", required: true },
      ts_only_field: { type: "string", required: false },
    });

    const report = compareProtocolMessages(
      { ExtraUpdate: pySchema },
      { ExtraUpdate: tsSchema },
    );

    expect(report.summary.errors).toBe(0);
    const infoDrift = report.drifts.find(
      (d) => d.severity === "info" && d.field === "ts_only_field",
    );
    expect(infoDrift).toBeDefined();
  });

  it("reports info for TS messages not in Python", () => {
    const py: MessageSchemaMap = {};
    const ts: MessageSchemaMap = {
      TsOnlyMessage: makeSchema("ts_only_msg", {
        type: { type: "string", required: true },
      }),
    };

    const report = compareProtocolMessages(py, ts);
    expect(report.summary.errors).toBe(0);

    const infoDrift = report.drifts.find(
      (d) => d.severity === "info" && d.field === "(message)",
    );
    expect(infoDrift).toBeDefined();
    expect(infoDrift?.message_text).toContain("ts_only_msg");
  });

  it("summary counts are accurate", () => {
    const py: MessageSchemaMap = {
      Msg1: makeSchema("msg1", {
        type: { type: "string", required: true },
        a: { type: "string", required: true },
        b: { type: "string", required: false },
      }),
      Msg2: makeSchema("msg2", {
        type: { type: "string", required: true },
      }),
    };
    const ts: MessageSchemaMap = {
      Msg1: makeSchema("msg1", {
        type: { type: "string", required: true },
        a: { type: "string", required: true },
        // b missing → warning
      }),
      // Msg2 missing → error
    };

    const report = compareProtocolMessages(py, ts);
    expect(report.summary.messagesChecked).toBe(2);
    expect(report.summary.errors).toBeGreaterThan(0);
    expect(report.summary.warnings).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// TS_MESSAGE_MANIFEST – static manifest validation
// ---------------------------------------------------------------------------

describe("TS_MESSAGE_MANIFEST", () => {
  it("has entries for at least 15 message types", () => {
    expect(Object.keys(TS_MESSAGE_MANIFEST).length).toBeGreaterThanOrEqual(15);
  });

  it("contains job_update discriminator", () => {
    const entry = Object.values(TS_MESSAGE_MANIFEST).find(
      (s) => s.typeDiscriminator === "job_update",
    );
    expect(entry).toBeDefined();
  });

  it("contains node_update discriminator", () => {
    const entry = Object.values(TS_MESSAGE_MANIFEST).find(
      (s) => s.typeDiscriminator === "node_update",
    );
    expect(entry).toBeDefined();
  });

  it("contains output_update discriminator", () => {
    const entry = Object.values(TS_MESSAGE_MANIFEST).find(
      (s) => s.typeDiscriminator === "output_update",
    );
    expect(entry).toBeDefined();
  });

  it("JobUpdate has required status and type fields", () => {
    const schema = TS_MESSAGE_MANIFEST["JobUpdate"];
    expect(schema).toBeDefined();
    expect(schema.fields["status"]?.required).toBe(true);
    expect(schema.fields["type"]?.required).toBe(true);
  });

  it("NodeUpdate has required node_id, node_name, node_type, status", () => {
    const schema = TS_MESSAGE_MANIFEST["NodeUpdate"];
    expect(schema).toBeDefined();
    expect(schema.fields["node_id"]?.required).toBe(true);
    expect(schema.fields["node_name"]?.required).toBe(true);
    expect(schema.fields["node_type"]?.required).toBe(true);
    expect(schema.fields["status"]?.required).toBe(true);
  });

  it("Prediction has required id, user_id, node_id, status", () => {
    const schema = TS_MESSAGE_MANIFEST["Prediction"];
    expect(schema).toBeDefined();
    expect(schema.fields["id"]?.required).toBe(true);
    expect(schema.fields["user_id"]?.required).toBe(true);
    expect(schema.fields["node_id"]?.required).toBe(true);
    expect(schema.fields["status"]?.required).toBe(true);
  });

  it("Chunk has no required fields besides type", () => {
    const schema = TS_MESSAGE_MANIFEST["Chunk"];
    expect(schema).toBeDefined();
    const required = Object.entries(schema.fields)
      .filter(([k, v]) => v.required && k !== "type")
      .map(([k]) => k);
    expect(required).toHaveLength(0);
  });

  it("all entries have a non-empty typeDiscriminator", () => {
    for (const [name, schema] of Object.entries(TS_MESSAGE_MANIFEST)) {
      expect(typeof schema.typeDiscriminator).toBe("string");
      expect(schema.typeDiscriminator.length).toBeGreaterThan(0);
    }
  });

  it("all field defs have type string and boolean required", () => {
    for (const [msgName, schema] of Object.entries(TS_MESSAGE_MANIFEST)) {
      for (const [fieldName, def] of Object.entries(schema.fields)) {
        expect(typeof def.type).toBe("string");
        expect(typeof def.required).toBe("boolean");
      }
    }
  });
});
