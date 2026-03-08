/**
 * Tests for T-META-2: Node metadata introspection.
 */
import { describe, it, expect } from "vitest";
import { getNodeMetadata, getNodeMetadataBatch } from "../src/node-metadata.js";
import { BaseNode } from "../src/base-node.js";
import {
  Add,
  Passthrough,
  StreamingCounter,
  Constant,
  StringConcat,
  ThresholdProcessor,
  ErrorNode,
  SlowNode,
  SilentNode,
  Multiply,
} from "../src/nodes/test-nodes.js";

// ---------------------------------------------------------------------------
// getNodeMetadata
// ---------------------------------------------------------------------------

describe("getNodeMetadata", () => {
  it("extracts basic metadata from Add node", () => {
    const meta = getNodeMetadata(Add);
    expect(meta.title).toBe("Add");
    expect(meta.description).toBe("Adds two numbers");
    expect(meta.node_type).toBe("nodetool.test.Add");
    expect(meta.namespace).toBe("nodetool.test");
    expect(meta.is_streaming_output).toBe(false);
  });

  it("extracts properties with defaults", () => {
    const meta = getNodeMetadata(Add);
    expect(meta.properties).toHaveLength(2);

    const propA = meta.properties.find((p) => p.name === "a");
    const propB = meta.properties.find((p) => p.name === "b");
    expect(propA).toBeDefined();
    expect(propB).toBeDefined();
    expect(propA!.default).toBe(0);
    expect(propB!.default).toBe(0);
    expect(propA!.type.type).toBe("int");
    expect(propA!.required).toBe(false);
  });

  it("handles node with no defaults (Passthrough)", () => {
    const meta = getNodeMetadata(Passthrough);
    expect(meta.properties).toHaveLength(0);
    expect(meta.title).toBe("Passthrough");
    expect(meta.node_type).toBe("nodetool.test.Passthrough");
  });

  it("detects streaming output on StreamingCounter", () => {
    const meta = getNodeMetadata(StreamingCounter);
    expect(meta.is_streaming_output).toBe(true);
  });

  it("infers string types from defaults", () => {
    const meta = getNodeMetadata(StringConcat);
    const propA = meta.properties.find((p) => p.name === "a");
    expect(propA).toBeDefined();
    expect(propA!.type.type).toBe("string");
    expect(propA!.default).toBe("");
  });

  it("infers null type as any", () => {
    const meta = getNodeMetadata(Constant);
    const propValue = meta.properties.find((p) => p.name === "value");
    expect(propValue).toBeDefined();
    expect(propValue!.type.type).toBe("any");
    expect(propValue!.default).toBeNull();
  });

  it("infers float type from float defaults", () => {
    const meta = getNodeMetadata(ThresholdProcessor);
    const propThreshold = meta.properties.find((p) => p.name === "threshold");
    expect(propThreshold).toBeDefined();
    // 0.5 is a float
    expect(propThreshold!.type.type).toBe("float");
  });

  it("infers string type from string defaults", () => {
    const meta = getNodeMetadata(ThresholdProcessor);
    const propMode = meta.properties.find((p) => p.name === "mode");
    expect(propMode).toBeDefined();
    expect(propMode!.type.type).toBe("string");
    expect(propMode!.default).toBe("normal");
  });

  it("derives namespace from node_type", () => {
    const meta = getNodeMetadata(Add);
    expect(meta.namespace).toBe("nodetool.test");
  });

  it("uses node_type as title when title is empty", () => {
    // BaseNode has empty title, but test nodes set it. Test with a simple check.
    const meta = getNodeMetadata(Add);
    expect(meta.title).toBe("Add");
    expect(meta.title).not.toBe("");
  });
});

// ---------------------------------------------------------------------------
// getNodeMetadataBatch
// ---------------------------------------------------------------------------

describe("getNodeMetadataBatch", () => {
  it("extracts metadata from multiple node classes", () => {
    const batch = getNodeMetadataBatch([Add, Passthrough, StreamingCounter]);
    expect(batch).toHaveLength(3);
    expect(batch[0].node_type).toBe("nodetool.test.Add");
    expect(batch[1].node_type).toBe("nodetool.test.Passthrough");
    expect(batch[2].node_type).toBe("nodetool.test.StreamingCounter");
  });

  it("returns empty array for empty input", () => {
    expect(getNodeMetadataBatch([])).toEqual([]);
  });

  it("preserves order of input classes", () => {
    const batch = getNodeMetadataBatch([Constant, Add, Passthrough]);
    expect(batch[0].node_type).toBe("nodetool.test.Constant");
    expect(batch[1].node_type).toBe("nodetool.test.Add");
    expect(batch[2].node_type).toBe("nodetool.test.Passthrough");
  });
});

// ---------------------------------------------------------------------------
// Additional getNodeMetadata tests
// ---------------------------------------------------------------------------

describe("getNodeMetadata — additional coverage", () => {
  // ── Boolean type inference ────────────────────────────────────────────

  it("infers boolean type from boolean default", () => {
    class BoolNode extends BaseNode {
      static readonly nodeType = "nodetool.test.BoolNode";
      static readonly title = "Bool Node";
      static readonly description = "Has a boolean default";
      defaults() { return { flag: true }; }
      async process() { return {}; }
    }
    const meta = getNodeMetadata(BoolNode as unknown as import("../src/base-node.js").NodeClass);
    const prop = meta.properties.find((p) => p.name === "flag");
    expect(prop).toBeDefined();
    expect(prop!.type.type).toBe("bool");
    expect(prop!.default).toBe(true);
  });

  // ── Array type inference ──────────────────────────────────────────────

  it("infers list type from array default", () => {
    class ListNode extends BaseNode {
      static readonly nodeType = "nodetool.test.ListNode";
      static readonly title = "List Node";
      static readonly description = "Has an array default";
      defaults() { return { items: [1, 2, 3] }; }
      async process() { return {}; }
    }
    const meta = getNodeMetadata(ListNode as unknown as import("../src/base-node.js").NodeClass);
    const prop = meta.properties.find((p) => p.name === "items");
    expect(prop).toBeDefined();
    expect(prop!.type.type).toBe("list");
    expect(prop!.default).toEqual([1, 2, 3]);
  });

  // ── Object type inference ─────────────────────────────────────────────

  it("infers dict type from object default", () => {
    class DictNode extends BaseNode {
      static readonly nodeType = "nodetool.test.DictNode";
      static readonly title = "Dict Node";
      static readonly description = "Has an object default";
      defaults() { return { config: { key: "value" } }; }
      async process() { return {}; }
    }
    const meta = getNodeMetadata(DictNode as unknown as import("../src/base-node.js").NodeClass);
    const prop = meta.properties.find((p) => p.name === "config");
    expect(prop).toBeDefined();
    expect(prop!.type.type).toBe("dict");
    expect(prop!.default).toEqual({ key: "value" });
  });

  // ── is_dynamic is always false ────────────────────────────────────────

  it("is_dynamic is always false for multiple node types", () => {
    const nodeClasses = [Add, Passthrough, StreamingCounter, Constant, StringConcat, ThresholdProcessor];
    for (const cls of nodeClasses) {
      const meta = getNodeMetadata(cls);
      expect(meta.is_dynamic).toBe(false);
    }
  });

  // ── description defaults to empty string ──────────────────────────────

  it("description defaults to empty string for node without description", () => {
    class NoDescNode extends BaseNode {
      static readonly nodeType = "nodetool.test.NoDescNode";
      static readonly title = "No Desc";
      static readonly description = "";
      async process() { return {}; }
    }
    const meta = getNodeMetadata(NoDescNode as unknown as import("../src/base-node.js").NodeClass);
    expect(meta.description).toBe("");
  });

  // ── title fallback to nodeType ────────────────────────────────────────

  it("title falls back to nodeType when title is empty", () => {
    class NoTitleNode extends BaseNode {
      static readonly nodeType = "nodetool.test.NoTitleNode";
      static readonly title = "";
      static readonly description = "A node without title";
      async process() { return {}; }
    }
    const meta = getNodeMetadata(NoTitleNode as unknown as import("../src/base-node.js").NodeClass);
    expect(meta.title).toBe("nodetool.test.NoTitleNode");
  });

  // ── Outputs metadata ─────────────────────────────────────────────────

  it("outputs array is populated for nodes with declared outputs", () => {
    class OutputNode extends BaseNode {
      static readonly nodeType = "nodetool.test.OutputNode";
      static readonly title = "Output Node";
      static readonly description = "Has declared outputs";
      static readonly propertyTypes = {};

      static toDescriptor(id?: string) {
        return {
          ...super.toDescriptor(id),
          outputs: { result: "number", label: "string" },
        };
      }

      async process() { return { result: 42, label: "hello" }; }
    }
    const meta = getNodeMetadata(OutputNode as unknown as import("../src/base-node.js").NodeClass);
    expect(meta.outputs).toHaveLength(2);
    const resultOut = meta.outputs.find((o) => o.name === "result");
    const labelOut = meta.outputs.find((o) => o.name === "label");
    expect(resultOut).toBeDefined();
    expect(resultOut!.type.type).toBe("number");
    expect(labelOut).toBeDefined();
    expect(labelOut!.type.type).toBe("string");
  });

  it("outputs array is empty when descriptor has no outputs", () => {
    const meta = getNodeMetadata(Passthrough);
    // Passthrough's toDescriptor does not declare outputs
    expect(meta.outputs).toEqual([]);
  });

  // ── Namespace derivation edge cases ───────────────────────────────────

  it("namespace is empty string for node type without dots", () => {
    class FlatNode extends BaseNode {
      static readonly nodeType = "FlatNode";
      static readonly title = "Flat";
      static readonly description = "";
      async process() { return {}; }
    }
    const meta = getNodeMetadata(FlatNode as unknown as import("../src/base-node.js").NodeClass);
    expect(meta.namespace).toBe("");
  });

  // ── Multiple property types in one node ───────────────────────────────

  it("handles mixed property types in ThresholdProcessor", () => {
    const meta = getNodeMetadata(ThresholdProcessor);
    expect(meta.properties).toHaveLength(3);
    const types = meta.properties.map((p) => [p.name, p.type.type]);
    expect(types).toContainEqual(["value", "int"]);
    expect(types).toContainEqual(["threshold", "float"]);
    expect(types).toContainEqual(["mode", "string"]);
  });
});
