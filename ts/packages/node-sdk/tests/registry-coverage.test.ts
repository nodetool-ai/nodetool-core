/**
 * Additional NodeRegistry tests for coverage:
 *  - listMetadata
 *  - listRegisteredNodeTypesWithoutMetadata
 *  - register with explicit metadata
 *  - global register function
 *  - getMetadata fallback to loaded metadata
 *  - _resolveLoadedMetadata Node suffix stripping
 */

import { describe, it, expect, beforeEach } from "vitest";
import { NodeRegistry, register } from "../src/registry.js";
import { BaseNode } from "../src/base-node.js";
import type { NodeMetadata } from "../src/metadata.js";

class TestNodeA extends BaseNode {
  static readonly nodeType = "test.A";
  static readonly title = "A";
  static readonly description = "";
  async process() { return {}; }
}

class TestNodeB extends BaseNode {
  static readonly nodeType = "test.B";
  static readonly title = "B";
  static readonly description = "";
  async process() { return {}; }
}

class TestNodeWithSuffix extends BaseNode {
  static readonly nodeType = "test.SampleNode";
  static readonly title = "Sample Node";
  static readonly description = "";
  async process() { return {}; }
}

const sampleMetadata: NodeMetadata = {
  title: "A",
  description: "A node",
  namespace: "test",
  node_type: "test.A",
  properties: [],
  outputs: [],
};

describe("NodeRegistry – listMetadata", () => {
  let registry: NodeRegistry;

  beforeEach(() => {
    registry = new NodeRegistry();
  });

  it("returns metadata for all registered nodes that have metadata", () => {
    registry.register(TestNodeA, { metadata: sampleMetadata });
    registry.register(TestNodeB); // no metadata

    const mds = registry.listMetadata();
    expect(mds).toHaveLength(1);
    expect(mds[0].node_type).toBe("test.A");
  });

  it("returns empty array when no registered nodes have metadata", () => {
    registry.register(TestNodeA);
    const mds = registry.listMetadata();
    expect(mds).toHaveLength(0);
  });
});

describe("NodeRegistry – listRegisteredNodeTypesWithoutMetadata", () => {
  let registry: NodeRegistry;

  beforeEach(() => {
    registry = new NodeRegistry();
  });

  it("returns node types that have no metadata", () => {
    registry.register(TestNodeA, { metadata: sampleMetadata });
    registry.register(TestNodeB); // no metadata

    const noMeta = registry.listRegisteredNodeTypesWithoutMetadata();
    expect(noMeta).toEqual(["test.B"]);
  });

  it("returns empty when all have metadata", () => {
    registry.register(TestNodeA, { metadata: sampleMetadata });
    const noMeta = registry.listRegisteredNodeTypesWithoutMetadata();
    expect(noMeta).toEqual([]);
  });
});

describe("NodeRegistry – register with explicit metadata", () => {
  it("stores provided metadata", () => {
    const registry = new NodeRegistry();
    registry.register(TestNodeA, { metadata: sampleMetadata });
    expect(registry.getMetadata("test.A")).toEqual(sampleMetadata);
  });
});

describe("NodeRegistry – global register function", () => {
  it("registers on NodeRegistry.global", () => {
    // Clear existing registrations by creating fresh global
    const globalBefore = NodeRegistry.global.has("test.A");

    register(TestNodeA);
    expect(NodeRegistry.global.has("test.A")).toBe(true);
  });
});

describe("NodeRegistry – getMetadata with loaded metadata", () => {
  it("falls back to loaded metadata when no registered metadata", () => {
    const loaded = new Map<string, NodeMetadata>();
    loaded.set("test.A", sampleMetadata);

    const registry = new NodeRegistry({ metadataByType: loaded });
    registry.register(TestNodeA);

    expect(registry.getMetadata("test.A")).toEqual(sampleMetadata);
  });

  it("prefers registered metadata over loaded metadata", () => {
    const loaded = new Map<string, NodeMetadata>();
    loaded.set("test.A", { ...sampleMetadata, title: "Loaded" });

    const registry = new NodeRegistry({ metadataByType: loaded });
    const registeredMeta = { ...sampleMetadata, title: "Registered" };
    registry.register(TestNodeA, { metadata: registeredMeta });

    expect(registry.getMetadata("test.A")?.title).toBe("Registered");
  });
});

describe("NodeRegistry – _resolveLoadedMetadata Node suffix", () => {
  it("resolves metadata when nodeType ends with Node by stripping suffix", () => {
    const loaded = new Map<string, NodeMetadata>();
    loaded.set("test.Sample", {
      ...sampleMetadata,
      node_type: "test.Sample",
      title: "Sample",
    });

    const registry = new NodeRegistry({ metadataByType: loaded });
    registry.register(TestNodeWithSuffix);

    // test.SampleNode -> tries test.SampleNode first, then test.Sample
    expect(registry.getMetadata("test.SampleNode")?.title).toBe("Sample");
  });
});
