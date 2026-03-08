import { describe, it, expect } from "vitest";
import {
  NodeRegistry,
  createGraphNodeTypeResolver,
} from "../src/registry.js";
import type { NodeMetadata } from "../src/metadata.js";
import { BaseNode } from "../src/base-node.js";

const sampleMetadata: NodeMetadata = {
  title: "Strict Node",
  description: "desc",
  namespace: "test",
  node_type: "test.StrictNode",
  properties: [
    {
      name: "value",
      type: { type: "int", type_args: [] },
    },
  ],
  outputs: [
    {
      name: "output",
      type: { type: "list", type_args: [{ type: "string", type_args: [] }] },
    },
  ],
  is_streaming_output: true,
};

class LazyNode extends BaseNode {
  static readonly nodeType = "test.lazy.Node";
  static readonly title = "Lazy";
  static readonly description = "";
  async process() {
    return {};
  }
}

describe("createGraphNodeTypeResolver", () => {
  it("resolves registered metadata into kernel graph metadata", async () => {
    const registry = new NodeRegistry({
      metadataByType: new Map([["test.StrictNode", sampleMetadata]]),
    });
    const resolver = createGraphNodeTypeResolver(registry);

    const resolved = await resolver.resolveNodeType("test.StrictNode");
    expect(resolved).toEqual({
      nodeType: "test.StrictNode",
      propertyTypes: { value: "int" },
      outputs: { output: "list[string]" },
      isDynamic: false,
      descriptorDefaults: {
        name: "Strict Node",
        is_streaming_output: true,
      },
    });
  });

  it("supports Python-style Node suffix fallback", async () => {
    const registry = new NodeRegistry({
      metadataByType: new Map([
        [
          "test.Strict",
          {
            ...sampleMetadata,
            node_type: "test.Strict",
          },
        ],
      ]),
    });
    const resolver = createGraphNodeTypeResolver(registry);

    const resolved = await resolver.resolveNodeType("test.StrictNode");
    expect(resolved?.nodeType).toBe("test.Strict");
  });

  it("invokes namespace loader when metadata is missing", async () => {
    const registry = new NodeRegistry();
    let called = false;
    const lazyResolver = createGraphNodeTypeResolver(registry, {
      loadNamespace: (namespace, reg) => {
        if (namespace !== "test.lazy") return;
        called = true;
        reg.register(LazyNode, {
          metadata: {
            ...sampleMetadata,
            title: "Lazy",
            namespace: "test.lazy",
            node_type: "test.lazy.Node",
          },
        });
      },
    });

    const resolved = await lazyResolver.resolveNodeType("test.lazy.Node");
    expect(called).toBe(true);
    expect(resolved?.nodeType).toBe("test.lazy.Node");
  });
});
