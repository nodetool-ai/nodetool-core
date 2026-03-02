import path from "node:path";
import { describe, expect, it } from "vitest";
import { NodeRegistry } from "@nodetool/node-sdk";
import { ALL_BASE_NODES, registerBaseNodes } from "../src/index.js";

function workspaceRootFromTestsDir(): string {
  return path.resolve(__dirname, "../../../../..");
}

describe("Python metadata parity", () => {
  it("loads Python package metadata from JSON", () => {
    const registry = new NodeRegistry();
    const loaded = registry.loadPythonMetadata({
      roots: [workspaceRootFromTestsDir()],
      maxDepth: 7,
    });

    expect(loaded.files.length).toBeGreaterThan(0);
    expect(loaded.nodesByType.size).toBeGreaterThan(0);
  });

  it("keeps TS base-node implementation aligned with Python metadata", () => {
    const registry = new NodeRegistry({ strictMetadata: true });
    registry.loadPythonMetadata({
      roots: [workspaceRootFromTestsDir()],
      maxDepth: 7,
    });
    registerBaseNodes(registry);

    for (const NodeClass of ALL_BASE_NODES) {
      const md = registry.getMetadata(NodeClass.nodeType);
      expect(md, `Missing metadata for ${NodeClass.nodeType}`).toBeDefined();
      if (!md) continue;

      const expectedNodeType = NodeClass.nodeType.endsWith("Node")
        ? NodeClass.nodeType.slice(0, -4)
        : NodeClass.nodeType;
      expect([NodeClass.nodeType, expectedNodeType]).toContain(md.node_type);
      expect(typeof md.title).toBe("string");
      expect(typeof md.description).toBe("string");
    }
  });
});
