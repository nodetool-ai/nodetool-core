/**
 * Tests for T-K-14: Graph.fromDict.
 */
import { describe, it, expect } from "vitest";
import { Graph, GraphValidationError } from "../src/graph.js";

describe("T-K-14: Graph.fromDict", () => {
  it("creates graph from valid dict", () => {
    const graph = Graph.fromDict({
      nodes: [
        { id: "n1", type: "test.Add" },
        { id: "n2", type: "test.Mul" },
      ],
      edges: [
        { source: "n1", sourceHandle: "output", target: "n2", targetHandle: "a" },
      ],
    });
    expect(graph.nodes).toHaveLength(2);
    expect(graph.edges).toHaveLength(1);
    expect(graph.findNode("n1")).toBeDefined();
    expect(graph.findNode("n2")).toBeDefined();
  });

  it("creates graph with empty nodes and edges", () => {
    const graph = Graph.fromDict({ nodes: [], edges: [] });
    expect(graph.nodes).toHaveLength(0);
    expect(graph.edges).toHaveLength(0);
  });

  it("throws on missing nodes field", () => {
    expect(() => Graph.fromDict({} as any)).toThrow(GraphValidationError);
  });

  it("throws on missing edges field", () => {
    expect(() => Graph.fromDict({ nodes: [] } as any)).toThrow(GraphValidationError);
  });

  it("throws when nodes is not an array", () => {
    expect(() => Graph.fromDict({ nodes: "bad", edges: [] } as any)).toThrow(GraphValidationError);
  });

  it("throws when edges is not an array", () => {
    expect(() => Graph.fromDict({ nodes: [], edges: "bad" } as any)).toThrow(GraphValidationError);
  });

  it("preserves node properties", () => {
    const graph = Graph.fromDict({
      nodes: [
        {
          id: "n1",
          type: "test.Node",
          name: "my_node",
          is_streaming_output: true,
          outputs: { result: "int" },
        },
      ],
      edges: [],
    });
    const node = graph.findNode("n1");
    expect(node?.name).toBe("my_node");
    expect(node?.is_streaming_output).toBe(true);
    expect(node?.outputs?.result).toBe("int");
  });

  it("preserves edge properties", () => {
    const graph = Graph.fromDict({
      nodes: [
        { id: "n1", type: "test.A" },
        { id: "n2", type: "test.B" },
      ],
      edges: [
        {
          id: "e1",
          source: "n1",
          sourceHandle: "out",
          target: "n2",
          targetHandle: "in",
          edge_type: "control",
        },
      ],
    });
    expect(graph.edges[0].id).toBe("e1");
    expect(graph.edges[0].edge_type).toBe("control");
  });
});
