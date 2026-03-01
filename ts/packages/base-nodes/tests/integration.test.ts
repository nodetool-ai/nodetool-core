import { describe, it, expect } from "vitest";
import { WorkflowRunner } from "@nodetool/kernel";
import { NodeRegistry } from "@nodetool/node-sdk";
import type { NodeDescriptor, Edge } from "@nodetool/protocol";
import {
  registerBaseNodes,
  IfNode,
  ForEachNode,
  RerouteNode,
  ListRangeNode,
  GenerateSequenceNode,
} from "../src/index.js";

function makeRegistry(): NodeRegistry {
  const registry = new NodeRegistry();
  registerBaseNodes(registry);
  return registry;
}

function makeRunner(registry: NodeRegistry): WorkflowRunner {
  return new WorkflowRunner("test-job", {
    resolveExecutor: (node) => {
      if (!registry.has(node.type)) {
        return {
          async process(inputs: Record<string, unknown>) {
            return inputs;
          },
        };
      }
      return registry.resolve(node);
    },
  });
}

describe("integration: If node with input source", () => {
  it("routes to true sink", async () => {
    const nodes: NodeDescriptor[] = [
      { id: "src", type: "test.Input", name: "value" },
      { id: "if", type: IfNode.nodeType, properties: { condition: true } },
      { id: "sink", type: RerouteNode.nodeType, name: "out" },
    ];
    const edges: Edge[] = [
      { source: "src", sourceHandle: "value", target: "if", targetHandle: "value" },
      {
        source: "if",
        sourceHandle: "if_true",
        target: "sink",
        targetHandle: "input_value",
      },
    ];

    const result = await makeRunner(makeRegistry()).run(
      { job_id: "if-1", params: { value: "hello" } },
      { nodes, edges }
    );

    expect(result.status).toBe("completed");
    expect(result.outputs.out).toContain("hello");
  });
});

describe("integration: ForEach node streaming output", () => {
  it("streams all items from generated range", async () => {
    const nodes: NodeDescriptor[] = [
      { id: "trigger", type: "test.Input", name: "start" },
      {
        id: "range",
        type: ListRangeNode.nodeType,
        properties: { stop: 4, step: 1 },
      },
      { id: "each", type: ForEachNode.nodeType, is_streaming_output: true },
      { id: "sink", type: RerouteNode.nodeType, name: "values" },
    ];

    const edges: Edge[] = [
      {
        source: "trigger",
        sourceHandle: "value",
        target: "range",
        targetHandle: "start",
      },
      {
        source: "range",
        sourceHandle: "output",
        target: "each",
        targetHandle: "input_list",
      },
      {
        source: "each",
        sourceHandle: "output",
        target: "sink",
        targetHandle: "input_value",
      },
    ];

    const result = await makeRunner(makeRegistry()).run(
      { job_id: "fe-1", params: { start: 1 } },
      { nodes, edges }
    );

    expect(result.status).toBe("completed");
    expect(result.outputs.values).toEqual([3]);
  });
});

describe("integration: GenerateSequence streaming output", () => {
  it("streams sequence values to sink", async () => {
    const nodes: NodeDescriptor[] = [
      { id: "trigger", type: "test.Input", name: "start" },
      {
        id: "seq",
        type: GenerateSequenceNode.nodeType,
        is_streaming_output: true,
        properties: { stop: 4, step: 1 },
      },
      { id: "sink", type: RerouteNode.nodeType, name: "values" },
    ];
    const edges: Edge[] = [
      {
        source: "trigger",
        sourceHandle: "value",
        target: "seq",
        targetHandle: "start",
      },
      {
        source: "seq",
        sourceHandle: "output",
        target: "sink",
        targetHandle: "input_value",
      },
    ];

    const result = await makeRunner(makeRegistry()).run(
      { job_id: "seq-1", params: { start: 1 } },
      { nodes, edges }
    );

    expect(result.status).toBe("completed");
    expect(result.outputs.values).toEqual([3]);
  });
});
