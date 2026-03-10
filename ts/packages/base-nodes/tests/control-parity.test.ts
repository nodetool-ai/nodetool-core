import { describe, expect, it } from "vitest";
import { WorkflowRunner } from "@nodetool/kernel";
import { NodeRegistry } from "@nodetool/node-sdk";
import type { Edge, NodeDescriptor } from "@nodetool/protocol";
import {
  registerBaseNodes,
  CollectNode,
  ForEachNode,
  GenerateSequenceNode,
  IfNode,
  OutputNode,
} from "../src/index.js";

function makeRegistry(): NodeRegistry {
  const registry = new NodeRegistry();
  registerBaseNodes(registry);
  return registry;
}

function makeRunner(registry: NodeRegistry): WorkflowRunner {
  return new WorkflowRunner("control-parity", {
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

async function runWorkflow(
  nodes: NodeDescriptor[],
  edges: Edge[],
  params: Record<string, unknown> = {},
) {
  return makeRunner(makeRegistry()).run(
    { job_id: `control-parity-${Date.now()}`, params },
    { nodes, edges },
  );
}

describe("control parity: If node", () => {
  it("routes static true input to the true sink and leaves the false sink null", async () => {
    const result = await runWorkflow(
      [
        { id: "src", type: "test.Input", name: "value" },
        { id: "if", type: IfNode.nodeType, properties: { condition: true } },
        { id: "true", type: OutputNode.nodeType, name: "true_sink" },
        { id: "false", type: OutputNode.nodeType, name: "false_sink" },
      ],
      [
        { source: "src", sourceHandle: "value", target: "if", targetHandle: "value" },
        { source: "if", sourceHandle: "if_true", target: "true", targetHandle: "value" },
        { source: "if", sourceHandle: "if_false", target: "false", targetHandle: "value" },
      ],
      { value: "hello" },
    );

    expect(result.status).toBe("completed");
    expect(result.outputs.true_sink).toEqual(["hello"]);
    expect(result.outputs.false_sink).toEqual([null]);
  });

  it("routes static false input to the false sink and leaves the true sink null", async () => {
    const result = await runWorkflow(
      [
        { id: "src", type: "test.Input", name: "value" },
        { id: "if", type: IfNode.nodeType, properties: { condition: false } },
        { id: "true", type: OutputNode.nodeType, name: "true_sink" },
        { id: "false", type: OutputNode.nodeType, name: "false_sink" },
      ],
      [
        { source: "src", sourceHandle: "value", target: "if", targetHandle: "value" },
        { source: "if", sourceHandle: "if_true", target: "true", targetHandle: "value" },
        { source: "if", sourceHandle: "if_false", target: "false", targetHandle: "value" },
      ],
      { value: "hello" },
    );

    expect(result.status).toBe("completed");
    expect(result.outputs.true_sink).toEqual([null]);
    expect(result.outputs.false_sink).toEqual(["hello"]);
  });

  it("passes an entire generated stream through the true branch", async () => {
    const result = await runWorkflow(
      [
        {
          id: "src",
          type: GenerateSequenceNode.nodeType,
          is_streaming_output: true,
          properties: { start: 0, stop: 3, step: 1 },
        },
        { id: "if", type: IfNode.nodeType, properties: { condition: true } },
        { id: "collect", type: CollectNode.nodeType },
        { id: "out", type: OutputNode.nodeType, name: "passed" },
      ],
      [
        { source: "src", sourceHandle: "output", target: "if", targetHandle: "value" },
        { source: "if", sourceHandle: "if_true", target: "collect", targetHandle: "input_item" },
        { source: "collect", sourceHandle: "output", target: "out", targetHandle: "value" },
      ],
    );

    expect(result.status).toBe("completed");
    expect(result.outputs.passed).toEqual([[0, 1, 2]]);
  });

  it("zips condition and value streams and routes each item to the matching branch", async () => {
    const result = await runWorkflow(
      [
        {
          id: "cond",
          type: ForEachNode.nodeType,
          is_streaming_output: true,
          properties: { input_list: [true, true, false] },
        },
        {
          id: "val",
          type: ForEachNode.nodeType,
          is_streaming_output: true,
          properties: { input_list: ["A", "B", "C"] },
        },
        { id: "if", type: IfNode.nodeType, sync_mode: "zip_all" },
        { id: "true_out", type: OutputNode.nodeType, name: "true_sink" },
        { id: "false_out", type: OutputNode.nodeType, name: "false_sink" },
      ],
      [
        { source: "cond", sourceHandle: "output", target: "if", targetHandle: "condition" },
        { source: "val", sourceHandle: "output", target: "if", targetHandle: "value" },
        { source: "if", sourceHandle: "if_true", target: "true_out", targetHandle: "value" },
        { source: "if", sourceHandle: "if_false", target: "false_out", targetHandle: "value" },
      ],
    );

    const branchUpdates = result.messages
      .filter((message) => message.type === "output_update")
      .map((message) => ({
        node_id: message.node_id,
        value: message.value,
      }));
    const trueValues = branchUpdates
      .filter((message) => message.node_id === "true_out" && message.value != null)
      .map((message) => message.value);
    const falseValues = branchUpdates
      .filter((message) => message.node_id === "false_out" && message.value != null)
      .map((message) => message.value);

    expect(result.status).toBe("completed");
    expect(trueValues).toEqual(["A", "B"]);
    expect(falseValues).toEqual(["C"]);
  });
});

describe("control parity: Collect node", () => {
  it("aggregates a generated stream into one final list", async () => {
    const result = await runWorkflow(
      [
        {
          id: "src",
          type: GenerateSequenceNode.nodeType,
          is_streaming_output: true,
          properties: { start: 0, stop: 3, step: 1 },
        },
        { id: "collect", type: CollectNode.nodeType },
        { id: "out", type: OutputNode.nodeType, name: "items" },
      ],
      [
        { source: "src", sourceHandle: "output", target: "collect", targetHandle: "input_item" },
        { source: "collect", sourceHandle: "output", target: "out", targetHandle: "value" },
      ],
    );

    expect(result.status).toBe("completed");
    expect(result.outputs.items).toEqual([[0, 1, 2]]);
  });

  it("emits an empty list for an empty stream", async () => {
    const result = await runWorkflow(
      [
        { id: "collect", type: CollectNode.nodeType },
        { id: "out", type: OutputNode.nodeType, name: "items" },
      ],
      [
        { source: "collect", sourceHandle: "output", target: "out", targetHandle: "value" },
      ],
    );

    expect(result.status).toBe("completed");
    expect(result.outputs.items).toEqual([[]]);
  });
});
