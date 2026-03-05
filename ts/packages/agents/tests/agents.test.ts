import { describe, it, expect, vi } from "vitest";
import { SimpleAgent } from "../src/simple-agent.js";
import { TaskPlanner } from "../src/task-planner.js";
import { TaskExecutor } from "../src/task-executor.js";
import type { Step, Task } from "../src/types.js";
import type { ProcessingMessage } from "@nodetool/protocol";

// ---------------------------------------------------------------------------
// Mock helpers
// ---------------------------------------------------------------------------

/**
 * Create a mock provider that yields items from a sequence of responses.
 * Each call to generateMessages consumes the next response from the queue.
 * If the queue is exhausted, returns an empty stream.
 */
function createMockProvider(
  responseSequence: Array<
    Array<
      | { type: "chunk"; content: string; done?: boolean }
      | { id: string; name: string; args: Record<string, unknown> }
    >
  >,
) {
  let callIndex = 0;
  return {
    provider: "mock",
    hasToolSupport: () => true,
    generateMessages: async function* () {
      const items = responseSequence[callIndex] ?? [];
      callIndex++;
      for (const item of items) {
        yield item;
      }
    },
    generateMessage: vi.fn(),
    getAvailableLanguageModels: vi.fn().mockResolvedValue([]),
    getAvailableImageModels: vi.fn().mockResolvedValue([]),
    getAvailableVideoModels: vi.fn().mockResolvedValue([]),
    getAvailableTTSModels: vi.fn().mockResolvedValue([]),
    getAvailableASRModels: vi.fn().mockResolvedValue([]),
    getAvailableEmbeddingModels: vi.fn().mockResolvedValue([]),
    getContainerEnv: () => ({}),
    textToImage: vi.fn(),
    imageToImage: vi.fn(),
    textToSpeech: vi.fn(),
    automaticSpeechRecognition: vi.fn(),
    textToVideo: vi.fn(),
    imageToVideo: vi.fn(),
    generateEmbedding: vi.fn(),
    isContextLengthError: () => false,
  } as any;
}

/**
 * Minimal mock ProcessingContext.
 */
function createMockContext() {
  const store = new Map<string, unknown>();
  return {
    storeStepResult: vi.fn(async (key: string, value: unknown) => {
      store.set(key, value);
      return key;
    }),
    loadStepResult: vi.fn(async (key: string) => {
      return store.get(key);
    }),
    set: vi.fn((key: string, value: unknown) => {
      store.set(key, value);
    }),
    get: vi.fn((key: string) => {
      return store.get(key);
    }),
    _store: store,
  } as any;
}

// ---------------------------------------------------------------------------
// SimpleAgent
// ---------------------------------------------------------------------------

describe("SimpleAgent", () => {
  it("executes and returns result from a single step", async () => {
    const provider = createMockProvider([
      [
        { type: "chunk", content: "thinking..." },
        {
          id: "tc_1",
          name: "finish_step",
          args: { result: { value: "hello" } },
        },
      ],
    ]);

    const agent = new SimpleAgent({
      name: "test-agent",
      objective: "Say hello",
      provider,
      model: "test-model",
      tools: [],
      outputSchema: {
        type: "object",
        properties: { value: { type: "string" } },
      },
    });

    const context = createMockContext();
    const messages: ProcessingMessage[] = [];
    for await (const msg of agent.execute(context)) {
      messages.push(msg);
    }

    expect(agent.getResults()).toEqual({ value: "hello" });
    expect(agent.task).not.toBeNull();
    expect(agent.task!.steps).toHaveLength(1);
    expect(agent.task!.steps[0].completed).toBe(true);
  });

  it("yields processing messages during execution", async () => {
    const provider = createMockProvider([
      [
        { type: "chunk", content: "Working on it..." },
        { type: "chunk", content: " Almost done." },
        {
          id: "tc_1",
          name: "finish_step",
          args: { result: { done: true } },
        },
      ],
    ]);

    const agent = new SimpleAgent({
      name: "test-agent",
      objective: "Do something",
      provider,
      model: "test-model",
      tools: [],
      outputSchema: { type: "object", properties: { done: { type: "boolean" } } },
    });

    const context = createMockContext();
    const messages: ProcessingMessage[] = [];
    for await (const msg of agent.execute(context)) {
      messages.push(msg);
    }

    const types = messages.map((m) => m.type);
    expect(types).toContain("task_update");
    expect(types).toContain("chunk");
    expect(types).toContain("step_result");

    // Verify we got the chunk content
    const chunks = messages.filter((m) => m.type === "chunk");
    expect(chunks.length).toBeGreaterThanOrEqual(2);
  });
});

// ---------------------------------------------------------------------------
// TaskPlanner
// ---------------------------------------------------------------------------

describe("TaskPlanner", () => {
  it("creates a task with steps from LLM response", async () => {
    const taskPayload = {
      title: "My Task",
      steps: [
        { id: "step_a", instructions: "Do A", depends_on: [] },
        { id: "step_b", instructions: "Do B", depends_on: ["step_a"] },
      ],
    };

    const provider = createMockProvider([
      [
        { type: "chunk", content: "Planning..." },
        {
          id: "tc_plan",
          name: "create_task",
          args: taskPayload,
        },
      ],
    ]);

    const planner = new TaskPlanner({
      provider,
      model: "test-model",
    });

    const context = createMockContext();
    const messages: ProcessingMessage[] = [];
    let task: Task | null = null;

    const gen = planner.plan("Build something", context);
    let result = await gen.next();
    while (!result.done) {
      messages.push(result.value);
      result = await gen.next();
    }
    task = result.value;

    expect(task).not.toBeNull();
    expect(task!.title).toBe("My Task");
    expect(task!.steps).toHaveLength(2);
    expect(task!.steps[0].id).toBe("step_a");
    expect(task!.steps[1].id).toBe("step_b");
    expect(task!.steps[1].dependsOn).toEqual(["step_a"]);

    // Should have a planning_update message
    const planningUpdates = messages.filter((m) => m.type === "planning_update");
    expect(planningUpdates.length).toBeGreaterThanOrEqual(1);
  });

  it("validates DAG structure and rejects circular deps", async () => {
    const circularPayload = {
      title: "Circular Task",
      steps: [
        { id: "step_x", instructions: "Do X", depends_on: ["step_y"] },
        { id: "step_y", instructions: "Do Y", depends_on: ["step_x"] },
      ],
    };

    const provider = createMockProvider([
      [
        {
          id: "tc_plan",
          name: "create_task",
          args: circularPayload,
        },
      ],
    ]);

    const planner = new TaskPlanner({
      provider,
      model: "test-model",
    });

    const context = createMockContext();
    const messages: ProcessingMessage[] = [];

    const gen = planner.plan("Circular objective", context);
    let result = await gen.next();
    while (!result.done) {
      messages.push(result.value);
      result = await gen.next();
    }
    const task = result.value;

    // Should return null due to circular dependencies
    expect(task).toBeNull();

    // Should have an error chunk about circular dependencies
    const errorChunks = messages.filter(
      (m) =>
        m.type === "chunk" &&
        "content" in m &&
        typeof (m as any).content === "string" &&
        (m as any).content.includes("circular"),
    );
    expect(errorChunks.length).toBeGreaterThanOrEqual(1);
  });
});

// ---------------------------------------------------------------------------
// TaskExecutor
// ---------------------------------------------------------------------------

describe("TaskExecutor", () => {
  it("executes steps in dependency order", async () => {
    const stepA: Step = {
      id: "step_a",
      instructions: "First step",
      completed: false,
      dependsOn: [],
      outputSchema: JSON.stringify({
        type: "object",
        properties: { data: { type: "string" } },
      }),
      logs: [],
    };

    const stepB: Step = {
      id: "step_b",
      instructions: "Second step",
      completed: false,
      dependsOn: ["step_a"],
      outputSchema: JSON.stringify({
        type: "object",
        properties: { result: { type: "string" } },
      }),
      logs: [],
    };

    const task: Task = {
      id: "task_1",
      title: "Sequential Task",
      steps: [stepA, stepB],
    };

    // Two calls to generateMessages: one for step_a, one for step_b
    const provider = createMockProvider([
      [
        { type: "chunk", content: "Executing A" },
        {
          id: "tc_a",
          name: "finish_step",
          args: { result: { data: "from_a" } },
        },
      ],
      [
        { type: "chunk", content: "Executing B" },
        {
          id: "tc_b",
          name: "finish_step",
          args: { result: { result: "from_b" } },
        },
      ],
    ]);

    const context = createMockContext();
    const executor = new TaskExecutor({
      provider,
      model: "test-model",
      context,
      tools: [],
      task,
    });

    const messages: ProcessingMessage[] = [];
    for await (const msg of executor.executeTasks()) {
      messages.push(msg);
    }

    // Both steps should be completed
    expect(stepA.completed).toBe(true);
    expect(stepB.completed).toBe(true);

    // step_a result should be stored before step_b runs
    expect(context.storeStepResult).toHaveBeenCalledWith("step_a", { data: "from_a" });
    expect(context.storeStepResult).toHaveBeenCalledWith("step_b", { result: "from_b" });

    // Verify step_result messages
    const stepResults = messages.filter((m) => m.type === "step_result");
    expect(stepResults).toHaveLength(2);
  });

  it("handles multi-step task with dependent steps", async () => {
    // Diamond dependency: A -> B, A -> C, B+C -> D
    const stepA: Step = {
      id: "a",
      instructions: "Root step",
      completed: false,
      dependsOn: [],
      outputSchema: JSON.stringify({ type: "object", properties: { v: { type: "number" } } }),
      logs: [],
    };
    const stepB: Step = {
      id: "b",
      instructions: "Branch 1",
      completed: false,
      dependsOn: ["a"],
      outputSchema: JSON.stringify({ type: "object", properties: { v: { type: "number" } } }),
      logs: [],
    };
    const stepC: Step = {
      id: "c",
      instructions: "Branch 2",
      completed: false,
      dependsOn: ["a"],
      outputSchema: JSON.stringify({ type: "object", properties: { v: { type: "number" } } }),
      logs: [],
    };
    const stepD: Step = {
      id: "d",
      instructions: "Merge step",
      completed: false,
      dependsOn: ["b", "c"],
      outputSchema: JSON.stringify({ type: "object", properties: { v: { type: "number" } } }),
      logs: [],
    };

    const task: Task = {
      id: "diamond_task",
      title: "Diamond dependency",
      steps: [stepA, stepB, stepC, stepD],
    };

    // The executor processes steps sequentially within each "round".
    // Round 1: step A (only one with no deps)
    // Round 2: steps B and C (both depend only on A)
    // Round 3: step D (depends on B and C)
    const provider = createMockProvider([
      // Round 1: step A
      [
        { type: "chunk", content: "A" },
        { id: "tc_a", name: "finish_step", args: { result: { v: 1 } } },
      ],
      // Round 2: step B
      [
        { type: "chunk", content: "B" },
        { id: "tc_b", name: "finish_step", args: { result: { v: 2 } } },
      ],
      // Round 2: step C
      [
        { type: "chunk", content: "C" },
        { id: "tc_c", name: "finish_step", args: { result: { v: 3 } } },
      ],
      // Round 3: step D
      [
        { type: "chunk", content: "D" },
        { id: "tc_d", name: "finish_step", args: { result: { v: 6 } } },
      ],
    ]);

    const context = createMockContext();
    const executor = new TaskExecutor({
      provider,
      model: "test-model",
      context,
      tools: [],
      task,
    });

    const messages: ProcessingMessage[] = [];
    for await (const msg of executor.executeTasks()) {
      messages.push(msg);
    }

    // All steps complete
    expect(stepA.completed).toBe(true);
    expect(stepB.completed).toBe(true);
    expect(stepC.completed).toBe(true);
    expect(stepD.completed).toBe(true);

    // Verify all step results were stored
    const stepResults = messages.filter((m) => m.type === "step_result");
    expect(stepResults).toHaveLength(4);

    // step D should have been the last to complete
    const resultIds = stepResults.map((m) => (m as any).step.id);
    expect(resultIds.indexOf("d")).toBeGreaterThan(resultIds.indexOf("a"));
    expect(resultIds.indexOf("d")).toBeGreaterThan(resultIds.indexOf("b"));
    expect(resultIds.indexOf("d")).toBeGreaterThan(resultIds.indexOf("c"));
  });
});
