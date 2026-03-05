import { describe, it, expect, vi } from "vitest";
import { StepExecutor } from "../src/step-executor.js";
import type { Step, Task } from "../src/types.js";
import type { ProcessingMessage } from "@nodetool/protocol";

/**
 * Minimal mock provider that returns a single assistant message
 * with a finish_step tool call.
 */
function createMockProvider(toolCallArgs?: Record<string, unknown>) {
  const args = toolCallArgs ?? { result: { answer: "42" } };
  return {
    provider: "mock",
    hasToolSupport: () => true,
    generateMessages: async function* () {
      // Yield a text chunk
      yield { type: "chunk" as const, content: "Working on it...", done: false };
      // Yield a finish_step tool call
      yield {
        id: "tc_1",
        name: "finish_step",
        args,
      };
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
 * Minimal mock context with storeStepResult and loadStepResult.
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
    set: vi.fn(),
    get: vi.fn(),
    _store: store,
  } as any;
}

describe("StepExecutor", () => {
  it("executes a simple step and captures result via finish_step", async () => {
    const step: Step = {
      id: "step_1",
      instructions: "Compute the answer to life",
      completed: false,
      dependsOn: [],
      outputSchema: JSON.stringify({
        type: "object",
        properties: { answer: { type: "string" } },
        required: ["answer"],
      }),
      logs: [],
    };

    const task: Task = {
      id: "task_1",
      title: "Test Task",
      steps: [step],
    };

    const provider = createMockProvider();
    const context = createMockContext();

    const executor = new StepExecutor({
      task,
      step,
      context,
      provider,
      model: "test-model",
    });

    const messages: ProcessingMessage[] = [];
    for await (const msg of executor.execute()) {
      messages.push(msg);
    }

    // Should have received a chunk, then step_result
    const types = messages.map((m) => m.type);
    expect(types).toContain("task_update"); // step_started
    expect(types).toContain("chunk");
    expect(types).toContain("step_result");

    // Step should be marked completed
    expect(step.completed).toBe(true);

    // Result should be stored
    expect(context.storeStepResult).toHaveBeenCalledWith("step_1", { answer: "42" });
    expect(executor.getResult()).toEqual({ answer: "42" });
  });

  it("handles text-only response with JSON extraction", async () => {
    const step: Step = {
      id: "step_2",
      instructions: "Generate a greeting",
      completed: false,
      dependsOn: [],
      outputSchema: JSON.stringify({
        type: "object",
        properties: { greeting: { type: "string" } },
      }),
      logs: [],
    };

    const task: Task = {
      id: "task_2",
      title: "Test Task 2",
      steps: [step],
    };

    // Provider that returns text with embedded JSON but no tool call
    const provider = {
      ...createMockProvider(),
      generateMessages: async function* () {
        yield {
          type: "chunk" as const,
          content: 'Here is the result: {"result": {"greeting": "hello"}}',
          done: false,
        };
      },
    } as any;

    const context = createMockContext();

    const executor = new StepExecutor({
      task,
      step,
      context,
      provider,
      model: "test-model",
    });

    const messages: ProcessingMessage[] = [];
    for await (const msg of executor.execute()) {
      messages.push(msg);
    }

    expect(step.completed).toBe(true);
    expect(executor.getResult()).toEqual({ greeting: "hello" });
  });
});
