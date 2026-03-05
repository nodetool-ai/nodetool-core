/**
 * TaskExecutor -- orchestrates execution of a complete Task plan.
 *
 * Port of src/nodetool/agents/task_executor.py (simplified)
 *
 * Iteratively finds steps whose dependencies are satisfied, runs
 * StepExecutor for each, and collects results until all steps complete
 * or the safety limit is reached.
 */

import type { BaseProvider } from "@nodetool/runtime";
import type { ProcessingContext } from "@nodetool/runtime";
import type { ProcessingMessage, Chunk } from "@nodetool/protocol";
import { StepExecutor } from "./step-executor.js";
import type { Tool } from "./tools/base-tool.js";
import type { Step, Task } from "./types.js";

const DEFAULT_MAX_STEPS = 50;
const DEFAULT_MAX_STEP_ITERATIONS = 10;
const DEFAULT_TOKEN_LIMIT = 128000;

export interface TaskExecutorOptions {
  provider: BaseProvider;
  model: string;
  context: ProcessingContext;
  tools: Tool[];
  task: Task;
  systemPrompt?: string;
  inputs?: Record<string, unknown>;
  maxSteps?: number;
  maxStepIterations?: number;
  maxTokenLimit?: number;
}

export class TaskExecutor {
  private provider: BaseProvider;
  private model: string;
  private tools: Tool[];
  private task: Task;
  private context: ProcessingContext;
  private inputs: Record<string, unknown>;
  private systemPrompt: string | undefined;
  private maxSteps: number;
  private maxStepIterations: number;
  private maxTokenLimit: number;

  constructor(opts: TaskExecutorOptions) {
    this.provider = opts.provider;
    this.model = opts.model;
    this.tools = opts.tools;
    this.task = opts.task;
    this.context = opts.context;
    this.inputs = opts.inputs ?? {};
    this.systemPrompt = opts.systemPrompt;
    this.maxSteps = opts.maxSteps ?? DEFAULT_MAX_STEPS;
    this.maxStepIterations = opts.maxStepIterations ?? DEFAULT_MAX_STEP_ITERATIONS;
    this.maxTokenLimit = opts.maxTokenLimit ?? DEFAULT_TOKEN_LIMIT;
  }

  /**
   * Execute all steps in the task plan, respecting dependency order.
   */
  async *executeTasks(): AsyncGenerator<ProcessingMessage> {
    // Seed inputs into context
    for (const [key, value] of Object.entries(this.inputs)) {
      this.context.set(key, value);
    }

    let stepsTaken = 0;

    while (!this.allTasksComplete() && stepsTaken < this.maxSteps) {
      stepsTaken++;

      const executableSteps = this.getExecutableSteps();

      if (executableSteps.length === 0) {
        if (!this.allTasksComplete()) {
          yield {
            type: "chunk",
            content: "\nNo executable steps but not all complete. Possible dependency issues.\n",
            done: false,
          } satisfies Chunk;
        }
        break;
      }

      // Execute steps sequentially
      for (const step of executableSteps) {
        const executor = new StepExecutor({
          task: this.task,
          step,
          context: this.context,
          provider: this.provider,
          model: this.model,
          tools: [...this.tools],
          systemPrompt: this.systemPrompt,
          maxTokenLimit: this.maxTokenLimit,
          maxIterations: this.maxStepIterations,
        });

        for await (const message of executor.execute()) {
          yield message;
        }
      }
    }
  }

  /**
   * Check if all steps in the task are completed.
   */
  private allTasksComplete(): boolean {
    return this.task.steps.every((step) => step.completed);
  }

  /**
   * Find steps whose dependencies are all satisfied (completed).
   */
  private getExecutableSteps(): Step[] {
    const completedIds = new Set(
      this.task.steps.filter((s) => s.completed).map((s) => s.id),
    );
    // Also count inputs as satisfied dependencies
    for (const key of Object.keys(this.inputs)) {
      completedIds.add(key);
    }

    return this.task.steps.filter(
      (step) =>
        !step.completed &&
        step.dependsOn.every((dep) => completedIds.has(dep)),
    );
  }
}
