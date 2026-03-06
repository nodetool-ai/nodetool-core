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
  /** ID of the final aggregation step (will use useFinishTask=true). */
  finalStepId?: string;
  /** Execute independent steps in parallel (default: false). */
  parallelExecution?: boolean;
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
  private finalStepId: string | undefined;
  private parallelExecution: boolean;
  private _finishStepId: string | undefined;

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
    this.finalStepId = opts.finalStepId;
    this.parallelExecution = opts.parallelExecution ?? false;
  }

  /**
   * Execute all steps in the task plan, respecting dependency order.
   * Supports both sequential and parallel execution modes.
   */
  async *executeTasks(): AsyncGenerator<ProcessingMessage> {
    // Seed inputs into context
    for (const [key, value] of Object.entries(this.inputs)) {
      this.context.set(key, value);
    }

    // Auto-detect finish step (last step) like Python does
    this._finishStepId = this.finalStepId ??
      (this.task.steps.length > 0 ? this.task.steps[this.task.steps.length - 1].id : undefined);

    let stepsTaken = 0;

    while (!this.allTasksComplete() && stepsTaken < this.maxSteps) {
      stepsTaken++;

      let executableSteps = this.getExecutableSteps();
      executableSteps = this.maybeDeferFinishStep(executableSteps);

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

      // Create step executors
      const stepGenerators = executableSteps.map((step) => {
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
          useFinishTask: this.isFinishStep(step),
        });
        return executor.execute();
      });

      if (this.parallelExecution && stepGenerators.length > 1) {
        // Execute all steps concurrently, merging yielded messages
        yield* mergeAsyncGenerators(stepGenerators);
      } else {
        // Execute steps sequentially
        for (const generator of stepGenerators) {
          for await (const message of generator) {
            yield message;
          }
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

  /**
   * Check if a step is the designated finish/aggregation step.
   */
  private isFinishStep(step: Step): boolean {
    if (this._finishStepId) {
      return step.id === this._finishStepId;
    }
    return this.task.steps.length > 0 && step === this.task.steps[this.task.steps.length - 1];
  }

  /**
   * Defer the finish step until all other steps are complete.
   * This ensures the final aggregation step runs last.
   */
  private maybeDeferFinishStep(executableSteps: Step[]): Step[] {
    if (!this._finishStepId) return executableSteps;

    const finishReady = executableSteps.some((s) => s.id === this._finishStepId);
    if (!finishReady) return executableSteps;

    const otherPending = this.task.steps.some(
      (s) => !s.completed && s.id !== this._finishStepId,
    );
    if (!otherPending) return executableSteps;

    return executableSteps.filter((s) => s.id !== this._finishStepId);
  }
}

// ---------------------------------------------------------------------------
// Async generator merge utility (TS equivalent of wrap_generators_parallel)
// ---------------------------------------------------------------------------

async function* mergeAsyncGenerators<T>(
  generators: AsyncGenerator<T>[],
): AsyncGenerator<T> {
  // Channel: a queue of resolved values with a promise-based pull mechanism
  const queue: T[] = [];
  let activeCount = generators.length;
  let resolve: (() => void) | null = null;
  let firstError: unknown = undefined;

  function notify() {
    if (resolve) {
      const r = resolve;
      resolve = null;
      r();
    }
  }

  // Start a consumer task for each generator
  const tasks = generators.map(async (gen) => {
    try {
      for await (const item of gen) {
        queue.push(item);
        notify();
      }
    } catch (e) {
      if (firstError === undefined) firstError = e;
    } finally {
      activeCount--;
      notify();
    }
  });

  // Yield items as they arrive
  while (activeCount > 0 || queue.length > 0) {
    if (queue.length > 0) {
      yield queue.shift()!;
    } else if (activeCount > 0) {
      // Wait for the next item or completion signal
      await new Promise<void>((r) => { resolve = r; });
    }
  }

  // Wait for all producer promises to settle
  await Promise.allSettled(tasks);

  if (firstError !== undefined) {
    throw firstError;
  }
}
