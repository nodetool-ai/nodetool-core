/**
 * StepExecutor -- the core execution engine for a single step.
 *
 * Port of src/nodetool/agents/step_executor.py
 *
 * Manages the LLM interaction loop: sending messages, handling tool calls,
 * monitoring token limits, and capturing the step result via finish_step
 * or inline JSON extraction.
 */

import type { BaseProvider, ProcessingContext, Message, ToolCall, ProviderStreamItem } from "@nodetool/runtime";
import {
  TaskUpdateEvent,
  type ProcessingMessage,
  type Chunk,
  type ToolCallUpdate,
  type StepResult,
  type TaskUpdate,
} from "@nodetool/protocol";
import type { Step, Task } from "./types.js";
import type { Tool } from "./tools/base-tool.js";
import { FinishStepTool } from "./tools/finish-step-tool.js";
import { extractJSON } from "./utils/json-parser.js";

const MAX_TOOL_RESULT_CHARS = 20000;
const DEFAULT_MAX_ITERATIONS = 30;
const DEFAULT_TOKEN_LIMIT = 128000;

const DEFAULT_EXECUTION_SYSTEM_PROMPT = `# Role
You are executing EXACTLY one step within a larger plan. Complete this step end-to-end.

# Hard Constraint: No Human Feedback
- Do NOT ask clarifying questions or request user input.
- If something is ambiguous or missing, choose the simplest reasonable assumption and proceed.

# Scope & Discipline
- Do ONLY what is required to satisfy this step objective; avoid tangents and extra work.
- Use upstream step results already present in context; do not ask for them again.

# Tool Use
- Use tools only when they materially improve correctness or are required.
- Keep non-tool messages concise.

# Completion (Tool Call Only)
- When the step is complete, CALL finish_step exactly once with:
  {"result": <result>}
- Do NOT output the final result in assistant text.
- Stop immediately after calling finish_step.`;

export interface StepExecutorOptions {
  task: Task;
  step: Step;
  context: ProcessingContext;
  provider: BaseProvider;
  model: string;
  tools?: Tool[];
  systemPrompt?: string;
  maxTokenLimit?: number;
  maxIterations?: number;
}

export class StepExecutor {
  private history: Message[] = [];
  private step: Step;
  private task: Task;
  private tools: Tool[];
  private provider: BaseProvider;
  private model: string;
  private context: ProcessingContext;
  private systemPrompt: string;
  private maxTokenLimit: number;
  private maxIterations: number;
  private inConclusionStage = false;
  private result: unknown = null;
  private finishStepTool: FinishStepTool | null = null;

  constructor(opts: StepExecutorOptions) {
    this.task = opts.task;
    this.step = opts.step;
    this.context = opts.context;
    this.provider = opts.provider;
    this.model = opts.model;
    this.tools = opts.tools ? [...opts.tools] : [];
    this.systemPrompt = opts.systemPrompt ?? DEFAULT_EXECUTION_SYSTEM_PROMPT;
    this.maxTokenLimit = opts.maxTokenLimit ?? DEFAULT_TOKEN_LIMIT;
    this.maxIterations = opts.maxIterations ?? DEFAULT_MAX_ITERATIONS;
  }

  /**
   * Rough token estimate based on JSON serialized history length / 4.
   */
  private estimateTokens(): number {
    return Math.ceil(JSON.stringify(this.history).length / 4);
  }

  /**
   * Execute the step, yielding ProcessingMessages as progress updates.
   */
  async *execute(): AsyncGenerator<ProcessingMessage> {
    // Setup finish_step tool if we have an output schema
    if (this.step.outputSchema) {
      try {
        const schema = JSON.parse(this.step.outputSchema) as Record<string, unknown>;
        this.finishStepTool = new FinishStepTool(schema);
      } catch {
        this.finishStepTool = new FinishStepTool();
      }
    }

    // Initialize history with system prompt
    const systemContent = this.buildSystemPrompt();
    this.history.push({ role: "system" as const, content: systemContent });

    // Build user message with instructions and dependency results
    const userContent = await this.buildUserMessage();
    this.history.push({ role: "user" as const, content: userContent });

    // Yield task update: step started
    yield {
      type: "task_update",
      node_id: this.step.id,
      task: { id: this.task.id, title: this.task.title },
      step: { id: this.step.id, instructions: this.step.instructions },
      event: TaskUpdateEvent.StepStarted,
    } satisfies TaskUpdate;

    this.step.startTime = Date.now();
    let iterations = 0;

    while (!this.step.completed && iterations < this.maxIterations) {
      iterations++;

      // Check token budget
      if (this.estimateTokens() > this.maxTokenLimit && !this.inConclusionStage) {
        this.enterConclusionStage();
      }

      // Determine available tools
      const currentTools = this.getCurrentTools();
      const providerTools = currentTools.map((t) => t.toProviderTool());

      // Call LLM
      let content = "";
      const toolCalls: ToolCall[] = [];

      const stream = this.provider.generateMessages({
        messages: [...this.history],
        model: this.model,
        tools: providerTools.length > 0 ? providerTools : undefined,
      });

      for await (const item of stream) {
        if (isChunk(item)) {
          content += item.content ?? "";
          yield {
            type: "chunk",
            node_id: this.step.id,
            content: item.content,
            done: false,
          } satisfies Chunk;
        }
        if (isToolCall(item)) {
          toolCalls.push(item);
        }
      }

      // Append assistant message to history
      const assistantMsg: Message = { role: "assistant", content };
      if (toolCalls.length > 0) {
        assistantMsg.toolCalls = toolCalls;
      }
      this.history.push(assistantMsg);

      // Process tool calls
      if (toolCalls.length > 0) {
        for (const tc of toolCalls) {
          if (tc.name === "finish_step" && this.finishStepTool) {
            this.result = tc.args["result"] ?? tc.args;
            await this.context.storeStepResult(this.step.id, this.result);
            this.step.completed = true;
            this.step.endTime = Date.now();

            yield {
              type: "step_result",
              step: { id: this.step.id, instructions: this.step.instructions },
              result: this.result,
            } satisfies StepResult;
            break;
          } else {
            // Execute regular tool
            const tool = this.tools.find((t) => t.name === tc.name);
            let toolResult: unknown;
            try {
              toolResult = tool
                ? await tool.process(this.context, tc.args ?? {})
                : { error: `Unknown tool: ${tc.name}` };
            } catch (e) {
              toolResult = { error: String(e) };
            }

            const resultStr = JSON.stringify(toolResult);
            const truncated =
              resultStr.length > MAX_TOOL_RESULT_CHARS
                ? resultStr.slice(0, MAX_TOOL_RESULT_CHARS) + "\n[truncated]"
                : resultStr;

            this.history.push({
              role: "tool",
              toolCallId: tc.id,
              content: truncated,
            });

            yield {
              type: "tool_call_update",
              node_id: this.step.id,
              name: tc.name,
              args: tc.args,
            } satisfies ToolCallUpdate;
          }
        }
      } else if (content && !this.step.completed) {
        // No tool calls -- try extracting completion from text response
        const parsed = extractJSON(content);
        if (parsed && typeof parsed === "object") {
          const obj = parsed as Record<string, unknown>;
          // Accept if it looks like a completion payload
          const candidate = "result" in obj ? obj["result"] : parsed;
          this.result = candidate;
          await this.context.storeStepResult(this.step.id, this.result);
          this.step.completed = true;
          this.step.endTime = Date.now();

          yield {
            type: "step_result",
            step: { id: this.step.id, instructions: this.step.instructions },
            result: this.result,
          } satisfies StepResult;
        }
      }
    }

    // If we exhausted iterations without completing, yield a final event
    if (!this.step.completed) {
      this.step.completed = true;
      this.step.endTime = Date.now();
      yield {
        type: "task_update",
        node_id: this.step.id,
        task: { id: this.task.id, title: this.task.title },
        step: { id: this.step.id, instructions: this.step.instructions },
        event: TaskUpdateEvent.StepFailed,
      } satisfies TaskUpdate;
    }
  }

  /**
   * Get the tools available for the current stage.
   */
  private getCurrentTools(): Tool[] {
    if (this.inConclusionStage) {
      return this.finishStepTool ? [this.finishStepTool] : [];
    }
    const allTools = [...this.tools];
    if (this.finishStepTool) allTools.push(this.finishStepTool);
    return allTools;
  }

  /**
   * Transition to conclusion stage: restrict tools to finish_step only.
   */
  private enterConclusionStage(): void {
    this.inConclusionStage = true;
    this.history.push({
      role: "system",
      content:
        "TOKEN LIMIT APPROACHING. You must now synthesize your findings and call finish_step to complete this step. Do not make any more tool calls except finish_step.",
    });
  }

  /**
   * Build the system prompt for this step.
   */
  private buildSystemPrompt(): string {
    let prompt = this.systemPrompt;
    if (this.step.outputSchema) {
      prompt +=
        "\n\n# Output Schema\nThe result must match:\n```json\n" +
        this.step.outputSchema +
        "\n```";
    }
    prompt += `\n\nToday's date is ${new Date().toISOString().slice(0, 10)}`;
    return prompt;
  }

  /**
   * Build the initial user message with instructions and dependency results.
   */
  private async buildUserMessage(): Promise<string> {
    let msg = this.step.instructions;

    if (this.step.dependsOn.length > 0) {
      for (const depId of this.step.dependsOn) {
        const depResult = await this.context.loadStepResult(depId);
        if (depResult !== undefined && depResult !== null) {
          msg += `\n\nResult from step "${depId}":\n${JSON.stringify(depResult, null, 2)}`;
        }
      }
    }

    return msg;
  }

  /**
   * Get the captured result after execution completes.
   */
  getResult(): unknown {
    return this.result;
  }
}

// ---------------------------------------------------------------------------
// Helpers to discriminate ProviderStreamItem union members
// ---------------------------------------------------------------------------

function isChunk(item: ProviderStreamItem): item is { type: "chunk"; content?: string; done?: boolean } {
  return "type" in item && (item as unknown as Record<string, unknown>)["type"] === "chunk";
}

function isToolCall(item: ProviderStreamItem): item is ToolCall {
  return "name" in item && typeof (item as unknown as Record<string, unknown>)["name"] === "string" && "id" in item;
}
