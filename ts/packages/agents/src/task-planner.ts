/**
 * TaskPlanner -- uses an LLM to decompose an objective into a Task with Steps.
 *
 * Port of src/nodetool/agents/task_planner.py (simplified)
 */

import type { BaseProvider, ProcessingContext, Message } from "@nodetool/runtime";
import type { ProcessingMessage, Chunk } from "@nodetool/protocol";
import type { Step, Task } from "./types.js";
import type { Tool } from "./tools/base-tool.js";
import { extractJSON } from "./utils/json-parser.js";
import { randomUUID } from "node:crypto";

const DEFAULT_PLANNING_SYSTEM_PROMPT = `You are a TaskArchitect that transforms user objectives into executable Task plans.

A Task has a title and a list of Steps. Each Step has:
- id: unique snake_case identifier
- instructions: clear, actionable instructions
- dependsOn: list of step IDs this depends on ([] for none)
- outputSchema (optional): JSON schema string for the step output
- tools (optional): list of tool names this step can use

Requirements:
- All step IDs must be unique
- Dependencies must form a valid DAG (no cycles)
- All referenced dependency IDs must exist as step IDs
- Steps should be atomic (smallest executable units)

Call the create_task tool with your plan.`;

const PLAN_CREATION_PROMPT_TEMPLATE = `Create an executable Task for this objective using the create_task tool.

Objective: {{objective}}

Available tools:
{{toolsInfo}}

Output schema:
{{outputSchema}}`;

/**
 * Schema for the create_task tool used by the planner.
 */
const CREATE_TASK_SCHEMA = {
  type: "object",
  properties: {
    title: { type: "string", description: "Task title" },
    steps: {
      type: "array",
      items: {
        type: "object",
        properties: {
          id: { type: "string" },
          instructions: { type: "string" },
          depends_on: { type: "array", items: { type: "string" } },
          output_schema: { type: "string" },
          tools: { type: "array", items: { type: "string" } },
        },
        required: ["id", "instructions", "depends_on"],
      },
    },
  },
  required: ["title", "steps"],
};

export interface TaskPlannerOptions {
  provider: BaseProvider;
  model: string;
  tools?: Tool[];
  systemPrompt?: string;
  outputSchema?: Record<string, unknown>;
}

export class TaskPlanner {
  private provider: BaseProvider;
  private model: string;
  private tools: Tool[];
  private systemPrompt: string;
  private outputSchema: Record<string, unknown> | undefined;

  constructor(opts: TaskPlannerOptions) {
    this.provider = opts.provider;
    this.model = opts.model;
    this.tools = opts.tools ?? [];
    this.systemPrompt = opts.systemPrompt ?? DEFAULT_PLANNING_SYSTEM_PROMPT;
    this.outputSchema = opts.outputSchema;
  }

  /**
   * Generate a Task plan from an objective.
   */
  async *plan(
    objective: string,
    _context: ProcessingContext,
  ): AsyncGenerator<ProcessingMessage, Task | null> {
    const toolsInfo = this.tools.length > 0
      ? this.tools.map((t) => `- ${t.name}: ${t.description}`).join("\n")
      : "No execution tools available.";

    const userPrompt = PLAN_CREATION_PROMPT_TEMPLATE
      .replace("{{objective}}", objective)
      .replace("{{toolsInfo}}", toolsInfo)
      .replace(
        "{{outputSchema}}",
        this.outputSchema ? JSON.stringify(this.outputSchema, null, 2) : "None specified",
      );

    const messages: Message[] = [
      { role: "system", content: this.systemPrompt },
      { role: "user", content: userPrompt },
    ];

    // Call LLM with the create_task tool
    let content = "";
    let taskData: Record<string, unknown> | null = null;

    const stream = this.provider.generateMessages({
      messages,
      model: this.model,
      tools: [
        {
          name: "create_task",
          description: "Create an executable task with steps.",
          inputSchema: CREATE_TASK_SCHEMA,
        },
      ],
    });

    for await (const item of stream) {
      if ("type" in item && (item as unknown as Record<string, unknown>)["type"] === "chunk") {
        const chunk = item as { content?: string };
        content += chunk.content ?? "";
        yield {
          type: "chunk",
          content: chunk.content,
          done: false,
        } satisfies Chunk;
      }
      if ("name" in item && item.name === "create_task") {
        taskData = item.args as Record<string, unknown>;
      }
    }

    // If no tool call, try extracting from text
    if (!taskData && content) {
      const parsed = extractJSON(content);
      if (parsed && typeof parsed === "object") {
        taskData = parsed as Record<string, unknown>;
      }
    }

    if (!taskData) {
      yield {
        type: "chunk",
        content: "\nFailed to generate a task plan.\n",
        done: true,
      } satisfies Chunk;
      return null;
    }

    // Build Task from the LLM response
    const task = this.buildTask(taskData);

    // Validate DAG
    if (!this.validateDAG(task)) {
      yield {
        type: "chunk",
        content: "\nGenerated plan has circular dependencies.\n",
        done: true,
      } satisfies Chunk;
      return null;
    }

    yield {
      type: "planning_update",
      phase: "complete",
      status: "success",
      content: `Plan created: ${task.title} (${task.steps.length} steps)`,
    } as ProcessingMessage;

    return task;
  }

  /**
   * Build a Task object from raw LLM output data.
   */
  private buildTask(data: Record<string, unknown>): Task {
    const title = typeof data["title"] === "string" ? data["title"] : "Untitled Task";
    const rawSteps = Array.isArray(data["steps"]) ? data["steps"] : [];

    const steps: Step[] = rawSteps.map((s: unknown) => {
      const raw = s as Record<string, unknown>;
      return {
        id: typeof raw["id"] === "string" ? raw["id"] : randomUUID(),
        instructions: typeof raw["instructions"] === "string" ? raw["instructions"] : "",
        completed: false,
        dependsOn: Array.isArray(raw["depends_on"])
          ? (raw["depends_on"] as string[])
          : Array.isArray(raw["dependsOn"])
            ? (raw["dependsOn"] as string[])
            : [],
        outputSchema: typeof raw["output_schema"] === "string"
          ? raw["output_schema"]
          : typeof raw["outputSchema"] === "string"
            ? raw["outputSchema"]
            : undefined,
        tools: Array.isArray(raw["tools"]) ? (raw["tools"] as string[]) : undefined,
        logs: [],
      };
    });

    return {
      id: randomUUID(),
      title,
      steps,
    };
  }

  /**
   * Validate that the task steps form a valid DAG (no cycles).
   */
  private validateDAG(task: Task): boolean {
    const stepIds = new Set(task.steps.map((s) => s.id));
    const visited = new Set<string>();
    const inStack = new Set<string>();

    const stepMap = new Map(task.steps.map((s) => [s.id, s]));

    const hasCycle = (id: string): boolean => {
      if (inStack.has(id)) return true;
      if (visited.has(id)) return false;

      visited.add(id);
      inStack.add(id);

      const step = stepMap.get(id);
      if (step) {
        for (const dep of step.dependsOn) {
          if (stepIds.has(dep) && hasCycle(dep)) {
            return true;
          }
        }
      }

      inStack.delete(id);
      return false;
    };

    for (const step of task.steps) {
      if (hasCycle(step.id)) return false;
    }

    return true;
  }
}
