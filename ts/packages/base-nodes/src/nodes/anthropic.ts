/**
 * Anthropic Claude Agent node.
 * Uses the Claude Agent SDK to run Claude with tool-use capabilities.
 */

import { BaseNode } from "@nodetool/node-sdk";
import type { NodeClass } from "@nodetool/node-sdk";

export class ClaudeAgentNode extends BaseNode {
  static readonly nodeType = "anthropic.agents.ClaudeAgent";
  static readonly title = "Claude Agent";
  static readonly description =
    "Run Claude as an agent in a sandboxed environment with tool use capabilities. " +
    "claude, agent, ai, anthropic, sandbox, assistant";

  defaults() {
    return {
      prompt: "",
      model: { provider: "", id: "" },
      system_prompt: "",
      max_turns: 20,
      allowed_tools: ["Read", "Write", "Bash"],
      permission_mode: "acceptEdits",
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const prompt = String(inputs.prompt ?? this._props.prompt ?? "").trim();
    if (!prompt) throw new Error("Prompt is required");

    const apiKey =
      (inputs._secrets as Record<string, string>)?.ANTHROPIC_API_KEY ||
      process.env.ANTHROPIC_API_KEY ||
      "";
    if (!apiKey) throw new Error("ANTHROPIC_API_KEY is not configured");

    const model = (inputs.model ?? this._props.model ?? {}) as Record<string, unknown>;
    const modelId = String(model.id || "claude-sonnet-4-20250514");
    const systemPrompt = String(inputs.system_prompt ?? this._props.system_prompt ?? "");
    const maxTurns = Number(inputs.max_turns ?? this._props.max_turns ?? 20);
    const permissionMode = String(inputs.permission_mode ?? this._props.permission_mode ?? "acceptEdits");
    const allowedTools = (inputs.allowed_tools ?? this._props.allowed_tools ?? ["Read", "Write", "Bash"]) as string[];

    // Use Claude Agent SDK via dynamic import
    // The SDK provides query() async iterator for streaming agent responses
    try {
      const sdk = await import("claude-agent-sdk" as string);
      const { query } = sdk;

      const options = {
        model: modelId,
        system_prompt: systemPrompt || undefined,
        max_turns: maxTurns,
        allowed_tools: allowedTools,
        permission_mode: permissionMode,
        env: { ANTHROPIC_API_KEY: apiKey },
      };

      let fullText = "";
      for await (const message of query({ prompt, options })) {
        if (message?.content) {
          for (const content of message.content) {
            if (content?.text) {
              fullText += content.text;
            }
          }
        }
      }

      return { text: fullText };
    } catch (e: unknown) {
      const err = e as Error;
      if (err.message?.includes("Cannot find module")) {
        throw new Error(
          "Claude Agent SDK (claude-agent-sdk) is not installed. " +
          "Install it to use the ClaudeAgent node."
        );
      }
      throw new Error(`Claude Agent error: ${err.message}`);
    }
  }
}

export const ANTHROPIC_NODES: readonly NodeClass[] = [ClaudeAgentNode];
