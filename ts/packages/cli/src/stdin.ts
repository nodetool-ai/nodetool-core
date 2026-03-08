/**
 * Stdin mode — read chat messages (or agent objectives) from stdin, write responses to stdout.
 *
 * Activated automatically when stdin is not a TTY (i.e. piped input).
 * Each non-empty line is sent as a user message / agent objective.
 * The assistant reply streams to stdout, followed by a newline.
 * Conversation history is preserved across lines within a single session.
 *
 * Supports both direct-provider mode and WebSocket server mode (--url).
 */

import readline from "node:readline";
import type { Message } from "@nodetool/runtime";
import { ProcessingContext } from "@nodetool/runtime";
import { processChat } from "@nodetool/chat";
import { Agent } from "@nodetool/agents";
import { createProvider, WebSocketProvider } from "./providers.js";
import { WebSocketChatClient } from "./websocket-client.js";
import { getSecret } from "@nodetool/security";

export interface StdinModeOptions {
  provider: string;
  model: string;
  workspaceDir: string;
  agentMode?: boolean;
  wsUrl?: string;
}

export async function runStdinMode(opts: StdinModeOptions): Promise<void> {
  const wsClient = opts.wsUrl ? new WebSocketChatClient(opts.wsUrl) : null;
  if (wsClient) {
    await wsClient.connect();
  }

  // Direct mode: create provider once for the session
  const directProvider = wsClient ? null : await createProvider(opts.provider);

  const threadId = crypto.randomUUID();
  const chatHistory: Message[] = [];

  const rl = readline.createInterface({ input: process.stdin, terminal: false });

  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    if (opts.agentMode) {
      // --- Agent mode: each line is an objective ---
      const prov = wsClient
        ? new WebSocketProvider(wsClient, opts.model, opts.provider)
        : directProvider!;

      const agent = new Agent({
        name: "stdin-agent",
        objective: trimmed,
        provider: prov,
        model: opts.model,
        tools: [],
      });

      const ctx = new ProcessingContext({ jobId: crypto.randomUUID(), userId: "1", workspaceDir: opts.workspaceDir, secretResolver: getSecret });
      let taskResult: string | null = null;

      for await (const msg of agent.execute(ctx)) {
        if (msg.type === "chunk") {
          if (taskResult === null) {
            process.stdout.write((msg as { content?: string }).content ?? "");
          }
        } else if (msg.type === "step_result") {
          const sr = msg as { result: unknown; is_task_result: boolean };
          if (sr.is_task_result) {
            taskResult = typeof sr.result === "string" ? sr.result : JSON.stringify(sr.result, null, 2);
          }
        } else if (msg.type === "planning_update") {
          process.stderr.write(`[planning] ${(msg as { content: string }).content.slice(0, 80)}\n`);
        } else if (msg.type === "task_update") {
          process.stderr.write(`[task] ${(msg as { event: string }).event}\n`);
        } else if (msg.type === "tool_call_update") {
          process.stderr.write(`[tool] ${(msg as { name: string }).name}\n`);
        }
      }

      if (taskResult !== null) {
        process.stdout.write(taskResult);
      }

    } else if (wsClient) {
      // --- Regular chat via WebSocket ---
      for await (const event of wsClient.chat(trimmed, threadId, opts.model, opts.provider)) {
        if (event.type === "chunk") {
          process.stdout.write(event.content);
        } else if (event.type === "error") {
          process.stderr.write(`Error: ${event.message}\n`);
          break;
        } else if (event.type === "done") {
          break;
        }
      }

    } else {
      // --- Regular chat via direct provider ---
      await processChat({
        userInput: trimmed,
        messages: chatHistory,
        model: opts.model,
        provider: directProvider!,
        context: new ProcessingContext({ jobId: crypto.randomUUID(), userId: "1", workspaceDir: opts.workspaceDir, secretResolver: getSecret }),
        tools: [],
        callbacks: {
          onChunk: (text) => { process.stdout.write(text); },
        },
      });
    }

    process.stdout.write("\n");
  }

  wsClient?.disconnect();
}
