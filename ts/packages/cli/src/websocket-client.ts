/**
 * WebSocket client for connecting the CLI to a NodeTool server.
 * Handles chat (stateful, with thread history) and inference (stateless, for agent mode).
 */

import WebSocket from "ws";

export type ChatEvent =
  | { type: "chunk"; content: string }
  | { type: "tool_call"; id: string; name: string; args: Record<string, unknown> }
  | { type: "error"; message: string }
  | { type: "done" };

export class WebSocketChatClient {
  private ws: WebSocket | null = null;
  private contentQueue: Array<Record<string, unknown>> = [];
  private contentWaiters: Array<(event: Record<string, unknown> | null) => void> = [];

  constructor(private readonly wsUrl: string) {}

  async connect(): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      const ws = new WebSocket(this.wsUrl);

      ws.on("open", () => {
        this.ws = ws;
        // Switch server to text/JSON mode
        ws.send(JSON.stringify({ command: "set_mode", data: { mode: "text" } }));
        resolve();
      });

      ws.on("error", (err) => reject(err));

      ws.on("message", (data: Buffer | string) => {
        try {
          const msg = JSON.parse(
            typeof data === "string" ? data : data.toString("utf8"),
          ) as Record<string, unknown>;
          this.handleMessage(msg);
        } catch {
          // ignore malformed messages
        }
      });

      ws.on("close", () => {
        this.ws = null;
        // Unblock all pending waiters
        for (const waiter of this.contentWaiters) waiter(null);
        this.contentWaiters = [];
      });
    });
  }

  private handleMessage(msg: Record<string, unknown>): void {
    const type = typeof msg.type === "string" ? msg.type : null;

    // Auto-respond to server pings
    if (type === "ping") {
      this.send({ type: "pong", ts: Date.now() / 1000 });
      return;
    }

    // Route content events to waiting generators
    if (this.isContentEvent(type)) {
      const waiter = this.contentWaiters.shift();
      if (waiter) {
        waiter(msg);
      } else {
        this.contentQueue.push(msg);
      }
    }
    // Ignore: system_stats, command acks (no type field), etc.
  }

  private isContentEvent(type: string | null): boolean {
    return (
      type === "chunk" ||
      type === "message" ||
      type === "tool_call" ||
      type === "job_update" ||
      type === "error" ||
      type === "inference_done" ||
      type === "generation_stopped"
    );
  }

  private send(data: unknown): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  private nextContent(): Promise<Record<string, unknown> | null> {
    if (this.contentQueue.length > 0) {
      return Promise.resolve(this.contentQueue.shift()!);
    }
    return new Promise<Record<string, unknown> | null>((resolve) => {
      this.contentWaiters.push(resolve);
    });
  }

  /** Stateful chat — messages are saved to DB, thread history is loaded. */
  async *chat(
    content: string,
    threadId: string,
    model: string,
    provider: string,
    tools?: unknown[],
  ): AsyncGenerator<ChatEvent> {
    this.send({ command: "chat_message", data: { role: "user", content, thread_id: threadId, model, provider, tools: tools ?? [] } });
    while (true) {
      const event = await this.nextContent();
      if (!event) {
        yield { type: "done" };
        return;
      }
      const type = event.type as string;
      if (type === "chunk") {
        // Done chunk signals completion — matches Python's {"type": "chunk", "done": true}
        if (event.done === true) {
          yield { type: "done" };
          return;
        }
        yield { type: "chunk", content: typeof event.content === "string" ? event.content : "" };
      } else if (type === "message") {
        // Message events (assistant tool calls, tool results, final assistant) — skip in CLI stream
        // The final assistant message after done chunk won't reach here since we return on done
        continue;
      } else if (type === "job_update" || type === "generation_stopped") {
        yield { type: "done" };
        return;
      } else if (type === "error") {
        yield { type: "error", message: typeof event.message === "string" ? event.message : "Unknown error" };
        return;
      }
    }
  }

  /** Stateless inference — takes full messages array, streams back, no DB. Used by agent mode. */
  async *inference(
    messages: unknown[],
    model: string,
    provider: string,
    tools?: unknown[],
  ): AsyncGenerator<ChatEvent> {
    this.send({ command: "inference", data: { messages, model, provider, tools: tools ?? [] } });
    while (true) {
      const event = await this.nextContent();
      if (!event) {
        yield { type: "done" };
        return;
      }
      const type = event.type as string;
      if (type === "chunk") {
        yield { type: "chunk", content: typeof event.content === "string" ? event.content : "" };
      } else if (type === "tool_call") {
        yield {
          type: "tool_call",
          id: typeof event.id === "string" ? event.id : "",
          name: typeof event.name === "string" ? event.name : "",
          args: (event.args ?? {}) as Record<string, unknown>,
        };
      } else if (type === "inference_done" || type === "generation_stopped") {
        yield { type: "done" };
        return;
      } else if (type === "error") {
        yield { type: "error", message: typeof event.message === "string" ? event.message : "Unknown error" };
        return;
      }
    }
  }

  /** Stop in-progress generation. Pass threadId for chat, omit for inference. */
  stop(threadId?: string): void {
    const data: Record<string, unknown> = {};
    if (threadId) data["thread_id"] = threadId;
    this.send({ command: "stop", data });
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}
