import { randomUUID } from "node:crypto";
import { pack, unpack } from "msgpackr";
import { WorkflowRunner, type NodeExecutor } from "@nodetool/kernel";
import {
  Job,
  Message,
  ModelChangeEvent,
  ModelObserver,
  Thread,
  Workflow,
  type DBModel,
} from "@nodetool/models";
import type {
  ProviderTool,
  Message as ProviderMessage,
  BaseProvider,
  ProcessingContext,
} from "@nodetool/runtime";
import { ProcessingContext as RuntimeProcessingContext } from "@nodetool/runtime";
import type {
  UnifiedCommandType,
  WebSocketCommandEnvelope,
  WebSocketMode,
} from "@nodetool/protocol";

export interface WebSocketReceiveFrame {
  type: string;
  bytes?: Uint8Array | null;
  text?: string | null;
}

export interface WebSocketConnection {
  accept(): Promise<void>;
  receive(): Promise<WebSocketReceiveFrame>;
  sendBytes(data: Uint8Array): Promise<void>;
  sendText(data: string): Promise<void>;
  close(code?: number, reason?: string): Promise<void>;
  clientState?: "connected" | "disconnected";
  applicationState?: "connected" | "disconnected";
}

export interface RunJobRequest {
  job_id?: string;
  workflow_id?: string;
  user_id?: string;
  auth_token?: string;
  params?: Record<string, unknown>;
  graph?: { nodes: Array<Record<string, unknown>>; edges: Array<Record<string, unknown>> };
  explicit_types?: boolean;
}

interface ActiveJob {
  jobId: string;
  workflowId: string | null;
  context: ProcessingContext;
  runner: WorkflowRunner;
  finished: boolean;
  status: "running" | "completed" | "failed" | "cancelled";
  error?: string;
  streamTask?: Promise<void>;
}

class ToolBridge {
  private waiters = new Map<string, (value: Record<string, unknown>) => void>();

  createWaiter(toolCallId: string): Promise<Record<string, unknown>> {
    return new Promise((resolve) => {
      this.waiters.set(toolCallId, resolve);
    });
  }

  resolveResult(toolCallId: string, payload: Record<string, unknown>): void {
    const resolve = this.waiters.get(toolCallId);
    if (!resolve) return;
    this.waiters.delete(toolCallId);
    resolve(payload);
  }

  cancelAll(): void {
    this.waiters.clear();
  }
}

export interface UnifiedWebSocketRunnerOptions {
  userId?: string;
  authToken?: string;
  defaultModel?: string;
  defaultProvider?: string;
  resolveExecutor: (node: { id: string; type: string; [key: string]: unknown }) => NodeExecutor;
  resolveProvider?: (providerId: string) => Promise<BaseProvider>;
  getSystemStats?: () => Record<string, unknown>;
  workspaceResolver?: (workflowId: string, userId: string) => Promise<string | null>;
}

export class UnifiedWebSocketRunner {
  websocket: WebSocketConnection | null = null;
  mode: WebSocketMode = "binary";
  userId: string | null;
  authToken: string | null;

  private defaultModel: string;
  private defaultProvider: string;
  private resolveExecutor: UnifiedWebSocketRunnerOptions["resolveExecutor"];
  private resolveProvider?: UnifiedWebSocketRunnerOptions["resolveProvider"];
  private getSystemStats: () => Record<string, unknown>;
  private workspaceResolver?: UnifiedWebSocketRunnerOptions["workspaceResolver"];

  private sendLock: Promise<void> = Promise.resolve();
  private activeJobs = new Map<string, ActiveJob>();
  private currentTask: Promise<void> | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private statsTimer: NodeJS.Timeout | null = null;
  private chatRequestSeq = 0;
  private clientToolsManifest: Record<string, Record<string, unknown>> = {};
  private toolBridge = new ToolBridge();
  private observerRegistered = false;

  constructor(options: UnifiedWebSocketRunnerOptions) {
    this.userId = options.userId ?? null;
    this.authToken = options.authToken ?? null;
    this.defaultModel = options.defaultModel ?? "gpt-oss:20b";
    this.defaultProvider = options.defaultProvider ?? "ollama";
    this.resolveExecutor = options.resolveExecutor;
    this.resolveProvider = options.resolveProvider;
    this.workspaceResolver = options.workspaceResolver;
    this.getSystemStats =
      options.getSystemStats ??
      (() => ({
        timestamp: Date.now(),
        process_uptime_sec: process.uptime(),
        memory: process.memoryUsage(),
      }));
  }

  async connect(websocket: WebSocketConnection, userId?: string, authToken?: string): Promise<void> {
    if (userId) this.userId = userId;
    if (authToken) this.authToken = authToken;
    this.userId = this.userId ?? "1";

    await websocket.accept();
    this.websocket = websocket;

    this.startHeartbeat();
    this.startStatsBroadcast();
    this.registerObserver();
  }

  async disconnect(): Promise<void> {
    this.stopHeartbeat();
    this.stopStatsBroadcast();
    this.unregisterObserver();
    this.toolBridge.cancelAll();

    this.currentTask = null;
    for (const [jobId, job] of this.activeJobs) {
      job.runner.cancel();
      this.activeJobs.delete(jobId);
    }

    if (this.websocket) {
      try {
        await this.websocket.close();
      } catch {
        // no-op
      }
    }
    this.websocket = null;
  }

  private serializeForJson(value: unknown): unknown {
    if (value instanceof Uint8Array) return Array.from(value);
    if (value instanceof Date) return value.toISOString();
    if (Array.isArray(value)) return value.map((v) => this.serializeForJson(v));
    if (value && typeof value === "object") {
      const out: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
        out[k] = this.serializeForJson(v);
      }
      return out;
    }
    return value;
  }

  async sendMessage(message: Record<string, unknown>): Promise<void> {
    if (!this.websocket) return;
    if (this.websocket.clientState === "disconnected" || this.websocket.applicationState === "disconnected") {
      return;
    }

    const payload = this.mode === "text" ? (this.serializeForJson(message) as Record<string, unknown>) : message;

    const prev = this.sendLock;
    let release!: () => void;
    this.sendLock = new Promise<void>((resolve) => {
      release = resolve;
    });

    await prev;
    try {
      if (this.mode === "binary") {
        await this.websocket.sendBytes(pack(payload));
      } else {
        await this.websocket.sendText(JSON.stringify(payload));
      }
    } finally {
      release();
    }
  }

  async receiveMessage(): Promise<Record<string, unknown> | null> {
    if (!this.websocket) {
      throw new Error("WebSocket is not connected");
    }

    const message = await this.websocket.receive();
    if (message.type === "websocket.disconnect") return null;

    if (message.bytes) {
      this.mode = "binary";
      return unpack(message.bytes) as Record<string, unknown>;
    }
    if (message.text) {
      this.mode = "text";
      return JSON.parse(message.text) as Record<string, unknown>;
    }
    return null;
  }

  private async getWorkflowGraph(req: RunJobRequest): Promise<{ nodes: Array<Record<string, unknown>>; edges: Array<Record<string, unknown>> }> {
    if (req.graph) return req.graph;
    if (!req.workflow_id || !this.userId) {
      throw new Error("workflow_id or graph is required");
    }
    const workflow = await Workflow.find(this.userId, req.workflow_id);
    if (!workflow) throw new Error(`Workflow not found: ${req.workflow_id}`);
    return workflow.graph as { nodes: Array<Record<string, unknown>>; edges: Array<Record<string, unknown>> };
  }

  async runJob(req: RunJobRequest): Promise<void> {
    const userId = req.user_id ?? this.userId ?? "1";
    const workflowId = req.workflow_id ?? null;
    const jobId = req.job_id ?? randomUUID();
    const graph = await this.getWorkflowGraph(req);
    const workspaceDir = workflowId && this.workspaceResolver ? await this.workspaceResolver(workflowId, userId) : null;

    const context = new RuntimeProcessingContext({
      jobId,
      workflowId,
      userId,
      workspaceDir,
      assetOutputMode: this.mode === "text" ? "data_uri" : "raw",
    });

    const runner = new WorkflowRunner(jobId, {
      resolveExecutor: (node) => this.resolveExecutor(node as { id: string; type: string; [key: string]: unknown }),
      executionContext: context,
    });

    const active: ActiveJob = {
      jobId,
      workflowId,
      context,
      runner,
      finished: false,
      status: "running",
    };
    this.activeJobs.set(jobId, active);

    try {
      const existing = await Job.get(jobId);
      if (!existing) {
        await Job.create({
          id: jobId,
          workflow_id: workflowId ?? "",
          user_id: userId,
          status: "running",
          params: req.params ?? {},
          graph,
        });
      }
    } catch {
      // Persistence is best-effort in TS runtime mode.
    }

    const executePromise = runner.run(
      {
        job_id: jobId,
        workflow_id: workflowId ?? undefined,
        params: req.params ?? {},
      },
      graph as unknown as { nodes: Array<{ id: string; type: string; [key: string]: unknown }>; edges: Array<{ id: string; source: string; target: string; sourceHandle: string; targetHandle: string; type?: "data" | "control" }> },
    );

    active.streamTask = this.streamJobMessages(active, executePromise);
  }

  private async streamJobMessages(active: ActiveJob, executePromise: Promise<{ status: "completed" | "failed" | "cancelled"; error?: string }>): Promise<void> {
    let terminalSeen = false;
    await this.sendMessage({ type: "job_update", status: "running", job_id: active.jobId, workflow_id: active.workflowId });

    void executePromise
      .then((result) => {
        active.status = result.status;
        active.error = result.error;
      })
      .catch((err) => {
        active.status = "failed";
        active.error = err instanceof Error ? err.message : String(err);
      })
      .finally(() => {
        active.finished = true;
      });

    while (!active.finished || active.context.hasMessages()) {
      while (active.context.hasMessages()) {
        const msg = active.context.popMessage();
        if (!msg) break;
        const outbound: Record<string, unknown> = {
          ...(msg as unknown as Record<string, unknown>),
          job_id: (msg as unknown as Record<string, unknown>).job_id ?? active.jobId,
          workflow_id: (msg as unknown as Record<string, unknown>).workflow_id ?? active.workflowId,
        };
        await this.sendMessage(outbound);
        if (outbound.type === "job_update") {
          const status = String(outbound.status ?? "");
          if (["completed", "failed", "cancelled", "error", "suspended"].includes(status)) {
            terminalSeen = true;
          }
        }
      }
      if (!active.finished) {
        await new Promise((resolve) => setTimeout(resolve, 10));
      }
    }

    if (!terminalSeen) {
      await this.sendMessage({
        type: "job_update",
        status: active.status,
        job_id: active.jobId,
        workflow_id: active.workflowId,
        error: active.error,
      });
    }

    this.activeJobs.delete(active.jobId);
  }

  async reconnectJob(jobId: string, workflowId?: string): Promise<void> {
    const active = this.activeJobs.get(jobId);
    if (!active) {
      throw new Error(`Job ${jobId} not found`);
    }

    await this.sendMessage({
      type: "job_update",
      status: active.status,
      job_id: jobId,
      workflow_id: workflowId ?? active.workflowId,
    });

    for (const status of Object.values(active.context.getNodeStatuses())) {
      await this.sendMessage({ ...(status as unknown as Record<string, unknown>), job_id: jobId, workflow_id: workflowId ?? active.workflowId });
    }
    for (const status of Object.values(active.context.getEdgeStatuses())) {
      await this.sendMessage({ ...(status as unknown as Record<string, unknown>), job_id: jobId, workflow_id: workflowId ?? active.workflowId });
    }
  }

  async resumeJob(jobId: string, workflowId?: string): Promise<void> {
    await this.reconnectJob(jobId, workflowId);
  }

  async cancelJob(jobId: string, workflowId?: string): Promise<Record<string, unknown>> {
    if (!jobId) {
      return { error: "No job_id provided" };
    }

    const active = this.activeJobs.get(jobId);
    if (!active) {
      return { error: "Job not found or already completed", job_id: jobId, workflow_id: workflowId ?? "" };
    }

    active.runner.cancel();
    active.finished = true;
    active.status = "cancelled";
    return {
      message: "Job cancellation requested",
      job_id: jobId,
      workflow_id: workflowId ?? active.workflowId ?? "",
    };
  }

  getStatus(jobId?: string): Record<string, unknown> {
    if (jobId) {
      const active = this.activeJobs.get(jobId);
      if (!active) {
        return { status: "not_found", job_id: jobId };
      }
      return {
        status: active.status,
        job_id: active.jobId,
        workflow_id: active.workflowId,
      };
    }

    return {
      active_jobs: Array.from(this.activeJobs.values()).map((job) => ({
        job_id: job.jobId,
        workflow_id: job.workflowId,
        status: job.status,
      })),
    };
  }

  async clearModels(): Promise<Record<string, unknown>> {
    return { message: "Model clearing is managed by provider implementations in TS runtime" };
  }

  private async ensureThreadExists(threadId?: string): Promise<string> {
    const userId = this.userId ?? "1";
    if (!threadId) {
      const thread = await Thread.create({ user_id: userId, title: "" });
      return thread.id;
    }
    const existing = await Thread.find(userId, threadId);
    if (existing) return existing.id;
    const thread = await Thread.create({ id: threadId, user_id: userId, title: "" });
    return thread.id;
  }

  private dbMessageToProviderMessage(m: Message): ProviderMessage {
    return {
      role: (m.role as ProviderMessage["role"]) ?? "user",
      content: m.content ?? "",
      toolCallId: null,
      toolCalls: Array.isArray(m.tool_calls) ? (m.tool_calls as Array<{ id: string; name: string; args: Record<string, unknown> }>) : null,
      threadId: m.thread_id,
    };
  }

  async handleChatMessage(data: Record<string, unknown>, requestSeq?: number): Promise<void> {
    const threadId = await this.ensureThreadExists(typeof data.thread_id === "string" ? data.thread_id : undefined);
    data.thread_id = threadId;
    const providerId = (typeof data.provider === "string" ? data.provider : this.defaultProvider) as string;
    const model = (typeof data.model === "string" ? data.model : this.defaultModel) as string;
    const content = typeof data.content === "string" ? data.content : "";

    await Message.create({
      user_id: this.userId ?? "1",
      thread_id: threadId,
      role: "user",
      content,
      provider: providerId,
      model,
      workflow_id: typeof data.workflow_id === "string" ? data.workflow_id : null,
    });

    if (requestSeq !== undefined && requestSeq !== this.chatRequestSeq) return;

    if (!this.resolveProvider) {
      await this.sendMessage({ type: "error", message: "No provider resolver configured", thread_id: threadId });
      return;
    }

    const [messages] = await Message.paginate(threadId, { limit: 1000 });
    const providerMessages = messages.map((m) => this.dbMessageToProviderMessage(m));
    const provider = await this.resolveProvider(providerId);

    let finalText = "";
    const tools: ProviderTool[] = [];
    for await (const item of provider.generateMessages({ messages: providerMessages, model, tools })) {
      if ("type" in item && item.type === "chunk") {
        const contentPart = typeof item.content === "string" ? item.content : "";
        finalText += contentPart;
        await this.sendMessage({ ...item, thread_id: threadId });
      } else {
        const toolItem = item as { id: string; name: string; args: Record<string, unknown> };
        await this.sendMessage({
          type: "tool_call_update",
          thread_id: threadId,
          tool_call_id: toolItem.id,
          name: toolItem.name,
          args: toolItem.args,
        });
      }
    }

    await Message.create({
      user_id: this.userId ?? "1",
      thread_id: threadId,
      role: "assistant",
      content: finalText,
      provider: providerId,
      model,
      workflow_id: typeof data.workflow_id === "string" ? data.workflow_id : null,
    });
  }

  async handleCommand(command: WebSocketCommandEnvelope): Promise<Record<string, unknown>> {
    const data = command.data ?? {};
    const jobId = typeof data.job_id === "string" ? data.job_id : undefined;
    const workflowId = typeof data.workflow_id === "string" ? data.workflow_id : undefined;

    switch (command.command as UnifiedCommandType) {
      case "clear_models":
        return this.clearModels();
      case "run_job":
        await this.runJob(data as unknown as RunJobRequest);
        return { message: "Job started", workflow_id: workflowId ?? null };
      case "reconnect_job":
        if (!jobId) return { error: "job_id is required" };
        void this.reconnectJob(jobId, workflowId);
        return { message: `Reconnecting to job ${jobId}`, job_id: jobId, workflow_id: workflowId ?? null };
      case "resume_job":
        if (!jobId) return { error: "job_id is required" };
        void this.resumeJob(jobId, workflowId);
        return { message: `Resumption initiated for job ${jobId}`, job_id: jobId, workflow_id: workflowId ?? null };
      case "stream_input":
        if (!jobId) return { error: "job_id is required" };
        {
          const active = this.activeJobs.get(jobId);
          if (!active) return { error: "No active job/context" };
          const inputName = typeof data.input === "string" ? data.input : "";
          if (!inputName.trim()) return { error: "Invalid input name" };
          const value = data.value;
          const handle = typeof data.handle === "string" ? data.handle : undefined;
          try {
            await active.runner.pushInputValue(inputName, value, handle);
            return {
              message: "Input item streamed",
              job_id: jobId,
              workflow_id: workflowId ?? active.workflowId,
            };
          } catch (err) {
            return {
              error: err instanceof Error ? err.message : String(err),
              job_id: jobId,
              workflow_id: workflowId ?? active.workflowId,
            };
          }
        }
      case "end_input_stream":
        if (!jobId) return { error: "job_id is required" };
        {
          const active = this.activeJobs.get(jobId);
          if (!active) return { error: "No active job/context" };
          const inputName = typeof data.input === "string" ? data.input : "";
          if (!inputName.trim()) return { error: "Invalid input name" };
          const handle = typeof data.handle === "string" ? data.handle : undefined;
          try {
            active.runner.finishInputStream(inputName, handle);
            return {
              message: "Input stream ended",
              job_id: jobId,
              workflow_id: workflowId ?? active.workflowId,
            };
          } catch (err) {
            return {
              error: err instanceof Error ? err.message : String(err),
              job_id: jobId,
              workflow_id: workflowId ?? active.workflowId,
            };
          }
        }
      case "cancel_job":
        if (!jobId) return { error: "job_id is required" };
        return this.cancelJob(jobId, workflowId);
      case "get_status":
        return this.getStatus(jobId);
      case "set_mode": {
        const mode = data.mode;
        if (mode !== "binary" && mode !== "text") {
          return { error: "mode must be binary or text" };
        }
        this.mode = mode;
        return { message: `Mode set to ${mode}` };
      }
      case "chat_message": {
        const threadId = data.thread_id;
        if (typeof threadId !== "string" || threadId.length === 0) {
          return { error: "thread_id is required for chat_message command" };
        }
        this.chatRequestSeq += 1;
        const seq = this.chatRequestSeq;
        this.currentTask = this.handleChatMessage(data, seq);
        void this.currentTask.catch(async (err) => {
          await this.sendMessage({ type: "error", message: err instanceof Error ? err.message : String(err) });
        });
        return { message: "Chat message processing started", thread_id: threadId };
      }
      case "stop": {
        const threadId = typeof data.thread_id === "string" ? data.thread_id : undefined;
        if (!jobId && !threadId) {
          return { error: "job_id or thread_id is required for stop command" };
        }
        if (threadId) {
          this.chatRequestSeq += 1;
          this.currentTask = null;
        }
        if (jobId) {
          const active = this.activeJobs.get(jobId);
          if (active) {
            active.runner.cancel();
            active.finished = true;
            active.status = "cancelled";
          }
        }
        this.toolBridge.cancelAll();
        await this.sendMessage({
          type: "generation_stopped",
          message: "Generation stopped by user",
          job_id: jobId ?? null,
          thread_id: threadId ?? null,
        });
        return { message: "Stop command processed", job_id: jobId ?? null, thread_id: threadId ?? null };
      }
      default:
        return { error: "Unknown command" };
    }
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      void this.sendMessage({ type: "ping", ts: Date.now() / 1000 });
    }, 25_000);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private startStatsBroadcast(): void {
    this.stopStatsBroadcast();
    this.statsTimer = setInterval(() => {
      void this.sendMessage({ type: "system_stats", stats: this.getSystemStats() });
    }, 1_000);
  }

  private stopStatsBroadcast(): void {
    if (this.statsTimer) {
      clearInterval(this.statsTimer);
      this.statsTimer = null;
    }
  }

  private registerObserver(): void {
    if (this.observerRegistered) return;
    ModelObserver.subscribe(this.onModelChange);
    this.observerRegistered = true;
  }

  private unregisterObserver(): void {
    if (!this.observerRegistered) return;
    ModelObserver.unsubscribe(this.onModelChange);
    this.observerRegistered = false;
  }

  private onModelChange = (instance: DBModel, event: ModelChangeEvent): void => {
    if (!this.websocket) return;
    void this.sendMessage({
      type: "resource_change",
      event,
      resource_type: instance.constructor.name.toLowerCase(),
      resource: {
        id: instance.partitionValue(),
        etag: instance.getEtag(),
      },
    });
  };

  async run(websocket: WebSocketConnection): Promise<void> {
    await this.connect(websocket, this.userId ?? undefined, this.authToken ?? undefined);
    try {
      await this.receiveMessages();
    } finally {
      await this.disconnect();
    }
  }

  async receiveMessages(): Promise<void> {
    while (true) {
      const data = await this.receiveMessage();
      if (data === null) break;

      const msgType = typeof data.type === "string" ? data.type : null;
      if (msgType === "client_tools_manifest") {
        const tools = Array.isArray(data.tools) ? data.tools : [];
        this.clientToolsManifest = {};
        for (const tool of tools) {
          if (tool && typeof tool === "object" && typeof (tool as Record<string, unknown>).name === "string") {
            const name = (tool as Record<string, unknown>).name as string;
            this.clientToolsManifest[name] = tool as Record<string, unknown>;
          }
        }
        continue;
      }

      if (msgType === "tool_result") {
        const toolCallId = typeof data.tool_call_id === "string" ? data.tool_call_id : null;
        if (toolCallId) {
          this.toolBridge.resolveResult(toolCallId, data);
        }
        continue;
      }

      if (msgType === "ping") {
        await this.sendMessage({ type: "pong", ts: Date.now() / 1000 });
        continue;
      }

      if (typeof data.command === "string") {
        try {
          const command = data as unknown as WebSocketCommandEnvelope;
          const response = await this.handleCommand(command);
          await this.sendMessage(response);
        } catch (err) {
          await this.sendMessage({ error: "invalid_command", details: err instanceof Error ? err.message : String(err) });
        }
        continue;
      }

      await this.sendMessage({
        error: "invalid_message",
        message: "All messages must include a 'command' field. Use 'chat_message' command for chat.",
      });
    }
  }
}
