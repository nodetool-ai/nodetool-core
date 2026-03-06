import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import {
  Workflow,
  Job,
  Message,
  Thread,
  Asset,
  Secret,
  MemoryAdapterFactory,
  getGlobalAdapterResolver,
  setGlobalAdapterResolver,
} from "@nodetool/models";
import { loadPythonPackageMetadata, type NodeMetadata } from "@nodetool/node-sdk";
import { handleModelsApiRequest } from "./models-api.js";
import { handleOpenAIRequest, type OpenAIApiOptions } from "./openai-api.js";
import { handleOAuthRequest } from "./oauth-api.js";
import { handleSkillsRequest, handleFontsRequest } from "./skills-api.js";
import { handleCostRequest } from "./cost-api.js";
import { handleWorkspaceRequest } from "./workspace-api.js";

type JsonObject = Record<string, unknown>;

export interface HttpApiOptions {
  metadataRoots?: string[];
  metadataMaxDepth?: number;
  userIdHeader?: string;
  baseUrl?: string;
  openai?: OpenAIApiOptions;
}

export interface WorkflowRequestBody {
  name: string;
  tool_name?: string | null;
  package_name?: string | null;
  path?: string | null;
  tags?: string[] | null;
  description?: string | null;
  thumbnail?: string | null;
  thumbnail_url?: string | null;
  access: string;
  graph?: {
    nodes: Record<string, unknown>[];
    edges: Record<string, unknown>[];
  } | null;
  settings?: Record<string, unknown> | null;
  run_mode?: string | null;
  workspace_id?: string | null;
  html_app?: string | null;
}

const defaultMemoryFactory = new MemoryAdapterFactory();
let workflowTableInitialized = false;
let messageTableInitialized = false;
let threadTableInitialized = false;
let jobTableInitialized = false;
let assetTableInitialized = false;
let secretTableInitialized = false;

function ensureAdapterResolver(): void {
  if (!getGlobalAdapterResolver()) {
    setGlobalAdapterResolver((schema) => defaultMemoryFactory.getAdapter(schema));
  }
}

async function ensureWorkflowTable(): Promise<void> {
  if (workflowTableInitialized) return;
  ensureAdapterResolver();
  await Workflow.createTable();
  workflowTableInitialized = true;
}

async function ensureMessageTable(): Promise<void> {
  if (messageTableInitialized) return;
  ensureAdapterResolver();
  await Message.createTable();
  messageTableInitialized = true;
}

async function ensureThreadTable(): Promise<void> {
  if (threadTableInitialized) return;
  ensureAdapterResolver();
  await Thread.createTable();
  threadTableInitialized = true;
}

async function ensureJobTable(): Promise<void> {
  if (jobTableInitialized) return;
  ensureAdapterResolver();
  await Job.createTable();
  jobTableInitialized = true;
}

async function ensureSecretTable(): Promise<void> {
  if (secretTableInitialized) return;
  ensureAdapterResolver();
  await Secret.createTable();
  secretTableInitialized = true;
}

async function ensureAssetTable(): Promise<void> {
  if (assetTableInitialized) return;
  ensureAdapterResolver();
  await Asset.createTable();
  assetTableInitialized = true;
}

function normalizePath(pathname: string): string {
  if (pathname.length > 1 && pathname.endsWith("/")) {
    return pathname.slice(0, -1);
  }
  return pathname;
}

function jsonResponse(data: unknown, init?: ResponseInit): Response {
  return new Response(JSON.stringify(data), {
    status: init?.status ?? 200,
    headers: {
      "content-type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
}

function errorResponse(status: number, detail: string): Response {
  return jsonResponse({ detail }, { status });
}

function getUserId(request: Request, headerName: string): string {
  return request.headers.get(headerName) ?? request.headers.get("x-user-id") ?? "1";
}

async function parseJsonBody<T>(request: Request): Promise<T | null> {
  const contentType = request.headers.get("content-type") ?? "";
  if (!contentType.toLowerCase().includes("application/json")) {
    return null;
  }
  try {
    return (await request.json()) as T;
  } catch {
    return null;
  }
}

function toWorkflowResponse(workflow: Workflow): JsonObject {
  return {
    id: workflow.id,
    access: workflow.access,
    created_at: workflow.created_at,
    updated_at: workflow.updated_at,
    name: workflow.name,
    tool_name: workflow.tool_name,
    description: workflow.description,
    tags: workflow.tags,
    thumbnail: workflow.thumbnail,
    thumbnail_url: workflow.thumbnail_url,
    graph: workflow.graph,
    input_schema: null,
    output_schema: null,
    settings: workflow.settings,
    package_name: workflow.package_name,
    path: workflow.path,
    run_mode: workflow.run_mode,
    workspace_id: workflow.workspace_id,
    required_providers: null,
    required_models: null,
    html_app: workflow.html_app,
    etag: workflow.getEtag(),
  };
}

async function handleNodeMetadata(request: Request, options: HttpApiOptions): Promise<Response> {
  if (request.method !== "GET") {
    return errorResponse(405, "Method not allowed");
  }
  const loaded = loadPythonPackageMetadata({
    roots: options.metadataRoots,
    maxDepth: options.metadataMaxDepth,
  });
  const nodes: NodeMetadata[] = [...loaded.nodesByType.values()].sort((a, b) =>
    a.node_type.localeCompare(b.node_type)
  );
  return jsonResponse(nodes);
}

function parseLimit(url: URL, defaultLimit = 100): number {
  const raw = url.searchParams.get("limit");
  if (!raw) return defaultLimit;
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) return defaultLimit;
  return Math.min(parsed, 500);
}

async function createWorkflow(body: WorkflowRequestBody, userId: string): Promise<Workflow> {
  if (!body || typeof body.name !== "string" || typeof body.access !== "string") {
    throw new Error("Invalid workflow");
  }
  if (!body.graph || !Array.isArray(body.graph.nodes) || !Array.isArray(body.graph.edges)) {
    throw new Error("Invalid workflow");
  }

  return (await Workflow.create({
    user_id: userId,
    name: body.name,
    tool_name: body.tool_name ?? null,
    package_name: body.package_name ?? null,
    path: body.path ?? null,
    tags: body.tags ?? [],
    description: body.description ?? "",
    thumbnail: body.thumbnail ?? null,
    thumbnail_url: body.thumbnail_url ?? null,
    access: body.access === "public" ? "public" : "private",
    graph: body.graph,
    settings: body.settings ?? null,
    run_mode: body.run_mode ?? "workflow",
    workspace_id: body.workspace_id ?? null,
    html_app: body.html_app ?? null,
  })) as Workflow;
}

async function updateWorkflow(
  id: string,
  body: WorkflowRequestBody,
  userId: string
): Promise<Workflow> {
  if (!body || typeof body.name !== "string" || typeof body.access !== "string") {
    throw new Error("Invalid workflow");
  }
  if (!body.graph || !Array.isArray(body.graph.nodes) || !Array.isArray(body.graph.edges)) {
    throw new Error("Invalid workflow");
  }

  const existing = (await Workflow.get(id)) as Workflow | null;
  if (!existing) {
    throw new Error("Workflow not found");
  }

  if (existing.user_id !== userId) {
    throw new Error("Workflow not found");
  }

  existing.name = body.name;
  existing.tool_name = body.tool_name ?? null;
  existing.package_name = body.package_name ?? null;
  existing.path = body.path ?? null;
  existing.tags = body.tags ?? [];
  existing.description = body.description ?? "";
  existing.thumbnail = body.thumbnail ?? null;
  existing.thumbnail_url = body.thumbnail_url ?? null;
  existing.access = body.access === "public" ? "public" : "private";
  existing.graph = body.graph;
  existing.settings = body.settings ?? null;
  existing.run_mode = body.run_mode ?? existing.run_mode ?? "workflow";
  existing.workspace_id = body.workspace_id ?? null;
  existing.html_app = body.html_app ?? null;
  await existing.save();
  return existing;
}

async function handleWorkflowsRoot(request: Request, options: HttpApiOptions): Promise<Response> {
  const userId = getUserId(request, options.userIdHeader ?? "x-user-id");
  const url = new URL(request.url);

  if (request.method === "GET") {
    await ensureWorkflowTable();
    const limit = parseLimit(url, 100);
    const runMode = url.searchParams.get("run_mode") ?? undefined;
    const [workflows] = await Workflow.paginate(userId, { limit, runMode });
    return jsonResponse({
      workflows: workflows.map((w) => toWorkflowResponse(w)),
      next: null,
    });
  }

  if (request.method === "POST") {
    await ensureWorkflowTable();
    const body = await parseJsonBody<WorkflowRequestBody>(request);
    if (!body) return errorResponse(400, "Invalid JSON body");
    try {
      const workflow = await createWorkflow(body, userId);
      return jsonResponse(toWorkflowResponse(workflow));
    } catch (error) {
      const message = error instanceof Error ? error.message : "Invalid workflow";
      return errorResponse(400, message);
    }
  }

  return errorResponse(405, "Method not allowed");
}

async function handlePublicWorkflows(request: Request): Promise<Response> {
  if (request.method !== "GET") {
    return errorResponse(405, "Method not allowed");
  }
  await ensureWorkflowTable();
  const url = new URL(request.url);
  const limit = parseLimit(url, 100);
  const [workflows] = await Workflow.paginatePublic({ limit });
  return jsonResponse({
    workflows: workflows.map((w) => toWorkflowResponse(w)),
    next: null,
  });
}

async function handlePublicWorkflowById(request: Request, workflowId: string): Promise<Response> {
  if (request.method !== "GET") {
    return errorResponse(405, "Method not allowed");
  }
  await ensureWorkflowTable();
  const workflow = (await Workflow.get(workflowId)) as Workflow | null;
  if (!workflow || workflow.access !== "public") {
    return errorResponse(404, "Workflow not found");
  }
  return jsonResponse(toWorkflowResponse(workflow));
}

async function handleWorkflowById(
  request: Request,
  workflowId: string,
  options: HttpApiOptions
): Promise<Response> {
  const userId = getUserId(request, options.userIdHeader ?? "x-user-id");
  await ensureWorkflowTable();

  if (request.method === "GET") {
  const workflow = (await Workflow.get(workflowId)) as Workflow | null;
    if (!workflow) return errorResponse(404, "Workflow not found");
    if (workflow.access !== "public" && workflow.user_id !== userId) {
      return errorResponse(404, "Workflow not found");
    }
    return jsonResponse(toWorkflowResponse(workflow));
  }

  if (request.method === "PUT") {
    const body = await parseJsonBody<WorkflowRequestBody>(request);
    if (!body) return errorResponse(400, "Invalid JSON body");
    try {
      const workflow = await updateWorkflow(workflowId, body, userId);
      return jsonResponse(toWorkflowResponse(workflow));
    } catch (error) {
      const message = error instanceof Error ? error.message : "Invalid workflow";
      if (message === "Workflow not found") return errorResponse(404, message);
      return errorResponse(400, message);
    }
  }

  if (request.method === "DELETE") {
    const workflow = (await Workflow.get(workflowId)) as Workflow | null;
    if (!workflow) return errorResponse(404, "Workflow not found");
    if (workflow.user_id !== userId) return errorResponse(404, "Workflow not found");
    await workflow.delete();
    return new Response(null, { status: 204 });
  }

  return errorResponse(405, "Method not allowed");
}

// ── Message types & helpers ────────────────────────────────────────

interface MessageCreateBody {
  thread_id?: string | null;
  role: string;
  name?: string | null;
  content: string;
  tool_call_id?: string | null;
  tool_calls?: unknown[] | null;
}

function toMessageResponse(msg: Message): JsonObject {
  return {
    id: msg.id,
    user_id: msg.user_id,
    thread_id: msg.thread_id,
    role: msg.role,
    content: msg.content,
    tool_calls: msg.tool_calls,
    created_at: msg.created_at,
    updated_at: msg.updated_at,
  };
}

async function handleMessagesRoot(request: Request, options: HttpApiOptions): Promise<Response> {
  const userId = getUserId(request, options.userIdHeader ?? "x-user-id");

  if (request.method === "POST") {
    await ensureThreadTable();
    await ensureMessageTable();
    const body = await parseJsonBody<MessageCreateBody>(request);
    if (!body || typeof body.role !== "string" || typeof body.content !== "string") {
      return errorResponse(400, "Invalid JSON body");
    }
    let threadId = body.thread_id;
    if (!threadId) {
      const thread = (await Thread.create({
        user_id: userId,
        title: "New Thread",
      })) as Thread;
      threadId = thread.id;
    }
    const msg = (await Message.create({
      user_id: userId,
      thread_id: threadId,
      role: body.role,
      content: body.content,
      tool_calls: body.tool_calls ?? null,
    })) as Message;
    return jsonResponse(toMessageResponse(msg));
  }

  if (request.method === "GET") {
    await ensureMessageTable();
    const url = new URL(request.url);
    const threadId = url.searchParams.get("thread_id");
    if (!threadId) {
      return errorResponse(400, "thread_id is required");
    }
    const limit = parseLimit(url, 100);
    const [messages, cursor] = await Message.paginate(threadId, { limit });
    // Verify user ownership
    for (const msg of messages) {
      if (msg.user_id !== userId) {
        return errorResponse(404, "Message not found");
      }
    }
    return jsonResponse({
      messages: messages.map((m) => toMessageResponse(m)),
      next: cursor || null,
    });
  }

  return errorResponse(405, "Method not allowed");
}

async function handleMessageById(
  request: Request,
  messageId: string,
  options: HttpApiOptions
): Promise<Response> {
  if (request.method !== "GET") {
    return errorResponse(405, "Method not allowed");
  }
  const userId = getUserId(request, options.userIdHeader ?? "x-user-id");
  await ensureMessageTable();
  const msg = (await Message.get(messageId)) as Message | null;
  if (!msg || msg.user_id !== userId) {
    return errorResponse(404, "Message not found");
  }
  return jsonResponse(toMessageResponse(msg));
}

// ── Thread types & helpers ────────────────────────────────────────

interface ThreadCreateBody {
  title?: string | null;
}

interface ThreadUpdateBody {
  title: string;
}

function toThreadResponse(thread: Thread): JsonObject {
  return {
    id: thread.id,
    user_id: thread.user_id,
    title: thread.title,
    created_at: thread.created_at,
    updated_at: thread.updated_at,
  };
}

async function handleThreadsRoot(request: Request, options: HttpApiOptions): Promise<Response> {
  const userId = getUserId(request, options.userIdHeader ?? "x-user-id");

  if (request.method === "POST") {
    await ensureThreadTable();
    const body = await parseJsonBody<ThreadCreateBody>(request);
    const title = body?.title ?? "New Thread";
    const thread = (await Thread.create({
      user_id: userId,
      title,
    })) as Thread;
    return jsonResponse(toThreadResponse(thread));
  }

  if (request.method === "GET") {
    await ensureThreadTable();
    const url = new URL(request.url);
    const limit = parseLimit(url, 10);
    const [threads, cursor] = await Thread.paginate(userId, { limit });
    return jsonResponse({
      threads: threads.map((t) => toThreadResponse(t)),
      next: cursor || null,
    });
  }

  return errorResponse(405, "Method not allowed");
}

async function handleThreadById(
  request: Request,
  threadId: string,
  options: HttpApiOptions
): Promise<Response> {
  const userId = getUserId(request, options.userIdHeader ?? "x-user-id");
  await ensureThreadTable();

  if (request.method === "GET") {
    const thread = await Thread.find(userId, threadId);
    if (!thread) return errorResponse(404, "Thread not found");
    return jsonResponse(toThreadResponse(thread));
  }

  if (request.method === "PUT") {
    const body = await parseJsonBody<ThreadUpdateBody>(request);
    if (!body || typeof body.title !== "string") {
      return errorResponse(400, "Invalid JSON body");
    }
    const thread = await Thread.find(userId, threadId);
    if (!thread) return errorResponse(404, "Thread not found");
    thread.title = body.title;
    await thread.save();
    return jsonResponse(toThreadResponse(thread));
  }

  if (request.method === "DELETE") {
    const thread = await Thread.find(userId, threadId);
    if (!thread) return errorResponse(404, "Thread not found");
    // Delete all messages in the thread
    await ensureMessageTable();
    while (true) {
      const [messages] = await Message.paginate(threadId, { limit: 100 });
      if (!messages.length) break;
      for (const msg of messages) {
        if (msg.user_id === userId) {
          await msg.delete();
        }
      }
      if (messages.length < 100) break;
    }
    await thread.delete();
    return new Response(null, { status: 204 });
  }

  return errorResponse(405, "Method not allowed");
}

// ── Job types & helpers ───────────────────────────────────────────

function toJobResponse(job: Job): JsonObject {
  return {
    id: job.id,
    user_id: job.user_id,
    job_type: "workflow",
    status: job.status,
    workflow_id: job.workflow_id,
    started_at: job.started_at ?? null,
    finished_at: job.finished_at ?? null,
    error: job.error ?? null,
    cost: null,
  };
}

async function handleJobsRoot(request: Request, options: HttpApiOptions): Promise<Response> {
  if (request.method !== "GET") {
    return errorResponse(405, "Method not allowed");
  }

  const userId = getUserId(request, options.userIdHeader ?? "x-user-id");
  await ensureJobTable();

  const url = new URL(request.url);
  const limit = parseLimit(url, 100);
  const workflowId = url.searchParams.get("workflow_id") ?? undefined;

  const [jobs, nextStartKey] = await Job.paginate(userId, { limit, workflowId });

  return jsonResponse({
    jobs: jobs.map((j) => toJobResponse(j)),
    next_start_key: nextStartKey || null,
  });
}

async function handleJobById(
  request: Request,
  jobId: string,
  options: HttpApiOptions
): Promise<Response> {
  if (request.method !== "GET") {
    return errorResponse(405, "Method not allowed");
  }

  const userId = getUserId(request, options.userIdHeader ?? "x-user-id");
  await ensureJobTable();

  const job = (await Job.get(jobId)) as Job | null;
  if (!job || job.user_id !== userId) {
    return errorResponse(404, "Job not found");
  }

  return jsonResponse(toJobResponse(job));
}

async function handleJobCancel(
  request: Request,
  jobId: string,
  options: HttpApiOptions
): Promise<Response> {
  if (request.method !== "POST") {
    return errorResponse(405, "Method not allowed");
  }

  const userId = getUserId(request, options.userIdHeader ?? "x-user-id");
  await ensureJobTable();

  const job = (await Job.get(jobId)) as Job | null;
  if (!job || job.user_id !== userId) {
    return errorResponse(404, "Job not found");
  }

  job.markCancelled();
  await job.save();

  return jsonResponse(toJobResponse(job));
}

// ── Secrets types & helpers ────────────────────────────────────────

interface SecretUpdateBody {
  value: string;
  description?: string;
}

function toSecretResponse(secret: Secret): JsonObject {
  return {
    ...secret.toSafeObject(),
    is_configured: true,
  };
}

async function handleSecretsRoot(request: Request, options: HttpApiOptions): Promise<Response> {
  if (request.method !== "GET") {
    return errorResponse(405, "Method not allowed");
  }
  const userId = getUserId(request, options.userIdHeader ?? "x-user-id");
  await ensureSecretTable();

  const url = new URL(request.url);
  const limit = parseLimit(url, 100);
  const [secrets] = await Secret.listForUser(userId, limit);

  return jsonResponse({
    secrets: secrets.map((s) => toSecretResponse(s)),
    next_key: null,
  });
}

async function handleSecretByKey(
  request: Request,
  key: string,
  options: HttpApiOptions
): Promise<Response> {
  const userId = getUserId(request, options.userIdHeader ?? "x-user-id");
  await ensureSecretTable();

  if (request.method === "GET") {
    const secret = await Secret.find(userId, key);
    if (!secret) return errorResponse(404, "Secret not found");

    const response = toSecretResponse(secret) as Record<string, unknown>;
    const url = new URL(request.url);
    if (url.searchParams.get("decrypt") === "true") {
      try {
        response.value = await secret.getDecryptedValue();
      } catch (err) {
        const detail = err instanceof Error ? err.message : "Failed to decrypt secret";
        return errorResponse(500, detail);
      }
    }
    return jsonResponse(response);
  }

  if (request.method === "PUT") {
    const body = await parseJsonBody<SecretUpdateBody>(request);
    if (!body || typeof body.value !== "string") {
      return errorResponse(400, "Invalid JSON body");
    }
    try {
      const secret = await Secret.upsert({
        userId,
        key,
        value: body.value,
        description: body.description,
      });
      return jsonResponse(toSecretResponse(secret));
    } catch (err) {
      const detail = err instanceof Error ? err.message : "Failed to update secret";
      return errorResponse(500, detail);
    }
  }

  if (request.method === "DELETE") {
    const deleted = await Secret.deleteSecret(userId, key);
    if (!deleted) return errorResponse(404, "Secret not found");
    return jsonResponse({ message: "Secret deleted successfully" });
  }

  return errorResponse(405, "Method not allowed");
}

// ── Asset types & helpers ──────────────────────────────────────────

interface AssetCreateBody {
  name: string;
  content_type: string;
  parent_id: string;
  workflow_id?: string | null;
  node_id?: string | null;
  job_id?: string | null;
  metadata?: Record<string, unknown> | null;
  size?: number | null;
}

interface AssetUpdateBody {
  name?: string;
  content_type?: string;
  parent_id?: string;
  metadata?: Record<string, unknown>;
  size?: number;
}

function toAssetResponse(asset: Asset): JsonObject {
  return {
    id: asset.id,
    user_id: asset.user_id,
    workflow_id: asset.workflow_id ?? null,
    parent_id: asset.parent_id ?? null,
    name: asset.name,
    content_type: asset.content_type,
    size: asset.size ?? null,
    metadata: asset.metadata ?? null,
    created_at: asset.created_at,
    get_url: null,
    thumb_url: null,
    duration: asset.duration ?? null,
    node_id: asset.node_id ?? null,
    job_id: asset.job_id ?? null,
  };
}

async function deleteFolderRecursive(userId: string, folderId: string): Promise<string[]> {
  const deletedIds: string[] = [];
  const children = await Asset.getChildren(userId, folderId, 10000);
  for (const child of children) {
    if (child.content_type === "folder") {
      const subDeleted = await deleteFolderRecursive(userId, child.id);
      deletedIds.push(...subDeleted);
    } else {
      await child.delete();
      deletedIds.push(child.id);
    }
  }
  const folder = await Asset.find(userId, folderId);
  if (folder) {
    await folder.delete();
    deletedIds.push(folderId);
  }
  return deletedIds;
}

async function handleAssetsRoot(request: Request, options: HttpApiOptions): Promise<Response> {
  const userId = getUserId(request, options.userIdHeader ?? "x-user-id");
  await ensureAssetTable();

  if (request.method === "GET") {
    const url = new URL(request.url);
    const parentId = url.searchParams.get("parent_id") ?? undefined;
    const contentType = url.searchParams.get("content_type") ?? undefined;
    const workflowId = url.searchParams.get("workflow_id") ?? undefined;
    const nodeId = url.searchParams.get("node_id") ?? undefined;
    const jobId = url.searchParams.get("job_id") ?? undefined;
    const pageSizeRaw = url.searchParams.get("page_size");
    const pageSize = pageSizeRaw
      ? Math.min(Math.max(Number.parseInt(pageSizeRaw, 10) || 10000, 1), 10000)
      : 10000;

    // Default to home folder if no filters specified
    const effectiveParentId =
      parentId === undefined && !contentType && !workflowId && !nodeId && !jobId
        ? userId
        : parentId;

    const [assets, cursor] = await Asset.paginate(userId, {
      parentId: effectiveParentId,
      contentType,
      workflowId,
      nodeId,
      jobId,
      limit: pageSize,
    });

    return jsonResponse({
      assets: assets.map((a) => toAssetResponse(a)),
      next: cursor || null,
    });
  }

  if (request.method === "POST") {
    const body = await parseJsonBody<AssetCreateBody>(request);
    if (
      !body ||
      typeof body.name !== "string" ||
      typeof body.content_type !== "string" ||
      typeof body.parent_id !== "string"
    ) {
      return errorResponse(400, "Invalid JSON body: name, content_type, and parent_id are required");
    }

    const asset = (await Asset.create({
      user_id: userId,
      name: body.name,
      content_type: body.content_type,
      parent_id: body.parent_id,
      workflow_id: body.workflow_id ?? null,
      node_id: body.node_id ?? null,
      job_id: body.job_id ?? null,
      metadata: body.metadata ?? null,
      size: body.size ?? null,
    })) as Asset;

    return jsonResponse(toAssetResponse(asset));
  }

  return errorResponse(405, "Method not allowed");
}

async function handleAssetById(
  request: Request,
  assetId: string,
  options: HttpApiOptions
): Promise<Response> {
  const userId = getUserId(request, options.userIdHeader ?? "x-user-id");
  await ensureAssetTable();

  if (request.method === "GET") {
    // Special case: home folder
    if (assetId === userId) {
      return jsonResponse({
        id: userId,
        user_id: userId,
        workflow_id: null,
        parent_id: "",
        name: "Home",
        content_type: "folder",
        size: null,
        metadata: null,
        created_at: "",
        get_url: null,
        thumb_url: null,
        duration: null,
        node_id: null,
        job_id: null,
      });
    }

    const asset = await Asset.find(userId, assetId);
    if (!asset) return errorResponse(404, "Asset not found");
    return jsonResponse(toAssetResponse(asset));
  }

  if (request.method === "PUT") {
    const asset = await Asset.find(userId, assetId);
    if (!asset) return errorResponse(404, "Asset not found");

    const body = await parseJsonBody<AssetUpdateBody>(request);
    if (!body) return errorResponse(400, "Invalid JSON body");

    if (body.name !== undefined) asset.name = body.name;
    if (body.content_type !== undefined) asset.content_type = body.content_type;
    if (body.parent_id !== undefined) asset.parent_id = body.parent_id;
    if (body.metadata !== undefined) asset.metadata = body.metadata;
    if (body.size !== undefined) asset.size = body.size;
    await asset.save();
    return jsonResponse(toAssetResponse(asset));
  }

  if (request.method === "DELETE") {
    const asset = await Asset.find(userId, assetId);
    if (!asset) return errorResponse(404, "Asset not found");

    let deletedAssetIds: string[];
    if (asset.content_type === "folder") {
      deletedAssetIds = await deleteFolderRecursive(userId, assetId);
    } else {
      await asset.delete();
      deletedAssetIds = [assetId];
    }
    return jsonResponse({ deleted_asset_ids: deletedAssetIds });
  }

  return errorResponse(405, "Method not allowed");
}

export async function handleApiRequest(
  request: Request,
  options: HttpApiOptions = {}
): Promise<Response> {
  const url = new URL(request.url);
  const pathname = normalizePath(url.pathname);

  if (pathname === "/health") {
    return new Response("OK", { status: 200, headers: { "content-type": "text/plain" } });
  }

  if (pathname === "/ping") {
    return jsonResponse({ status: "healthy", timestamp: new Date().toISOString() });
  }

  if (pathname.startsWith("/v1/")) {
    const userId = getUserId(request, options.userIdHeader ?? "x-user-id");
    const response = await handleOpenAIRequest(request, pathname, userId, options.openai);
    if (response) return response;
  }

  if (pathname.startsWith("/api/oauth/")) {
    const response = await handleOAuthRequest(request, pathname, () => getUserId(request, options.userIdHeader ?? "x-user-id"));
    if (response) return response;
  }

  if (pathname === "/api/models" || pathname.startsWith("/api/models/")) {
    const response = await handleModelsApiRequest(request);
    if (response) return response;
  }

  if (pathname === "/api/nodes/metadata" || pathname === "/api/node/metadata") {
    return handleNodeMetadata(request, options);
  }

  if (pathname === "/api/settings/secrets") {
    return handleSecretsRoot(request, options);
  }

  if (pathname.startsWith("/api/settings/secrets/")) {
    const secretKey = decodeURIComponent(pathname.slice("/api/settings/secrets/".length));
    if (!secretKey) return errorResponse(404, "Not found");
    return handleSecretByKey(request, secretKey, options);
  }

  if (pathname === "/api/assets") {
    return handleAssetsRoot(request, options);
  }

  if (pathname.startsWith("/api/assets/")) {
    const assetId = decodeURIComponent(pathname.slice("/api/assets/".length));
    if (!assetId) return errorResponse(404, "Not found");
    return handleAssetById(request, assetId, options);
  }

  if (pathname === "/api/jobs") {
    return handleJobsRoot(request, options);
  }

  if (pathname.match(/^\/api\/jobs\/[^/]+\/cancel$/)) {
    const jobId = decodeURIComponent(pathname.slice("/api/jobs/".length, pathname.length - "/cancel".length));
    if (!jobId) return errorResponse(404, "Not found");
    return handleJobCancel(request, jobId, options);
  }

  if (pathname.startsWith("/api/jobs/")) {
    const jobId = decodeURIComponent(pathname.slice("/api/jobs/".length));
    if (!jobId) return errorResponse(404, "Not found");
    return handleJobById(request, jobId, options);
  }

  if (pathname === "/api/messages") {
    return handleMessagesRoot(request, options);
  }

  if (pathname.startsWith("/api/messages/")) {
    const messageId = decodeURIComponent(pathname.slice("/api/messages/".length));
    if (!messageId) return errorResponse(404, "Not found");
    return handleMessageById(request, messageId, options);
  }

  if (pathname === "/api/threads") {
    return handleThreadsRoot(request, options);
  }

  if (pathname.startsWith("/api/threads/")) {
    const threadId = decodeURIComponent(pathname.slice("/api/threads/".length));
    if (!threadId) return errorResponse(404, "Not found");
    return handleThreadById(request, threadId, options);
  }

  if (pathname === "/api/workflows") {
    return handleWorkflowsRoot(request, options);
  }

  if (pathname === "/api/workflows/public") {
    return handlePublicWorkflows(request);
  }

  if (pathname.startsWith("/api/workflows/public/")) {
    const workflowId = decodeURIComponent(pathname.slice("/api/workflows/public/".length));
    if (!workflowId) return errorResponse(404, "Not found");
    return handlePublicWorkflowById(request, workflowId);
  }

  if (pathname.startsWith("/api/workflows/")) {
    const workflowId = decodeURIComponent(pathname.slice("/api/workflows/".length));
    if (!workflowId) return errorResponse(404, "Not found");
    return handleWorkflowById(request, workflowId, options);
  }

  if (pathname === "/api/skills" || pathname.startsWith("/api/skills/")) {
    return handleSkillsRequest(request);
  }

  if (pathname === "/api/fonts" || pathname.startsWith("/api/fonts/")) {
    return handleFontsRequest(request);
  }

  if (pathname === "/api/costs" || pathname.startsWith("/api/costs/")) {
    const response = await handleCostRequest(request, options);
    if (response) return response;
  }

  if (pathname === "/api/workspaces" || pathname.startsWith("/api/workspaces/")) {
    const response = await handleWorkspaceRequest(request, options);
    if (response) return response;
  }

  if (pathname === "/api/users" || pathname.startsWith("/api/users/")) {
    return errorResponse(501, "User management not available");
  }

  return errorResponse(404, "Not found");
}

async function readNodeRequestBody(request: IncomingMessage): Promise<Uint8Array> {
  const chunks: Uint8Array[] = [];
  for await (const chunk of request) {
    if (chunk instanceof Uint8Array) {
      chunks.push(chunk);
    } else {
      chunks.push(Buffer.from(String(chunk)));
    }
  }
  const size = chunks.reduce((sum, c) => sum + c.byteLength, 0);
  const merged = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return merged;
}

export async function handleNodeHttpRequest(
  req: IncomingMessage,
  res: ServerResponse,
  options: HttpApiOptions = {}
): Promise<void> {
  const method = req.method ?? "GET";
  const baseUrl = options.baseUrl ?? "http://127.0.0.1:7777";
  const url = new URL(req.url ?? "/", baseUrl);
  const headers = new Headers();
  for (const [key, value] of Object.entries(req.headers)) {
    if (Array.isArray(value)) {
      for (const v of value) headers.append(key, v);
    } else if (value !== undefined) {
      headers.set(key, value);
    }
  }

  const hasBody = method !== "GET" && method !== "HEAD";
  const rawBody = hasBody ? await readNodeRequestBody(req) : undefined;
  const request = new Request(url.toString(), {
    method,
    headers,
    body: rawBody && rawBody.byteLength > 0 ? (rawBody as unknown as BodyInit) : undefined,
  });

  const response = await handleApiRequest(request, options);

  res.statusCode = response.status;
  response.headers.forEach((value, key) => {
    res.setHeader(key, value);
  });

  if (!response.body) {
    res.end();
    return;
  }

  const bytes = new Uint8Array(await response.arrayBuffer());
  res.end(Buffer.from(bytes));
}

export function createHttpApiServer(options: HttpApiOptions = {}): Server {
  return createServer((req, res) => {
    void handleNodeHttpRequest(req, res, options).catch((error) => {
      // eslint-disable-next-line no-console
      console.error("[createHttpApiServer] request failed", error);
      res.statusCode = 500;
      res.setHeader("content-type", "application/json");
      const detail = error instanceof Error ? error.message : String(error);
      res.end(JSON.stringify({ detail }));
    });
  });
}
