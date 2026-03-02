import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import {
  Workflow,
  MemoryAdapterFactory,
  getGlobalAdapterResolver,
  setGlobalAdapterResolver,
} from "@nodetool/models";
import { loadPythonPackageMetadata, type NodeMetadata } from "@nodetool/node-sdk";

type JsonObject = Record<string, unknown>;

export interface HttpApiOptions {
  metadataRoots?: string[];
  metadataMaxDepth?: number;
  userIdHeader?: string;
  baseUrl?: string;
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
      return (await Workflow.create({
      id,
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

export async function handleApiRequest(
  request: Request,
  options: HttpApiOptions = {}
): Promise<Response> {
  const url = new URL(request.url);
  const pathname = normalizePath(url.pathname);

  if (pathname === "/api/nodes/metadata" || pathname === "/api/node/metadata") {
    return handleNodeMetadata(request, options);
  }

  if (pathname === "/api/workflows") {
    return handleWorkflowsRoot(request, options);
  }

  if (pathname === "/api/workflows/public") {
    return handlePublicWorkflows(request);
  }

  if (pathname.startsWith("/api/workflows/")) {
    const workflowId = decodeURIComponent(pathname.slice("/api/workflows/".length));
    if (!workflowId) return errorResponse(404, "Not found");
    return handleWorkflowById(request, workflowId, options);
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
    body: rawBody && rawBody.byteLength > 0 ? rawBody : undefined,
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
      res.statusCode = 500;
      res.setHeader("content-type", "application/json");
      const detail = error instanceof Error ? error.message : String(error);
      res.end(JSON.stringify({ detail }));
    });
  });
}
