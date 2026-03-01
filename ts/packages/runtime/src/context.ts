/**
 * ProcessingContext – runtime context for node execution.
 *
 * Port of src/nodetool/workflows/processing_context.py.
 *
 * Provides:
 *   - Message queue for emitting ProcessingMessages.
 *   - Cache get/set interface.
 *   - Output normalization (sanitize memory URIs, etc.).
 *   - Asset handling with pluggable storage adapters.
 */

import type { ProcessingMessage } from "@nodetool/protocol";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, normalize, relative, resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

// ---------------------------------------------------------------------------
// Cache interface
// ---------------------------------------------------------------------------

export interface CacheAdapter {
  get(key: string): Promise<unknown | undefined>;
  set(key: string, value: unknown, ttlSeconds?: number): Promise<void>;
  has(key: string): Promise<boolean>;
  delete(key: string): Promise<void>;
}

/**
 * In-memory cache adapter (default for tests and single-process execution).
 */
export class MemoryCache implements CacheAdapter {
  private _store = new Map<string, { value: unknown; expires: number | null }>();

  async get(key: string): Promise<unknown | undefined> {
    const entry = this._store.get(key);
    if (!entry) return undefined;
    if (entry.expires !== null && Date.now() > entry.expires) {
      this._store.delete(key);
      return undefined;
    }
    return entry.value;
  }

  async set(key: string, value: unknown, ttlSeconds?: number): Promise<void> {
    const expires = ttlSeconds ? Date.now() + ttlSeconds * 1000 : null;
    this._store.set(key, { value, expires });
  }

  async has(key: string): Promise<boolean> {
    return (await this.get(key)) !== undefined;
  }

  async delete(key: string): Promise<void> {
    this._store.delete(key);
  }
}

// ---------------------------------------------------------------------------
// Storage adapter interface
// ---------------------------------------------------------------------------

export interface StorageAdapter {
  /** Store an asset and return a URI. */
  store(key: string, data: Uint8Array, contentType?: string): Promise<string>;

  /** Retrieve an asset by URI. */
  retrieve(uri: string): Promise<Uint8Array | null>;

  /** Check if an asset exists. */
  exists(uri: string): Promise<boolean>;
}

function isWithinRoot(root: string, target: string): boolean {
  const rel = relative(root, target);
  return rel === "" || (!rel.startsWith("..") && !isAbsolute(rel));
}

function normalizeStorageKey(key: string): string {
  const cleaned = normalize(key.replaceAll("\\", "/")).replace(/^\/+/, "");
  if (!cleaned || cleaned === "." || cleaned.startsWith("..") || cleaned.includes(`..${sep}`)) {
    throw new Error(`Invalid storage key: ${key}`);
  }
  return cleaned;
}

/**
 * In-memory storage adapter useful for tests and single-process ephemeral runs.
 */
export class InMemoryStorageAdapter implements StorageAdapter {
  private _store = new Map<string, Uint8Array>();

  async store(key: string, data: Uint8Array, _contentType?: string): Promise<string> {
    const normalized = normalizeStorageKey(key);
    this._store.set(normalized, new Uint8Array(data));
    return `memory://${normalized}`;
  }

  async retrieve(uri: string): Promise<Uint8Array | null> {
    if (!uri.startsWith("memory://")) return null;
    const key = uri.slice("memory://".length);
    const value = this._store.get(key);
    return value ? new Uint8Array(value) : null;
  }

  async exists(uri: string): Promise<boolean> {
    if (!uri.startsWith("memory://")) return false;
    const key = uri.slice("memory://".length);
    return this._store.has(key);
  }
}

/**
 * File-system storage adapter rooted to a single base directory.
 */
export class FileStorageAdapter implements StorageAdapter {
  readonly rootDir: string;

  constructor(rootDir: string) {
    this.rootDir = resolve(rootDir);
  }

  private resolvePathFromKey(key: string): string {
    const normalized = normalizeStorageKey(key);
    const absolute = resolve(join(this.rootDir, normalized));
    if (!isWithinRoot(this.rootDir, absolute)) {
      throw new Error(`Storage key escapes root: ${key}`);
    }
    return absolute;
  }

  private resolvePathFromUri(uri: string): string | null {
    if (!uri.startsWith("file://")) return null;
    const absolute = resolve(fileURLToPath(uri));
    if (!isWithinRoot(this.rootDir, absolute)) {
      return null;
    }
    return absolute;
  }

  async store(key: string, data: Uint8Array, _contentType?: string): Promise<string> {
    const absolutePath = this.resolvePathFromKey(key);
    await mkdir(dirname(absolutePath), { recursive: true });
    await writeFile(absolutePath, data);
    return pathToFileURL(absolutePath).toString();
  }

  async retrieve(uri: string): Promise<Uint8Array | null> {
    const absolutePath = this.resolvePathFromUri(uri);
    if (!absolutePath) return null;
    try {
      return await readFile(absolutePath);
    } catch {
      return null;
    }
  }

  async exists(uri: string): Promise<boolean> {
    return (await this.retrieve(uri)) !== null;
  }
}

/**
 * Resolve paths relative to a configured workspace root.
 *
 * Supported path forms:
 * - /workspace/foo/bar.txt
 * - workspace/foo/bar.txt
 * - absolute paths (treated as workspace-relative)
 * - relative paths
 */
export function resolveWorkspacePath(workspaceDir: string | null | undefined, path: string): string {
  if (workspaceDir == null) {
    throw new Error(
      "No workspace is assigned. File operations require a user-defined workspace. Please configure a workspace before performing disk I/O operations."
    );
  }
  if (workspaceDir === "") {
    throw new Error("Workspace directory is required");
  }

  const workspaceRoot = resolve(workspaceDir);
  const normalizedPath = path.replaceAll("\\", "/");

  let relativePath: string;
  if (normalizedPath.startsWith("/workspace/")) {
    relativePath = normalizedPath.slice("/workspace/".length);
  } else if (normalizedPath.startsWith("workspace/")) {
    relativePath = normalizedPath.slice("workspace/".length);
  } else if (isAbsolute(normalizedPath) || /^[A-Za-z]:\//.test(normalizedPath)) {
    if (normalizedPath.startsWith("/")) {
      relativePath = normalizedPath.slice(1);
    } else {
      relativePath = normalizedPath.replace(/^[A-Za-z]:\//, "");
    }
  } else {
    relativePath = normalizedPath;
  }

  const absPath = resolve(join(workspaceRoot, relativePath));
  if (!isWithinRoot(workspaceRoot, absPath)) {
    throw new Error(`Resolved path '${absPath}' is outside the workspace directory.`);
  }
  return absPath;
}

// ---------------------------------------------------------------------------
// ProcessingContext
// ---------------------------------------------------------------------------

export class ProcessingContext {
  readonly jobId: string;
  readonly workflowId: string | null;
  readonly userId: string;
  readonly workspaceDir: string | null;

  /** Message queue: all emitted processing messages. */
  private _messages: ProcessingMessage[] = [];

  /** Optional message listener (for real-time streaming). */
  private _onMessage: ((msg: ProcessingMessage) => void) | null = null;

  /** Cache adapter. */
  readonly cache: CacheAdapter;

  /** Storage adapter (optional, for asset handling). */
  readonly storage: StorageAdapter | null;

  constructor(opts: {
    jobId: string;
    workflowId?: string | null;
    userId?: string;
    workspaceDir?: string | null;
    cache?: CacheAdapter;
    storage?: StorageAdapter | null;
    onMessage?: (msg: ProcessingMessage) => void;
  }) {
    this.jobId = opts.jobId;
    this.workflowId = opts.workflowId ?? null;
    this.userId = opts.userId ?? "default";
    this.workspaceDir = opts.workspaceDir ?? null;
    this.cache = opts.cache ?? new MemoryCache();
    this.storage = opts.storage ?? null;
    this._onMessage = opts.onMessage ?? null;
  }

  // -----------------------------------------------------------------------
  // Message queue API
  // -----------------------------------------------------------------------

  /**
   * Emit a processing message.
   * Appended to the internal queue and forwarded to listener if set.
   */
  emit(msg: ProcessingMessage): void {
    this._messages.push(msg);
    if (this._onMessage) {
      this._onMessage(msg);
    }
  }

  /** Get all emitted messages. */
  getMessages(): ReadonlyArray<ProcessingMessage> {
    return this._messages;
  }

  /** Clear the message queue. */
  clearMessages(): void {
    this._messages = [];
  }

  // -----------------------------------------------------------------------
  // Output normalization
  // -----------------------------------------------------------------------

  /**
   * Sanitize memory URIs from output values before sending to client.
   * Replaces internal memory:// URIs with safe placeholders.
   *
   * Port of sanitize_memory_uris_for_client() from types.py.
   */
  static sanitizeForClient(value: unknown): unknown {
    if (value === null || value === undefined) return value;
    if (Array.isArray(value)) {
      return value.map((v) => ProcessingContext.sanitizeForClient(v));
    }
    if (typeof value !== "object") return value;

    const obj = value as Record<string, unknown>;
    const uri = obj.uri;
    const isAssetLike = "type" in obj && typeof uri === "string";

    if (isAssetLike && uri.startsWith("memory://")) {
      const sanitized: Record<string, unknown> = { ...obj };
      if (sanitized.data !== undefined && sanitized.data !== null) {
        sanitized.uri = "";
      } else if (sanitized.asset_id) {
        sanitized.uri = `asset://${String(sanitized.asset_id)}`;
      } else {
        sanitized.uri = "";
      }

      const result: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(sanitized)) {
        if (k === "uri" || k === "data" || k === "asset_id") {
          result[k] = v;
        } else {
          result[k] = ProcessingContext.sanitizeForClient(v);
        }
      }
      return result;
    }

    const result: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(obj)) {
      result[k] = ProcessingContext.sanitizeForClient(v);
    }
    return result;
  }

  /**
   * Resolve a file path against the configured workspace root.
   */
  resolveWorkspacePath(path: string): string {
    return resolveWorkspacePath(this.workspaceDir, path);
  }
}
