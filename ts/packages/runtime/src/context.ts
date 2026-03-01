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
import { randomUUID } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, normalize, relative, resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import type { BaseProvider } from "./providers/base-provider.js";

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

export type AssetOutputMode = "python" | "data_uri" | "storage_url" | "workspace" | "raw";

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

function joinStorageKey(prefix: string | undefined, key: string): string {
  const normalizedKey = normalizeStorageKey(key);
  if (!prefix) return normalizedKey;
  const normalizedPrefix = normalizeStorageKey(prefix);
  return `${normalizedPrefix}/${normalizedKey}`;
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

export interface S3Client {
  putObject(input: {
    bucket: string;
    key: string;
    body: Uint8Array;
    contentType?: string;
  }): Promise<void>;
  getObject(input: { bucket: string; key: string }): Promise<Uint8Array | null>;
  headObject(input: { bucket: string; key: string }): Promise<boolean>;
}

/**
 * S3-backed storage adapter with injected client operations.
 *
 * This avoids hard-coupling runtime to any specific SDK while providing
 * predictable URI behavior (`s3://bucket/key`).
 */
export class S3StorageAdapter implements StorageAdapter {
  readonly bucket: string;
  readonly prefix: string | null;
  readonly client: S3Client;

  constructor(opts: { bucket: string; client: S3Client; prefix?: string }) {
    if (!opts.bucket) {
      throw new Error("S3 bucket is required");
    }
    this.bucket = opts.bucket;
    this.client = opts.client;
    this.prefix = opts.prefix ? normalizeStorageKey(opts.prefix) : null;
  }

  private keyForStore(key: string): string {
    return joinStorageKey(this.prefix ?? undefined, key);
  }

  private parseUri(uri: string): { bucket: string; key: string } | null {
    if (!uri.startsWith("s3://")) return null;
    const withoutScheme = uri.slice("s3://".length);
    const slashIndex = withoutScheme.indexOf("/");
    if (slashIndex <= 0 || slashIndex === withoutScheme.length - 1) {
      return null;
    }
    const bucket = withoutScheme.slice(0, slashIndex);
    const key = withoutScheme.slice(slashIndex + 1);
    return { bucket, key };
  }

  async store(key: string, data: Uint8Array, contentType?: string): Promise<string> {
    const objectKey = this.keyForStore(key);
    await this.client.putObject({
      bucket: this.bucket,
      key: objectKey,
      body: data,
      contentType,
    });
    return `s3://${this.bucket}/${objectKey}`;
  }

  async retrieve(uri: string): Promise<Uint8Array | null> {
    const parsed = this.parseUri(uri);
    if (!parsed) return null;
    if (parsed.bucket !== this.bucket) return null;
    return this.client.getObject(parsed);
  }

  async exists(uri: string): Promise<boolean> {
    const parsed = this.parseUri(uri);
    if (!parsed) return false;
    if (parsed.bucket !== this.bucket) return false;
    return this.client.headObject(parsed);
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
  readonly assetOutputMode: AssetOutputMode;

  /** Message queue: all emitted processing messages. */
  private _messages: ProcessingMessage[] = [];

  /** Optional message listener (for real-time streaming). */
  private _onMessage: ((msg: ProcessingMessage) => void) | null = null;

  /** Cache adapter. */
  readonly cache: CacheAdapter;

  /** Storage adapter (optional, for asset handling). */
  readonly storage: StorageAdapter | null;
  /** Optional async provider resolver by provider id. */
  private _providerResolver:
    | ((providerId: string) => Promise<BaseProvider> | BaseProvider)
    | null = null;
  /** Provider cache keyed by provider id. */
  private _providers = new Map<string, BaseProvider>();

  constructor(opts: {
    jobId: string;
    workflowId?: string | null;
    userId?: string;
    workspaceDir?: string | null;
    assetOutputMode?: AssetOutputMode;
    cache?: CacheAdapter;
    storage?: StorageAdapter | null;
    onMessage?: (msg: ProcessingMessage) => void;
  }) {
    this.jobId = opts.jobId;
    this.workflowId = opts.workflowId ?? null;
    this.userId = opts.userId ?? "default";
    this.workspaceDir = opts.workspaceDir ?? null;
    this.assetOutputMode = opts.assetOutputMode ?? "python";
    this.cache = opts.cache ?? new MemoryCache();
    this.storage = opts.storage ?? null;
    this._onMessage = opts.onMessage ?? null;
  }

  // -----------------------------------------------------------------------
  // Provider resolution
  // -----------------------------------------------------------------------

  setProviderResolver(
    resolver: (providerId: string) => Promise<BaseProvider> | BaseProvider
  ): void {
    this._providerResolver = resolver;
  }

  registerProvider(providerId: string, provider: BaseProvider): void {
    this._providers.set(providerId, provider);
  }

  async getProvider(providerId: string): Promise<BaseProvider> {
    if (!providerId || providerId.trim() === "") {
      throw new Error("providerId is required");
    }

    const cached = this._providers.get(providerId);
    if (cached) return cached;

    if (!this._providerResolver) {
      throw new Error(`No provider registered for '${providerId}' and no resolver configured`);
    }

    const resolved = await this._providerResolver(providerId);
    this._providers.set(providerId, resolved);
    return resolved;
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

  private static isAssetLike(value: unknown): value is Record<string, unknown> {
    if (!value || typeof value !== "object" || Array.isArray(value)) return false;
    const v = value as Record<string, unknown>;
    return "type" in v && ("uri" in v || "data" in v || "asset_id" in v);
  }

  private static guessAssetMime(asset: Record<string, unknown>): string {
    const explicit = asset.mime_type ?? asset.content_type;
    if (typeof explicit === "string" && explicit) return explicit;

    const type = String(asset.type ?? "").toLowerCase();
    if (type.includes("image")) return "image/png";
    if (type.includes("audio")) return "audio/wav";
    if (type.includes("video")) return "video/mp4";
    if (type.includes("text")) return "text/plain";
    if (type.includes("model3d")) return "model/gltf-binary";
    return "application/octet-stream";
  }

  private static extForMime(mime: string): string {
    const map: Record<string, string> = {
      "image/png": "png",
      "image/jpeg": "jpg",
      "image/webp": "webp",
      "audio/wav": "wav",
      "audio/mpeg": "mp3",
      "video/mp4": "mp4",
      "text/plain": "txt",
      "application/json": "json",
      "model/gltf-binary": "glb",
    };
    return map[mime] ?? "bin";
  }

  private static decodeAssetData(data: unknown): Uint8Array | null {
    if (data === null || data === undefined) return null;
    if (data instanceof Uint8Array) return data;
    if (Array.isArray(data) && data.every((v) => Number.isInteger(v))) {
      return new Uint8Array(data as number[]);
    }
    if (typeof data === "string") {
      return Uint8Array.from(Buffer.from(data, "base64"));
    }
    return null;
  }

  private async getAssetBytes(asset: Record<string, unknown>): Promise<Uint8Array | null> {
    const decoded = ProcessingContext.decodeAssetData(asset.data);
    if (decoded) return decoded;

    const uri = asset.uri;
    if (typeof uri !== "string" || !this.storage) return null;
    return this.storage.retrieve(uri);
  }

  private async materializeAsset(
    asset: Record<string, unknown>,
    mode: AssetOutputMode
  ): Promise<Record<string, unknown>> {
    if (mode === "python" || mode === "raw") {
      return asset;
    }

    const bytes = await this.getAssetBytes(asset);
    if (!bytes) return asset;

    const mime = ProcessingContext.guessAssetMime(asset);

    if (mode === "data_uri") {
      const encoded = Buffer.from(bytes).toString("base64");
      return {
        ...asset,
        uri: `data:${mime};base64,${encoded}`,
      };
    }

    if (mode === "storage_url") {
      if (!this.storage) return asset;
      const key = `assets/${randomUUID()}.${ProcessingContext.extForMime(mime)}`;
      const uri = await this.storage.store(key, bytes, mime);
      return {
        ...asset,
        uri,
        data: undefined,
      };
    }

    if (mode === "workspace") {
      if (!this.workspaceDir) {
        throw new Error("workspace_dir is required for workspace asset output");
      }
      const workspaceAssets = new FileStorageAdapter(resolveWorkspacePath(this.workspaceDir, "assets"));
      const key = `${randomUUID()}.${ProcessingContext.extForMime(mime)}`;
      const uri = await workspaceAssets.store(key, bytes, mime);
      return {
        ...asset,
        uri,
        data: undefined,
      };
    }

    return asset;
  }

  /**
   * Recursively normalize workflow outputs, materializing asset-like values
   * according to the selected output mode.
   */
  async normalizeOutputValue(value: unknown, mode: AssetOutputMode = this.assetOutputMode): Promise<unknown> {
    if (value === null || value === undefined) return value;

    if (Array.isArray(value)) {
      return Promise.all(value.map((item) => this.normalizeOutputValue(item, mode)));
    }

    if (ProcessingContext.isAssetLike(value)) {
      return this.materializeAsset(value, mode);
    }

    if (typeof value === "object") {
      const entries = Object.entries(value as Record<string, unknown>);
      const normalized = await Promise.all(
        entries.map(async ([k, v]) => [k, await this.normalizeOutputValue(v, mode)] as const)
      );
      return Object.fromEntries(normalized);
    }

    return value;
  }
}
