/**
 * ProcessingContext – runtime context for node execution.
 *
 * Port of src/nodetool/workflows/processing_context.py.
 *
 * Provides:
 *   - Message queue for emitting ProcessingMessages.
 *   - Cache get/set interface.
 *   - Output normalization (sanitize memory URIs, etc.).
 *   - Asset handling stubs (to be implemented with storage adapters).
 */

import type { ProcessingMessage } from "@nodetool/protocol";

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
// Storage adapter interface (stub for Phase 6)
// ---------------------------------------------------------------------------

export interface StorageAdapter {
  /** Store an asset and return a URI. */
  store(key: string, data: Uint8Array, contentType?: string): Promise<string>;

  /** Retrieve an asset by URI. */
  retrieve(uri: string): Promise<Uint8Array | null>;

  /** Check if an asset exists. */
  exists(uri: string): Promise<boolean>;
}

// ---------------------------------------------------------------------------
// ProcessingContext
// ---------------------------------------------------------------------------

export class ProcessingContext {
  readonly jobId: string;
  readonly workflowId: string | null;
  readonly userId: string;

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
    cache?: CacheAdapter;
    storage?: StorageAdapter | null;
    onMessage?: (msg: ProcessingMessage) => void;
  }) {
    this.jobId = opts.jobId;
    this.workflowId = opts.workflowId ?? null;
    this.userId = opts.userId ?? "default";
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
    if (typeof value === "string") {
      if (value.startsWith("memory://")) {
        return "[memory reference]";
      }
      return value;
    }
    if (Array.isArray(value)) {
      return value.map((v) => ProcessingContext.sanitizeForClient(v));
    }
    if (value !== null && typeof value === "object") {
      const result: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
        result[k] = ProcessingContext.sanitizeForClient(v);
      }
      return result;
    }
    return value;
  }
}
