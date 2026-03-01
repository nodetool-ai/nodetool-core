/**
 * ProcessingContext tests.
 */

import { describe, it, expect } from "vitest";
import {
  ProcessingContext,
  MemoryCache,
  InMemoryStorageAdapter,
  FileStorageAdapter,
  S3StorageAdapter,
  resolveWorkspacePath,
  type S3Client,
} from "../src/context.js";
import type { ProcessingMessage, NodeUpdate } from "@nodetool/protocol";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";

describe("ProcessingContext – message queue", () => {
  it("collects emitted messages", () => {
    const ctx = new ProcessingContext({ jobId: "j1" });

    const msg: NodeUpdate = {
      type: "node_update",
      node_id: "n1",
      node_name: "Test",
      node_type: "test.Test",
      status: "running",
    };
    ctx.emit(msg);

    expect(ctx.getMessages()).toHaveLength(1);
    expect(ctx.getMessages()[0]).toBe(msg);
  });

  it("calls onMessage listener", () => {
    const received: ProcessingMessage[] = [];
    const ctx = new ProcessingContext({
      jobId: "j1",
      onMessage: (msg) => received.push(msg),
    });

    ctx.emit({
      type: "job_update",
      status: "running",
    });

    expect(received).toHaveLength(1);
    expect(received[0].type).toBe("job_update");
  });

  it("clearMessages empties the queue", () => {
    const ctx = new ProcessingContext({ jobId: "j1" });
    ctx.emit({ type: "job_update", status: "running" });
    ctx.clearMessages();
    expect(ctx.getMessages()).toHaveLength(0);
  });
});

describe("MemoryCache", () => {
  it("stores and retrieves values", async () => {
    const cache = new MemoryCache();
    await cache.set("key1", { data: 42 });
    expect(await cache.get("key1")).toEqual({ data: 42 });
  });

  it("returns undefined for missing keys", async () => {
    const cache = new MemoryCache();
    expect(await cache.get("missing")).toBeUndefined();
  });

  it("has() checks existence", async () => {
    const cache = new MemoryCache();
    expect(await cache.has("key")).toBe(false);
    await cache.set("key", 1);
    expect(await cache.has("key")).toBe(true);
  });

  it("delete() removes entries", async () => {
    const cache = new MemoryCache();
    await cache.set("key", 1);
    await cache.delete("key");
    expect(await cache.has("key")).toBe(false);
  });

  it("respects TTL", async () => {
    const cache = new MemoryCache();
    await cache.set("key", 1, 0.05); // 50ms TTL
    expect(await cache.get("key")).toBe(1);

    await new Promise((r) => setTimeout(r, 60));
    expect(await cache.get("key")).toBeUndefined();
  });
});

describe("ProcessingContext.sanitizeForClient", () => {
  it("does not rewrite plain memory:// strings", () => {
    expect(ProcessingContext.sanitizeForClient("memory://abc")).toBe("memory://abc");
  });

  it("preserves normal strings", () => {
    expect(ProcessingContext.sanitizeForClient("hello")).toBe("hello");
  });

  it("sanitizes nested objects", () => {
    const result = ProcessingContext.sanitizeForClient({
      url: "memory://img1",
      name: "test",
    });
    expect(result).toEqual({
      url: "memory://img1",
      name: "test",
    });
  });

  it("sanitizes arrays", () => {
    const result = ProcessingContext.sanitizeForClient([
      "memory://a",
      "safe",
    ]);
    expect(result).toEqual(["memory://a", "safe"]);
  });

  it("passes through non-string primitives", () => {
    expect(ProcessingContext.sanitizeForClient(42)).toBe(42);
    expect(ProcessingContext.sanitizeForClient(null)).toBe(null);
    expect(ProcessingContext.sanitizeForClient(true)).toBe(true);
  });

  it("sanitizes memory uri in serialized asset refs with inline data", () => {
    const value = {
      type: "ImageRef",
      uri: "memory://img-1",
      data: "base64-data",
      meta: { nested: { type: "TextRef", uri: "memory://txt-1", data: "x" } },
    };
    const result = ProcessingContext.sanitizeForClient(value);
    expect(result).toEqual({
      type: "ImageRef",
      uri: "",
      data: "base64-data",
      meta: { nested: { type: "TextRef", uri: "", data: "x" } },
    });
  });

  it("sanitizes memory uri in serialized asset refs with asset_id", () => {
    const value = {
      type: "ImageRef",
      uri: "memory://img-1",
      data: null,
      asset_id: "a123",
    };
    const result = ProcessingContext.sanitizeForClient(value);
    expect(result).toEqual({
      type: "ImageRef",
      uri: "asset://a123",
      data: null,
      asset_id: "a123",
    });
  });
});

describe("Storage adapters", () => {
  it("InMemoryStorageAdapter stores and retrieves bytes", async () => {
    const storage = new InMemoryStorageAdapter();
    const uri = await storage.store("assets/test.txt", new Uint8Array([1, 2, 3]));

    expect(uri).toBe("memory://assets/test.txt");
    expect(await storage.exists(uri)).toBe(true);
    expect(await storage.retrieve(uri)).toEqual(new Uint8Array([1, 2, 3]));
  });

  it("InMemoryStorageAdapter returns null/false for unknown URIs", async () => {
    const storage = new InMemoryStorageAdapter();
    expect(await storage.retrieve("memory://missing.bin")).toBeNull();
    expect(await storage.exists("memory://missing.bin")).toBe(false);
    expect(await storage.retrieve("file:///tmp/not-memory")).toBeNull();
  });

  it("FileStorageAdapter stores and retrieves bytes under root", async () => {
    const root = await mkdtemp(join(tmpdir(), "nodetool-ts-runtime-"));
    try {
      const storage = new FileStorageAdapter(root);
      const uri = await storage.store(
        "assets/out.bin",
        new Uint8Array([9, 8, 7, 6])
      );

      expect(uri.startsWith("file://")).toBe(true);
      expect(await storage.exists(uri)).toBe(true);
      const bytes = await storage.retrieve(uri);
      expect(bytes).not.toBeNull();
      expect(Array.from(bytes ?? [])).toEqual([9, 8, 7, 6]);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("FileStorageAdapter rejects traversal keys", async () => {
    const root = await mkdtemp(join(tmpdir(), "nodetool-ts-runtime-"));
    try {
      const storage = new FileStorageAdapter(root);
      await expect(storage.store("../escape.txt", new Uint8Array([1]))).rejects.toThrow(
        "Invalid storage key"
      );
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("S3StorageAdapter stores and returns s3 uri", async () => {
    const store = new Map<string, Uint8Array>();
    const client: S3Client = {
      async putObject(input) {
        store.set(`${input.bucket}/${input.key}`, new Uint8Array(input.body));
      },
      async getObject(input) {
        return store.get(`${input.bucket}/${input.key}`) ?? null;
      },
      async headObject(input) {
        return store.has(`${input.bucket}/${input.key}`);
      },
    };

    const storage = new S3StorageAdapter({
      bucket: "test-bucket",
      prefix: "runs/r1",
      client,
    });
    const uri = await storage.store(
      "assets/out.bin",
      new Uint8Array([4, 5, 6]),
      "application/octet-stream"
    );

    expect(uri).toBe("s3://test-bucket/runs/r1/assets/out.bin");
    expect(await storage.exists(uri)).toBe(true);
    expect(await storage.retrieve(uri)).toEqual(new Uint8Array([4, 5, 6]));
  });

  it("S3StorageAdapter returns null/false for other buckets or invalid uri", async () => {
    const client: S3Client = {
      async putObject() {},
      async getObject() {
        return new Uint8Array([1]);
      },
      async headObject() {
        return true;
      },
    };

    const storage = new S3StorageAdapter({ bucket: "bucket-a", client });
    expect(await storage.retrieve("s3://bucket-b/key")).toBeNull();
    expect(await storage.exists("s3://bucket-b/key")).toBe(false);
    expect(await storage.retrieve("file:///tmp/nope")).toBeNull();
    expect(await storage.exists("file:///tmp/nope")).toBe(false);
  });
});

describe("workspace path resolution", () => {
  it("resolves /workspace/ prefix", () => {
    const root = "/tmp/nodetool-workspace";
    expect(resolveWorkspacePath(root, "/workspace/out/a.txt")).toBe(
      "/tmp/nodetool-workspace/out/a.txt"
    );
  });

  it("resolves relative path", () => {
    const root = "/tmp/nodetool-workspace";
    expect(resolveWorkspacePath(root, "out/a.txt")).toBe(
      "/tmp/nodetool-workspace/out/a.txt"
    );
  });

  it("rejects traversal outside workspace", () => {
    const root = "/tmp/nodetool-workspace";
    expect(() => resolveWorkspacePath(root, "../etc/passwd")).toThrow(
      "outside the workspace directory"
    );
  });

  it("context.resolveWorkspacePath delegates to helper", () => {
    const ctx = new ProcessingContext({
      jobId: "j1",
      workspaceDir: "/tmp/nodetool-workspace",
    });
    expect(ctx.resolveWorkspacePath("workspace/out.json")).toBe(
      "/tmp/nodetool-workspace/out.json"
    );
  });
});

describe("output normalization", () => {
  it("materializes asset refs as data URIs", async () => {
    const ctx = new ProcessingContext({ jobId: "j1", assetOutputMode: "data_uri" });
    const value = {
      image: {
        type: "ImageRef",
        uri: "memory://img",
        data: Buffer.from("hello").toString("base64"),
      },
    };

    const normalized = (await ctx.normalizeOutputValue(value)) as {
      image: { uri: string };
    };
    expect(normalized.image.uri.startsWith("data:image/png;base64,")).toBe(true);
  });

  it("materializes asset refs to storage URLs via adapter", async () => {
    const storage = new InMemoryStorageAdapter();
    const ctx = new ProcessingContext({
      jobId: "j1",
      assetOutputMode: "storage_url",
      storage,
    });
    const value = {
      image: {
        type: "ImageRef",
        uri: "memory://img",
        data: Buffer.from("hello").toString("base64"),
      },
    };

    const normalized = (await ctx.normalizeOutputValue(value)) as {
      image: { uri: string; data?: unknown };
    };
    expect(normalized.image.uri.startsWith("memory://assets/")).toBe(true);
    expect(normalized.image.data).toBeUndefined();
    expect(await storage.exists(normalized.image.uri)).toBe(true);
  });

  it("materializes asset refs into workspace files", async () => {
    const root = await mkdtemp(join(tmpdir(), "nodetool-ts-workspace-"));
    try {
      const ctx = new ProcessingContext({
        jobId: "j1",
        assetOutputMode: "workspace",
        workspaceDir: root,
      });
      const value = {
        image: {
          type: "ImageRef",
          uri: "memory://img",
          data: Buffer.from("hello").toString("base64"),
        },
      };

      const normalized = (await ctx.normalizeOutputValue(value)) as {
        image: { uri: string; data?: unknown };
      };
      expect(normalized.image.uri.startsWith("file://")).toBe(true);
      const bytes = await readFile(fileURLToPath(normalized.image.uri));
      expect(bytes.toString("utf8")).toBe("hello");
      expect(normalized.image.data).toBeUndefined();
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
});
