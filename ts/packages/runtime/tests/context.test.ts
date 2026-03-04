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
import { BaseProvider } from "../src/providers/base-provider.js";
import type { Message, ProviderStreamItem, StreamingAudioChunk } from "../src/providers/types.js";
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

  it("supports hasMessages/popMessage/popMessageAsync", async () => {
    const ctx = new ProcessingContext({ jobId: "j1" });
    expect(ctx.hasMessages()).toBe(false);
    ctx.emit({ type: "job_update", status: "running" });
    expect(ctx.hasMessages()).toBe(true);
    const popped = ctx.popMessage();
    expect(popped?.type).toBe("job_update");
    expect(ctx.hasMessages()).toBe(false);

    const waiter = ctx.popMessageAsync();
    ctx.emit({ type: "job_update", status: "completed" });
    await expect(waiter).resolves.toMatchObject({ type: "job_update", status: "completed" });
  });

  it("tracks latest node and edge statuses", () => {
    const ctx = new ProcessingContext({ jobId: "j1" });
    ctx.emit({
      type: "node_update",
      node_id: "n1",
      node_name: "Test",
      node_type: "test.Node",
      status: "running",
    });
    ctx.emit({
      type: "edge_update",
      workflow_id: "w1",
      edge_id: "e1",
      status: "active",
    });
    expect(ctx.getNodeStatuses().n1).toMatchObject({ type: "node_update", status: "running" });
    expect(ctx.getEdgeStatuses().e1).toMatchObject({ type: "edge_update", status: "active" });
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

  it("materializes asset refs to temp URLs via resolver", async () => {
    const storage = new InMemoryStorageAdapter();
    const ctx = new ProcessingContext({
      jobId: "j1",
      assetOutputMode: "temp_url",
      storage,
      tempUrlResolver: (uri) => `https://temp.local/${encodeURIComponent(uri)}`,
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
    expect(normalized.image.uri.startsWith("https://temp.local/")).toBe(true);
    expect(normalized.image.data).toBeUndefined();
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

describe("ProcessingContext – asset helper methods", () => {
  const assetValue = {
    image: {
      type: "ImageRef",
      uri: "memory://img",
      data: Buffer.from("hello").toString("base64"),
    },
  };

  it("assetsToDataUri converts assets to data URIs", async () => {
    const ctx = new ProcessingContext({ jobId: "j1" });
    const normalized = (await ctx.assetsToDataUri(assetValue)) as { image: { uri: string } };
    expect(normalized.image.uri.startsWith("data:image/png;base64,")).toBe(true);
  });

  it("assetsToStorageUrl converts assets to stored URIs", async () => {
    const storage = new InMemoryStorageAdapter();
    const ctx = new ProcessingContext({ jobId: "j1", storage });
    const normalized = (await ctx.assetsToStorageUrl(assetValue)) as {
      image: { uri: string; data?: unknown };
    };
    expect(normalized.image.uri.startsWith("memory://assets/")).toBe(true);
    expect(normalized.image.data).toBeUndefined();
  });

  it("uploadAssetsToTemp converts assets to temp URLs", async () => {
    const storage = new InMemoryStorageAdapter();
    const ctx = new ProcessingContext({
      jobId: "j1",
      storage,
      tempUrlResolver: (uri) => `https://temp.local/${encodeURIComponent(uri)}`,
    });
    const normalized = (await ctx.uploadAssetsToTemp(assetValue)) as {
      image: { uri: string; data?: unknown };
    };
    expect(normalized.image.uri.startsWith("https://temp.local/")).toBe(true);
    expect(normalized.image.data).toBeUndefined();
  });
});

class MockProvider extends BaseProvider {
  constructor() {
    super("mock");
  }

  async generateMessage(_args: {
    messages: Message[];
    model: string;
    tools?: unknown[];
    maxTokens?: number;
    responseFormat?: Record<string, unknown>;
    jsonSchema?: Record<string, unknown>;
    temperature?: number;
    topP?: number;
    presencePenalty?: number;
    frequencyPenalty?: number;
  }): Promise<Message> {
    return {
      role: "assistant",
      content: "mock-generated-message",
    };
  }

  async *generateMessages(_args: {
    messages: Message[];
    model: string;
    tools?: unknown[];
    maxTokens?: number;
    responseFormat?: Record<string, unknown>;
    jsonSchema?: Record<string, unknown>;
    temperature?: number;
    topP?: number;
    presencePenalty?: number;
    frequencyPenalty?: number;
    audio?: Record<string, unknown>;
  }): AsyncGenerator<ProviderStreamItem> {
    yield { type: "chunk", content: "a", done: false };
    yield { type: "chunk", content: "b", done: true };
  }

  override async *textToSpeech(_args: {
    text: string;
    model: string;
    voice?: string;
    speed?: number;
  }): AsyncGenerator<StreamingAudioChunk> {
    yield { samples: new Int16Array([1, 2, 3]) };
  }
}

describe("ProcessingContext – variables and secrets", () => {
  it("supports get/set and persisted step results", async () => {
    const root = await mkdtemp(join(tmpdir(), "nodetool-ts-vars-"));
    try {
      const ctx = new ProcessingContext({
        jobId: "j1",
        workspaceDir: root,
        variables: { existing: 1 },
      });

      expect(ctx.get("existing", 0)).toBe(1);
      ctx.set("new_key", { ok: true });
      expect(ctx.get("new_key")).toEqual({ ok: true });

      const outPath = await ctx.storeStepResult("step_a", { n: 42 });
      expect(outPath.endsWith("step_a.json")).toBe(true);
      await expect(ctx.loadStepResult("step_a")).resolves.toEqual({ n: 42 });
      await expect(readFile(join(root, "var_new_key.json"), "utf8")).resolves.toContain('"ok": true');
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("supports getSecret/getSecretRequired", async () => {
    const ctx = new ProcessingContext({
      jobId: "j1",
      secretResolver: async (key) => (key === "OPENAI_API_KEY" ? "secret-value" : null),
    });

    await expect(ctx.getSecret("OPENAI_API_KEY")).resolves.toBe("secret-value");
    await expect(ctx.getSecret("MISSING")).resolves.toBeNull();
    await expect(ctx.getSecretRequired("OPENAI_API_KEY")).resolves.toBe("secret-value");
    await expect(ctx.getSecretRequired("MISSING")).rejects.toThrow("Missing required secret: MISSING");
  });
});

describe("ProcessingContext – HTTP helpers", () => {
  it("retries transient responses and downloads bytes/text", async () => {
    let calls = 0;
    const ctx = new ProcessingContext({
      jobId: "j1",
      fetchFn: async () => {
        calls += 1;
        if (calls === 1) {
          return new Response("retry", { status: 503 });
        }
        return new Response("hello", { status: 200 });
      },
    });

    const response = await ctx.httpGet("https://example.com", {
      retry: { maxRetries: 2, backoffMs: 1 },
    });
    expect(response.status).toBe(200);
    expect(calls).toBe(2);

    const bytes = await ctx.downloadFile("https://example.com/file", {
      retry: { maxRetries: 1, backoffMs: 1 },
    });
    expect(new TextDecoder().decode(bytes)).toBe("hello");
    await expect(ctx.downloadText("https://example.com/text")).resolves.toBe("hello");
  });
});

describe("ProcessingContext – provider prediction pipeline", () => {
  it("runs non-stream and emits prediction lifecycle updates", async () => {
    const ctx = new ProcessingContext({ jobId: "j1" });
    ctx.registerProvider("mock", new MockProvider());

    const out = await ctx.runProviderPrediction({
      provider: "mock",
      capability: "generate_message",
      model: "m1",
      nodeId: "n1",
      params: { messages: [{ role: "user", content: "hi" }] },
    });

    expect((out as Message).content).toBe("mock-generated-message");
    const predictionMessages = ctx.getMessages().filter((m) => m.type === "prediction");
    expect(predictionMessages).toHaveLength(2);
    expect((predictionMessages[0] as { status: string }).status).toBe("running");
    expect((predictionMessages[1] as { status: string }).status).toBe("completed");
  });

  it("streams provider capability and emits lifecycle updates", async () => {
    const ctx = new ProcessingContext({ jobId: "j1" });
    ctx.registerProvider("mock", new MockProvider());

    const chunks: ProviderStreamItem[] = [];
    for await (const item of ctx.streamProviderPrediction({
      provider: "mock",
      capability: "generate_messages",
      model: "m1",
      params: { messages: [{ role: "user", content: "hi" }] },
    })) {
      chunks.push(item as ProviderStreamItem);
    }

    expect(chunks).toHaveLength(2);
    const predictionMessages = ctx.getMessages().filter((m) => m.type === "prediction");
    expect(predictionMessages).toHaveLength(2);
    expect((predictionMessages[1] as { status: string }).status).toBe("completed");
  });
});

describe("ProcessingContext – copy and cost tracking", () => {
  it("copies key runtime state", async () => {
    const ctx = new ProcessingContext({
      jobId: "j1",
      workflowId: "w1",
      userId: "u1",
      variables: { x: 1 },
      environment: { APP_ENV: "test" },
      secretResolver: async () => "s",
    });
    ctx.registerProvider("mock", new MockProvider());
    ctx.trackOperationCost("op1", 1.25);

    const cloned = ctx.copy();
    expect(cloned.jobId).toBe("j1");
    expect(cloned.workflowId).toBe("w1");
    expect(cloned.userId).toBe("u1");
    expect(cloned.get("x")).toBe(1);
    expect(cloned.environment.APP_ENV).toBe("test");
    await expect(cloned.getSecretRequired("any")).resolves.toBe("s");
    await expect(cloned.getProvider("mock")).resolves.toBeInstanceOf(MockProvider);
    expect(cloned.getTotalCost()).toBeCloseTo(1.25);
  });

  it("tracks/reset costs", () => {
    const ctx = new ProcessingContext({ jobId: "j1" });
    ctx.trackOperationCost("tokens", 0.5, { provider: "openai" });
    ctx.addToTotalCost(0.2);
    expect(ctx.getTotalCost()).toBeCloseTo(0.7);
    expect(ctx.getOperationCosts()).toHaveLength(1);
    expect(ctx.getOperationCosts()[0]).toMatchObject({ operation: "tokens", provider: "openai" });
    ctx.resetTotalCost();
    expect(ctx.getTotalCost()).toBe(0);
    expect(ctx.getOperationCosts()).toHaveLength(0);
  });
});

describe("ProcessingContext – node result cache helpers", () => {
  it("generates deterministic cache keys and stores/retrieves results", async () => {
    const ctx = new ProcessingContext({ jobId: "j1", userId: "u1" });
    const props = { a: 1, b: "x" };
    const k1 = ctx.generateNodeCacheKey("nodetool.test.Node", props);
    const k2 = ctx.generateNodeCacheKey("nodetool.test.Node", { b: "x", a: 1 });
    expect(k1).toBe(k2);

    await ctx.cacheResult("nodetool.test.Node", props, { out: 123 }, 60);
    await expect(ctx.getCachedResult("nodetool.test.Node", props)).resolves.toEqual({ out: 123 });
  });
});

describe("ProcessingContext – memory helpers", () => {
  it("tracks memory:// values and reports stats", () => {
    const ctx = new ProcessingContext({ jobId: "j1" });
    ctx.set("memory://assets/a", { id: 1 });
    ctx.set("memory://assets/b", { id: 2 });
    ctx.set("memory://tmp/c", { id: 3 });

    expect(ctx.getMemoryStats()).toEqual({
      total: 3,
      byPrefix: { assets: 2, tmp: 1 },
    });
  });

  it("clears memory entries globally or by pattern", () => {
    const ctx = new ProcessingContext({ jobId: "j1" });
    ctx.set("memory://assets/a", 1);
    ctx.set("memory://tmp/b", 2);
    ctx.clearMemory("assets");
    expect(ctx.getMemoryStats()).toEqual({
      total: 1,
      byPrefix: { tmp: 1 },
    });

    ctx.clearMemory();
    expect(ctx.getMemoryStats()).toEqual({ total: 0, byPrefix: {} });
  });
});
