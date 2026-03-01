/**
 * ProcessingContext tests.
 */

import { describe, it, expect } from "vitest";
import { ProcessingContext, MemoryCache } from "../src/context.js";
import type { ProcessingMessage, NodeUpdate } from "@nodetool/protocol";

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
  it("replaces memory:// URIs", () => {
    expect(ProcessingContext.sanitizeForClient("memory://abc")).toBe(
      "[memory reference]"
    );
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
      url: "[memory reference]",
      name: "test",
    });
  });

  it("sanitizes arrays", () => {
    const result = ProcessingContext.sanitizeForClient([
      "memory://a",
      "safe",
    ]);
    expect(result).toEqual(["[memory reference]", "safe"]);
  });

  it("passes through non-string primitives", () => {
    expect(ProcessingContext.sanitizeForClient(42)).toBe(42);
    expect(ProcessingContext.sanitizeForClient(null)).toBe(null);
    expect(ProcessingContext.sanitizeForClient(true)).toBe(true);
  });
});
