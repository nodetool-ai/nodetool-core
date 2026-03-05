/**
 * Additional NodeInbox tests for coverage:
 *  - iterAnyWithEnvelope
 *  - put to unregistered handle (auto-create)
 *  - backpressure with closeAll
 *  - prepend to nonexistent handle
 *  - markSourceDone when count is already 0
 */

import { describe, it, expect } from "vitest";
import { NodeInbox } from "../src/inbox.js";

async function collect<T>(gen: AsyncGenerator<T>): Promise<T[]> {
  const items: T[] = [];
  for await (const item of gen) {
    items.push(item);
  }
  return items;
}

describe("NodeInbox – iterAnyWithEnvelope", () => {
  it("yields (handle, envelope) tuples in arrival order", async () => {
    const inbox = new NodeInbox();
    inbox.addUpstream("a", 1);
    inbox.addUpstream("b", 1);

    await inbox.put("a", "data-a", { source: "node1" });
    await inbox.put("b", "data-b", { source: "node2" });
    inbox.markSourceDone("a");
    inbox.markSourceDone("b");

    const items = await collect(inbox.iterAnyWithEnvelope());

    expect(items).toHaveLength(2);
    expect(items[0][0]).toBe("a");
    expect(items[0][1].data).toBe("data-a");
    expect(items[0][1].metadata).toEqual({ source: "node1" });
    expect(items[0][1].event_id).toBeTruthy();
    expect(items[0][1].timestamp).toBeGreaterThan(0);

    expect(items[1][0]).toBe("b");
    expect(items[1][1].data).toBe("data-b");
  });

  it("completes when all handles are drained", async () => {
    const inbox = new NodeInbox();
    inbox.addUpstream("x", 1);

    await inbox.put("x", 42);
    inbox.markSourceDone("x");

    const items = await collect(inbox.iterAnyWithEnvelope());
    expect(items).toHaveLength(1);
    expect(items[0][0]).toBe("x");
    expect(items[0][1].data).toBe(42);
  });

  it("completes when inbox is closed", async () => {
    const inbox = new NodeInbox();
    inbox.addUpstream("a", 1);

    const consumePromise = collect(inbox.iterAnyWithEnvelope());

    await new Promise((r) => setTimeout(r, 10));
    await inbox.closeAll();

    const items = await consumePromise;
    expect(items).toEqual([]);
  });
});

describe("NodeInbox – iterAnyWithEnvelope waiting for async data", () => {
  it("waits for data when nothing is buffered yet", async () => {
    const inbox = new NodeInbox();
    inbox.addUpstream("a", 1);

    // Start consuming before data is available
    const items: Array<[string, unknown]> = [];
    const consumePromise = (async () => {
      for await (const [handle, envelope] of inbox.iterAnyWithEnvelope()) {
        items.push([handle, envelope.data]);
      }
    })();

    // Wait a tick then provide data
    await new Promise((r) => setTimeout(r, 10));
    await inbox.put("a", "delayed-data");
    inbox.markSourceDone("a");

    await consumePromise;

    expect(items).toHaveLength(1);
    expect(items[0]).toEqual(["a", "delayed-data"]);
  });
});

describe("NodeInbox – tryPopAnyWithEnvelope stale entries", () => {
  it("skips stale arrival entries in envelope mode", async () => {
    const inbox = new NodeInbox();
    inbox.addUpstream("a", 1);
    inbox.addUpstream("b", 1);

    await inbox.put("a", "a1");
    await inbox.put("b", "b1");

    // Pop first using tryPopAnyWithEnvelope
    const popped1 = inbox.tryPopAnyWithEnvelope();
    expect(popped1).not.toBeNull();
    expect(popped1![0]).toBe("a");
    expect(popped1![1].data).toBe("a1");

    const popped2 = inbox.tryPopAnyWithEnvelope();
    expect(popped2).not.toBeNull();
    expect(popped2![0]).toBe("b");
    expect(popped2![1].data).toBe("b1");

    expect(inbox.tryPopAnyWithEnvelope()).toBeNull();
  });
});

describe("NodeInbox – auto-create handle on put", () => {
  it("creates buffer for unregistered handle", async () => {
    const inbox = new NodeInbox();
    // No addUpstream call, put to unknown handle
    await inbox.put("unknown", "value");

    expect(inbox.hasBuffered("unknown")).toBe(true);
    const popped = inbox.tryPopAny();
    expect(popped).toEqual(["unknown", "value"]);
  });
});

describe("NodeInbox – backpressure with closeAll", () => {
  it("unblocks put waiters when closed", async () => {
    const inbox = new NodeInbox(1); // buffer limit 1
    inbox.addUpstream("a", 1);

    await inbox.put("a", 1); // fills buffer

    let putResolved = false;
    const putPromise = inbox.put("a", 2).then(() => {
      putResolved = true;
    });

    await new Promise((r) => setTimeout(r, 10));
    expect(putResolved).toBe(false);

    // Close should unblock the put
    await inbox.closeAll();
    await putPromise;
    expect(putResolved).toBe(true);
  });

  it("rejects writes after close", async () => {
    const inbox = new NodeInbox();
    inbox.addUpstream("a", 1);

    await inbox.closeAll();
    await inbox.put("a", "should-be-ignored");

    expect(inbox.hasBuffered("a")).toBe(false);
  });
});

describe("NodeInbox – prepend edge cases", () => {
  it("prepend to nonexistent handle does nothing", () => {
    const inbox = new NodeInbox();
    // Should not throw
    inbox.prepend("nonexistent", {
      data: "test",
      metadata: {},
      timestamp: Date.now(),
      event_id: "test-id",
    });
    expect(inbox.hasAny()).toBe(false);
  });
});

describe("NodeInbox – markSourceDone edge cases", () => {
  it("markSourceDone when count is already 0 does not go negative", () => {
    const inbox = new NodeInbox();
    inbox.addUpstream("a", 1);
    inbox.markSourceDone("a");
    inbox.markSourceDone("a"); // extra call
    expect(inbox.isOpen("a")).toBe(false);
  });

  it("markSourceDone on unregistered handle does nothing", () => {
    const inbox = new NodeInbox();
    inbox.markSourceDone("unknown"); // should not throw
    expect(inbox.isOpen("unknown")).toBe(false);
  });
});

describe("NodeInbox – hasBuffered for unregistered handle", () => {
  it("returns false for handle with no buffer", () => {
    const inbox = new NodeInbox();
    expect(inbox.hasBuffered("nonexistent")).toBe(false);
  });
});

describe("NodeInbox – tryPopAnyWithEnvelope stale entries", () => {
  it("skips stale arrival entries", async () => {
    const inbox = new NodeInbox();
    inbox.addUpstream("a", 1);
    inbox.addUpstream("b", 1);

    await inbox.put("a", "a1");
    await inbox.put("b", "b1");

    // Pop from specific handle buffer directly to create stale arrival entry
    const popped1 = inbox.tryPopAny();
    expect(popped1).toEqual(["a", "a1"]);

    const popped2 = inbox.tryPopAny();
    expect(popped2).toEqual(["b", "b1"]);

    expect(inbox.tryPopAny()).toBeNull();
  });
});
