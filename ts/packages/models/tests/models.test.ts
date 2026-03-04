import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { setGlobalAdapterResolver, ModelObserver } from "../src/base-model.js";
import { MemoryAdapterFactory } from "../src/memory-adapter.js";
import { field } from "../src/condition-builder.js";
import { Job } from "../src/job.js";
import { Workflow } from "../src/workflow.js";
import { Asset } from "../src/asset.js";
import { Message } from "../src/message.js";
import { Thread } from "../src/thread.js";
import type { ModelClass } from "../src/base-model.js";

// ── Setup ────────────────────────────────────────────────────────────

const factory = new MemoryAdapterFactory();

async function setup() {
  factory.clear();
  setGlobalAdapterResolver((schema) => factory.getAdapter(schema));
  await (Job as unknown as ModelClass).createTable();
  await (Workflow as unknown as ModelClass).createTable();
  await (Asset as unknown as ModelClass).createTable();
  await (Message as unknown as ModelClass).createTable();
  await (Thread as unknown as ModelClass).createTable();
}

// ── Job ──────────────────────────────────────────────────────────────

describe("Job model", () => {
  beforeEach(setup);
  afterEach(() => ModelObserver.clear());

  it("creates with defaults", async () => {
    const job = await (Job as unknown as ModelClass<Job>).create({
      user_id: "u1",
      workflow_id: "w1",
    });
    expect(job.status).toBe("scheduled");
    expect(job.retry_count).toBe(0);
    expect(job.version).toBe(1);
    expect(job.id).toBeTruthy();
    expect(job.created_at).toBeTruthy();
  });

  it("state transitions", async () => {
    const job = await (Job as unknown as ModelClass<Job>).create({
      user_id: "u1",
      workflow_id: "w1",
    });

    job.markRunning("worker-1");
    expect(job.status).toBe("running");
    expect(job.worker_id).toBe("worker-1");
    expect(job.started_at).toBeTruthy();

    job.markSuspended("node-5", { foo: "bar" });
    expect(job.status).toBe("suspended");
    expect(job.suspended_node_id).toBe("node-5");
    expect(job.suspension_state_json).toEqual({ foo: "bar" });

    job.markResumed();
    expect(job.status).toBe("running");
    expect(job.suspended_node_id).toBeNull();

    job.markCompleted();
    expect(job.status).toBe("completed");
    expect(job.finished_at).toBeTruthy();
  });

  it("markFailed records error", async () => {
    const job = await (Job as unknown as ModelClass<Job>).create({
      user_id: "u1",
      workflow_id: "w1",
    });
    job.markFailed("boom");
    expect(job.status).toBe("failed");
    expect(job.error).toBe("boom");
  });

  it("heartbeat and stale check", async () => {
    const job = await (Job as unknown as ModelClass<Job>).create({
      user_id: "u1",
      workflow_id: "w1",
    });
    expect(job.isStale(1000)).toBe(true); // no heartbeat yet

    job.updateHeartbeat();
    expect(job.isStale(60_000)).toBe(false);
  });

  it("paginate by user and status", async () => {
    await (Job as unknown as ModelClass<Job>).create({
      user_id: "u1",
      workflow_id: "w1",
      status: "running",
    });
    await (Job as unknown as ModelClass<Job>).create({
      user_id: "u1",
      workflow_id: "w2",
      status: "completed",
    });
    await (Job as unknown as ModelClass<Job>).create({
      user_id: "u2",
      workflow_id: "w3",
      status: "running",
    });

    const [allForU1] = await Job.paginate("u1");
    expect(allForU1).toHaveLength(2);

    const [runningForU1] = await Job.paginate("u1", { status: "running" });
    expect(runningForU1).toHaveLength(1);
  });
});

// ── Workflow ─────────────────────────────────────────────────────────

describe("Workflow model", () => {
  beforeEach(setup);

  it("creates with defaults", async () => {
    const wf = await (Workflow as unknown as ModelClass<Workflow>).create({
      user_id: "u1",
      name: "My Workflow",
    });
    expect(wf.access).toBe("private");
    expect(wf.graph).toEqual({ nodes: [], edges: [] });
    expect(wf.tags).toEqual([]);
  });

  it("find respects ownership", async () => {
    const wf = await (Workflow as unknown as ModelClass<Workflow>).create({
      user_id: "u1",
      name: "Private WF",
    });

    expect(await Workflow.find("u1", wf.id)).not.toBeNull();
    expect(await Workflow.find("u2", wf.id)).toBeNull();
  });

  it("find allows public access", async () => {
    const wf = await (Workflow as unknown as ModelClass<Workflow>).create({
      user_id: "u1",
      name: "Public WF",
      access: "public",
    });
    expect(await Workflow.find("u2", wf.id)).not.toBeNull();
  });

  it("paginate filters by user", async () => {
    await (Workflow as unknown as ModelClass<Workflow>).create({
      user_id: "u1",
      name: "WF1",
    });
    await (Workflow as unknown as ModelClass<Workflow>).create({
      user_id: "u2",
      name: "WF2",
    });
    const [results] = await Workflow.paginate("u1");
    expect(results).toHaveLength(1);
    expect(results[0].name).toBe("WF1");
  });
});

// ── Asset ────────────────────────────────────────────────────────────

describe("Asset model", () => {
  beforeEach(setup);

  it("creates with defaults", async () => {
    const asset = await (Asset as unknown as ModelClass<Asset>).create({
      user_id: "u1",
      name: "photo.jpg",
      content_type: "image/jpeg",
    });
    expect(asset.parent_id).toBeNull();
    expect(asset.size).toBeNull();
  });

  it("computed properties", async () => {
    const img = await (Asset as unknown as ModelClass<Asset>).create({
      user_id: "u1",
      name: "photo.jpg",
      content_type: "image/jpeg",
    });
    expect(img.isFolder).toBe(false);
    expect(img.fileExtension).toBe("jpg");
    expect(img.hasThumbnail).toBe(true);

    const folder = await (Asset as unknown as ModelClass<Asset>).create({
      user_id: "u1",
      name: "My Folder",
      content_type: "folder",
    });
    expect(folder.isFolder).toBe(true);
    expect(folder.hasThumbnail).toBe(false);
  });

  it("paginate and getChildren", async () => {
    const folder = await (Asset as unknown as ModelClass<Asset>).create({
      user_id: "u1",
      name: "Folder",
      content_type: "folder",
    });
    await (Asset as unknown as ModelClass<Asset>).create({
      user_id: "u1",
      name: "file1.txt",
      content_type: "text/plain",
      parent_id: folder.id,
    });
    await (Asset as unknown as ModelClass<Asset>).create({
      user_id: "u1",
      name: "file2.txt",
      content_type: "text/plain",
      parent_id: folder.id,
    });

    const children = await Asset.getChildren("u1", folder.id);
    expect(children).toHaveLength(2);
  });
});

// ── Message ──────────────────────────────────────────────────────────

describe("Message model", () => {
  beforeEach(setup);

  it("creates with defaults", async () => {
    const msg = await (Message as unknown as ModelClass<Message>).create({
      user_id: "u1",
      thread_id: "t1",
      content: "Hello world",
    });
    expect(msg.role).toBe("user");
    expect(msg.tool_calls).toBeNull();
  });

  it("paginate by thread", async () => {
    await (Message as unknown as ModelClass<Message>).create({
      user_id: "u1",
      thread_id: "t1",
      content: "msg1",
    });
    await (Message as unknown as ModelClass<Message>).create({
      user_id: "u1",
      thread_id: "t1",
      content: "msg2",
    });
    await (Message as unknown as ModelClass<Message>).create({
      user_id: "u1",
      thread_id: "t2",
      content: "msg3",
    });

    const [msgs] = await Message.paginate("t1");
    expect(msgs).toHaveLength(2);
  });
});

// ── Thread ───────────────────────────────────────────────────────────

describe("Thread model", () => {
  beforeEach(setup);

  it("creates with defaults", async () => {
    const thread = await (Thread as unknown as ModelClass<Thread>).create({
      user_id: "u1",
      title: "Test Thread",
    });
    expect(thread.title).toBe("Test Thread");
    expect(thread.created_at).toBeTruthy();
  });

  it("find scoped to user", async () => {
    const thread = await (Thread as unknown as ModelClass<Thread>).create({
      user_id: "u1",
      title: "Private Thread",
    });
    expect(await Thread.find("u1", thread.id)).not.toBeNull();
    expect(await Thread.find("u2", thread.id)).toBeNull();
  });

  it("paginate by user", async () => {
    await (Thread as unknown as ModelClass<Thread>).create({
      user_id: "u1",
      title: "Thread A",
    });
    await (Thread as unknown as ModelClass<Thread>).create({
      user_id: "u2",
      title: "Thread B",
    });

    const [threads] = await Thread.paginate("u1");
    expect(threads).toHaveLength(1);
    expect(threads[0].title).toBe("Thread A");
  });
});
