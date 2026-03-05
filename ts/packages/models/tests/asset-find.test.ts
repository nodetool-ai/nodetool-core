import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { setGlobalAdapterResolver, ModelObserver } from "../src/base-model.js";
import { MemoryAdapterFactory } from "../src/memory-adapter.js";
import { Asset } from "../src/asset.js";
import type { ModelClass } from "../src/base-model.js";

const factory = new MemoryAdapterFactory();

async function setup() {
  factory.clear();
  setGlobalAdapterResolver((schema) => factory.getAdapter(schema));
  await (Asset as unknown as ModelClass).createTable();
}

describe("Asset.find", () => {
  beforeEach(setup);
  afterEach(() => ModelObserver.clear());

  it("returns asset when user matches", async () => {
    const asset = await (Asset as unknown as ModelClass<Asset>).create({
      user_id: "u1",
      name: "test.jpg",
      content_type: "image/jpeg",
    });

    const found = await Asset.find("u1", asset.id);
    expect(found).not.toBeNull();
    expect(found!.id).toBe(asset.id);
    expect(found!.name).toBe("test.jpg");
  });

  it("returns null when user does not match", async () => {
    const asset = await (Asset as unknown as ModelClass<Asset>).create({
      user_id: "u1",
      name: "test.jpg",
      content_type: "image/jpeg",
    });

    const found = await Asset.find("u2", asset.id);
    expect(found).toBeNull();
  });

  it("returns null for nonexistent asset id", async () => {
    const found = await Asset.find("u1", "nonexistent-id");
    expect(found).toBeNull();
  });
});

describe("Asset.paginate – additional filters", () => {
  beforeEach(setup);
  afterEach(() => ModelObserver.clear());

  it("filters by workflowId", async () => {
    await (Asset as unknown as ModelClass<Asset>).create({
      user_id: "u1",
      name: "a.txt",
      content_type: "text/plain",
      workflow_id: "wf1",
    });
    await (Asset as unknown as ModelClass<Asset>).create({
      user_id: "u1",
      name: "b.txt",
      content_type: "text/plain",
      workflow_id: "wf2",
    });

    const [results] = await Asset.paginate("u1", { workflowId: "wf1" });
    expect(results).toHaveLength(1);
    expect(results[0].name).toBe("a.txt");
  });

  it("filters by nodeId", async () => {
    await (Asset as unknown as ModelClass<Asset>).create({
      user_id: "u1",
      name: "a.txt",
      content_type: "text/plain",
      node_id: "n1",
    });
    await (Asset as unknown as ModelClass<Asset>).create({
      user_id: "u1",
      name: "b.txt",
      content_type: "text/plain",
      node_id: "n2",
    });

    const [results] = await Asset.paginate("u1", { nodeId: "n1" });
    expect(results).toHaveLength(1);
    expect(results[0].name).toBe("a.txt");
  });

  it("filters by jobId", async () => {
    await (Asset as unknown as ModelClass<Asset>).create({
      user_id: "u1",
      name: "a.txt",
      content_type: "text/plain",
      job_id: "j1",
    });
    await (Asset as unknown as ModelClass<Asset>).create({
      user_id: "u1",
      name: "b.txt",
      content_type: "text/plain",
      job_id: "j2",
    });

    const [results] = await Asset.paginate("u1", { jobId: "j1" });
    expect(results).toHaveLength(1);
    expect(results[0].name).toBe("a.txt");
  });
});
