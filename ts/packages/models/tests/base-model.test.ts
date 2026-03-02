import { describe, it, expect, beforeEach, afterEach } from "vitest";
import {
  DBModel,
  ModelObserver,
  ModelChangeEvent,
  createTimeOrderedUuid,
  computeEtag,
  setGlobalAdapterResolver,
} from "../src/base-model.js";
import { MemoryAdapterFactory } from "../src/memory-adapter.js";
import type { TableSchema, Row } from "../src/database-adapter.js";
import { field } from "../src/condition-builder.js";

// ── Test model ───────────────────────────────────────────────────────

const TEST_SCHEMA: TableSchema = {
  table_name: "test_records",
  primary_key: "id",
  columns: {
    id: { type: "string" },
    name: { type: "string" },
    value: { type: "number" },
  },
};

class TestRecord extends DBModel {
  static override schema = TEST_SCHEMA;
  static override indexes = [
    { name: "idx_test_records_name", columns: ["name"], unique: false },
  ];

  declare id: string;
  declare name: string;
  declare value: number;

  constructor(data: Row) {
    super(data);
    this.id ??= createTimeOrderedUuid();
    this.name ??= "";
    this.value ??= 0;
  }
}

// ── Setup ────────────────────────────────────────────────────────────

describe("DBModel", () => {
  const factory = new MemoryAdapterFactory();

  beforeEach(async () => {
    factory.clear();
    setGlobalAdapterResolver((schema) => factory.getAdapter(schema));
    await (TestRecord as any).createTable();
  });

  afterEach(() => {
    ModelObserver.clear();
  });

  // ── CRUD ───────────────────────────────────────────────────────────

  describe("CRUD", () => {
    it("create and retrieve a record", async () => {
      const rec = await (TestRecord as any).create({
        id: "r1",
        name: "Alpha",
        value: 42,
      });
      expect(rec).toBeInstanceOf(TestRecord);
      expect(rec.name).toBe("Alpha");

      const loaded = await (TestRecord as any).get("r1");
      expect(loaded).not.toBeNull();
      expect(loaded.value).toBe(42);
    });

    it("get returns null for missing key", async () => {
      const loaded = await (TestRecord as any).get("nonexistent");
      expect(loaded).toBeNull();
    });

    it("update modifies fields and saves", async () => {
      const rec = await (TestRecord as any).create({
        id: "r1",
        name: "Alpha",
        value: 1,
      });
      await rec.update({ value: 99 });
      const loaded = await (TestRecord as any).get("r1");
      expect(loaded.value).toBe(99);
    });

    it("delete removes the record", async () => {
      const rec = await (TestRecord as any).create({
        id: "r1",
        name: "Alpha",
        value: 1,
      });
      await rec.delete();
      const loaded = await (TestRecord as any).get("r1");
      expect(loaded).toBeNull();
    });

    it("reload refreshes from storage", async () => {
      const rec = await (TestRecord as any).create({
        id: "r1",
        name: "Alpha",
        value: 1,
      });
      // Manually update storage
      const adapter = (TestRecord as any).getAdapter();
      await adapter.save({ id: "r1", name: "Beta", value: 100 });

      await rec.reload();
      expect(rec.name).toBe("Beta");
      expect(rec.value).toBe(100);
    });
  });

  // ── Query ──────────────────────────────────────────────────────────

  describe("query", () => {
    beforeEach(async () => {
      await (TestRecord as any).create({ id: "a", name: "X", value: 10 });
      await (TestRecord as any).create({ id: "b", name: "Y", value: 20 });
      await (TestRecord as any).create({ id: "c", name: "X", value: 30 });
    });

    it("query all records", async () => {
      const [records, cursor] = await (TestRecord as any).query();
      expect(records).toHaveLength(3);
      expect(cursor).toBe("");
    });

    it("query with condition", async () => {
      const cond = field("name").equals("X");
      const [records] = await (TestRecord as any).query({ condition: cond });
      expect(records).toHaveLength(2);
    });

    it("query with limit", async () => {
      const [records, cursor] = await (TestRecord as any).query({ limit: 2 });
      expect(records).toHaveLength(2);
      expect(cursor).not.toBe("");
    });

    it("query with ordering", async () => {
      const [records] = await (TestRecord as any).query({
        orderBy: "value",
        reverse: true,
      });
      expect(records[0].value).toBe(30);
      expect(records[2].value).toBe(10);
    });
  });

  // ── Observer ───────────────────────────────────────────────────────

  describe("ModelObserver", () => {
    it("notifies on create", async () => {
      const events: ModelChangeEvent[] = [];
      ModelObserver.subscribe((_, evt) => events.push(evt), "TestRecord");

      await (TestRecord as any).create({ id: "r1", name: "A", value: 1 });
      // create calls save (UPDATED) + then CREATED
      expect(events).toContain(ModelChangeEvent.CREATED);
    });

    it("notifies on delete", async () => {
      const events: ModelChangeEvent[] = [];
      ModelObserver.subscribe((_, evt) => events.push(evt), "TestRecord");

      const rec = await (TestRecord as any).create({ id: "r1", name: "A", value: 1 });
      await rec.delete();
      expect(events).toContain(ModelChangeEvent.DELETED);
    });

    it("global observer receives all events", async () => {
      const events: ModelChangeEvent[] = [];
      ModelObserver.subscribe((_, evt) => events.push(evt));

      await (TestRecord as any).create({ id: "r1", name: "A", value: 1 });
      expect(events.length).toBeGreaterThan(0);
    });

    it("unsubscribe stops notifications", async () => {
      const events: ModelChangeEvent[] = [];
      const cb = (_: any, evt: ModelChangeEvent) => events.push(evt);
      ModelObserver.subscribe(cb, "TestRecord");
      ModelObserver.unsubscribe(cb, "TestRecord");

      await (TestRecord as any).create({ id: "r1", name: "A", value: 1 });
      // Should not have received the class-specific notification
      // (global observer may still fire if subscribed)
      expect(events.filter(e => e === ModelChangeEvent.CREATED)).toHaveLength(0);
    });
  });

  // ── Helpers ────────────────────────────────────────────────────────

  describe("helpers", () => {
    it("createTimeOrderedUuid returns 32-char hex", () => {
      const uuid = createTimeOrderedUuid();
      expect(uuid).toMatch(/^[0-9a-f]{32}$/);
    });

    it("computeEtag is deterministic", () => {
      const data = { a: 1, b: "hello" };
      expect(computeEtag(data)).toBe(computeEtag(data));
    });

    it("computeEtag changes when data changes", () => {
      const a = computeEtag({ x: 1 });
      const b = computeEtag({ x: 2 });
      expect(a).not.toBe(b);
    });

    it("getEtag on instance works", async () => {
      const rec = await (TestRecord as any).create({
        id: "r1",
        name: "A",
        value: 1,
      });
      const etag = rec.getEtag();
      expect(typeof etag).toBe("string");
      expect(etag.length).toBe(32); // MD5 hex
    });
  });

  // ── Table management ───────────────────────────────────────────────

  describe("table management", () => {
    it("dropTable clears data", async () => {
      await (TestRecord as any).create({ id: "r1", name: "A", value: 1 });
      await (TestRecord as any).dropTable();
      // Recreate table for clean state
      await (TestRecord as any).createTable();
      const [records] = await (TestRecord as any).query();
      expect(records).toHaveLength(0);
    });
  });
});
