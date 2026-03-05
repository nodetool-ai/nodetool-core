import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { setGlobalAdapterResolver, ModelObserver } from "../src/base-model.js";
import { MemoryAdapterFactory } from "../src/memory-adapter.js";
import { OAuthCredential } from "../src/oauth-credential.js";
import type { ModelClass } from "../src/base-model.js";

const factory = new MemoryAdapterFactory();

async function setup() {
  factory.clear();
  setGlobalAdapterResolver((schema) => factory.getAdapter(schema));
  await (OAuthCredential as unknown as ModelClass).createTable();
}

describe("OAuthCredential model", () => {
  beforeEach(setup);
  afterEach(() => ModelObserver.clear());

  it("creates with defaults", async () => {
    const cred = await (OAuthCredential as unknown as ModelClass<OAuthCredential>).create({
      user_id: "u1",
      provider: "google",
      account_id: "acc1",
      access_token: "tok123",
      token_type: "Bearer",
      received_at: new Date().toISOString(),
    });
    expect(cred.id).toBeTruthy();
    expect(cred.user_id).toBe("u1");
    expect(cred.provider).toBe("google");
    expect(cred.account_id).toBe("acc1");
    expect(cred.access_token).toBe("tok123");
    expect(cred.token_type).toBe("Bearer");
    expect(cred.refresh_token).toBeNull();
    expect(cred.username).toBeNull();
    expect(cred.scope).toBeNull();
    expect(cred.expires_at).toBeNull();
    expect(cred.created_at).toBeTruthy();
    expect(cred.updated_at).toBeTruthy();
  });

  it("constructor applies defaults for optional fields", () => {
    const cred = new OAuthCredential({
      user_id: "u1",
      provider: "github",
      account_id: "acc2",
      access_token: "tok456",
    });
    expect(cred.id).toBeTruthy();
    expect(cred.token_type).toBe("Bearer");
    expect(cred.refresh_token).toBeNull();
    expect(cred.username).toBeNull();
    expect(cred.scope).toBeNull();
    expect(cred.expires_at).toBeNull();
    expect(cred.received_at).toBeTruthy();
    expect(cred.created_at).toBeTruthy();
    expect(cred.updated_at).toBeTruthy();
  });

  it("beforeSave updates updated_at", async () => {
    const cred = await (OAuthCredential as unknown as ModelClass<OAuthCredential>).create({
      user_id: "u1",
      provider: "google",
      account_id: "acc1",
      access_token: "tok123",
      token_type: "Bearer",
      received_at: new Date().toISOString(),
    });
    const originalUpdatedAt = cred.updated_at;
    // Wait a tick so timestamp changes
    await new Promise((r) => setTimeout(r, 5));
    cred.access_token = "new-token";
    await cred.save();
    expect(cred.updated_at).not.toBe(originalUpdatedAt);
  });

  it("findByAccount returns matching credential", async () => {
    await (OAuthCredential as unknown as ModelClass<OAuthCredential>).create({
      user_id: "u1",
      provider: "google",
      account_id: "acc1",
      access_token: "tok-a",
      token_type: "Bearer",
      received_at: new Date().toISOString(),
    });

    const found = await OAuthCredential.findByAccount("u1", "google", "acc1");
    expect(found).not.toBeNull();
    expect(found!.access_token).toBe("tok-a");
  });

  it("findByAccount returns null when not found", async () => {
    const found = await OAuthCredential.findByAccount("u1", "google", "nonexistent");
    expect(found).toBeNull();
  });

  it("findByAccount scoped to user/provider/account", async () => {
    await (OAuthCredential as unknown as ModelClass<OAuthCredential>).create({
      user_id: "u1",
      provider: "google",
      account_id: "acc1",
      access_token: "tok-u1",
      token_type: "Bearer",
      received_at: new Date().toISOString(),
    });

    // Different user
    const r1 = await OAuthCredential.findByAccount("u2", "google", "acc1");
    expect(r1).toBeNull();

    // Different provider
    const r2 = await OAuthCredential.findByAccount("u1", "github", "acc1");
    expect(r2).toBeNull();

    // Different account
    const r3 = await OAuthCredential.findByAccount("u1", "google", "acc2");
    expect(r3).toBeNull();
  });

  it("listForUserAndProvider returns all matching credentials", async () => {
    await (OAuthCredential as unknown as ModelClass<OAuthCredential>).create({
      user_id: "u1",
      provider: "google",
      account_id: "acc1",
      access_token: "tok1",
      token_type: "Bearer",
      received_at: new Date().toISOString(),
    });
    await (OAuthCredential as unknown as ModelClass<OAuthCredential>).create({
      user_id: "u1",
      provider: "google",
      account_id: "acc2",
      access_token: "tok2",
      token_type: "Bearer",
      received_at: new Date().toISOString(),
    });
    await (OAuthCredential as unknown as ModelClass<OAuthCredential>).create({
      user_id: "u1",
      provider: "github",
      account_id: "acc3",
      access_token: "tok3",
      token_type: "Bearer",
      received_at: new Date().toISOString(),
    });
    await (OAuthCredential as unknown as ModelClass<OAuthCredential>).create({
      user_id: "u2",
      provider: "google",
      account_id: "acc4",
      access_token: "tok4",
      token_type: "Bearer",
      received_at: new Date().toISOString(),
    });

    const results = await OAuthCredential.listForUserAndProvider("u1", "google");
    expect(results).toHaveLength(2);
    const accounts = results.map((r) => r.account_id).sort();
    expect(accounts).toEqual(["acc1", "acc2"]);
  });

  it("listForUserAndProvider returns empty array when no matches", async () => {
    const results = await OAuthCredential.listForUserAndProvider("u1", "google");
    expect(results).toEqual([]);
  });

  it("upsert creates a new credential", async () => {
    const cred = await OAuthCredential.upsert({
      user_id: "u1",
      provider: "google",
      account_id: "acc1",
      access_token: "tok-new",
      token_type: "Bearer",
      received_at: new Date().toISOString(),
    });

    expect(cred.id).toBeTruthy();
    expect(cred.access_token).toBe("tok-new");
  });

  it("upsert updates an existing credential", async () => {
    const created = await OAuthCredential.upsert({
      user_id: "u1",
      provider: "google",
      account_id: "acc1",
      access_token: "tok-original",
      token_type: "Bearer",
      received_at: "2024-01-01T00:00:00.000Z",
    });
    const createdId = created.id;

    const updated = await OAuthCredential.upsert({
      user_id: "u1",
      provider: "google",
      account_id: "acc1",
      access_token: "tok-updated",
      token_type: "Bearer",
      received_at: "2024-06-01T00:00:00.000Z",
      refresh_token: "refresh-new",
      username: "user@example.com",
      scope: "read write",
      expires_at: "2025-01-01T00:00:00.000Z",
    });

    expect(updated.id).toBe(createdId);
    expect(updated.access_token).toBe("tok-updated");
    expect(updated.refresh_token).toBe("refresh-new");
    expect(updated.username).toBe("user@example.com");
    expect(updated.scope).toBe("read write");
    expect(updated.received_at).toBe("2024-06-01T00:00:00.000Z");
    expect(updated.expires_at).toBe("2025-01-01T00:00:00.000Z");
  });

  it("upsert preserves existing optional fields if not provided in update", async () => {
    await OAuthCredential.upsert({
      user_id: "u1",
      provider: "google",
      account_id: "acc1",
      access_token: "tok-original",
      token_type: "Bearer",
      received_at: "2024-01-01T00:00:00.000Z",
      refresh_token: "existing-refresh",
      username: "existing-user",
      scope: "existing-scope",
      expires_at: "2025-12-31T00:00:00.000Z",
    });

    const updated = await OAuthCredential.upsert({
      user_id: "u1",
      provider: "google",
      account_id: "acc1",
      access_token: "tok-updated",
      token_type: "Bearer",
      received_at: "2024-06-01T00:00:00.000Z",
      // not providing refresh_token, username, scope, expires_at
    });

    expect(updated.refresh_token).toBe("existing-refresh");
    expect(updated.username).toBe("existing-user");
    expect(updated.scope).toBe("existing-scope");
    expect(updated.expires_at).toBe("2025-12-31T00:00:00.000Z");
  });

  it("delete removes a credential", async () => {
    const cred = await (OAuthCredential as unknown as ModelClass<OAuthCredential>).create({
      user_id: "u1",
      provider: "google",
      account_id: "acc1",
      access_token: "tok1",
      token_type: "Bearer",
      received_at: new Date().toISOString(),
    });

    await cred.delete();

    const found = await OAuthCredential.findByAccount("u1", "google", "acc1");
    expect(found).toBeNull();
  });

  it("get retrieves by primary key", async () => {
    const cred = await (OAuthCredential as unknown as ModelClass<OAuthCredential>).create({
      user_id: "u1",
      provider: "google",
      account_id: "acc1",
      access_token: "tok1",
      token_type: "Bearer",
      received_at: new Date().toISOString(),
    });

    const found = await (OAuthCredential as unknown as ModelClass<OAuthCredential>).get(cred.id);
    expect(found).not.toBeNull();
    expect((found as OAuthCredential).provider).toBe("google");
  });
});
