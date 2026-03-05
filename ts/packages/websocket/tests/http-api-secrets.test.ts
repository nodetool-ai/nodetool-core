import { describe, it, expect, beforeEach } from "vitest";
import {
  MemoryAdapterFactory,
  setGlobalAdapterResolver,
  Secret,
} from "@nodetool/models";
import { setMasterKey } from "@nodetool/security";
import { handleApiRequest } from "../src/http-api.js";

const TEST_MASTER_KEY = "dGVzdC1tYXN0ZXIta2V5LWZvci11bml0LXRlc3Rz";

async function jsonBody(response: Response): Promise<unknown> {
  const text = await response.text();
  return text ? JSON.parse(text) : null;
}

function makeRequest(
  method: string,
  url: string,
  body?: unknown,
  userId = "user-1"
): Request {
  const init: RequestInit = {
    method,
    headers: {
      "content-type": "application/json",
      "x-user-id": userId,
    },
  };
  if (body !== undefined) {
    init.body = JSON.stringify(body);
  }
  return new Request(`http://localhost${url}`, init);
}

describe("HTTP API: settings/secrets", () => {
  beforeEach(async () => {
    const factory = new MemoryAdapterFactory();
    setGlobalAdapterResolver((schema) => factory.getAdapter(schema));
    setMasterKey(TEST_MASTER_KEY);
    await Secret.createTable();
  });

  it("GET /api/settings/secrets returns empty list initially", async () => {
    const res = await handleApiRequest(makeRequest("GET", "/api/settings/secrets"));
    expect(res.status).toBe(200);

    const data = (await jsonBody(res)) as { secrets: unknown[]; next_key: unknown };
    expect(data.secrets).toEqual([]);
    expect(data.next_key).toBeNull();
  });

  it("PUT /api/settings/secrets/:key creates a secret", async () => {
    const res = await handleApiRequest(
      makeRequest("PUT", "/api/settings/secrets/OPENAI_API_KEY", {
        value: "sk-test-123",
        description: "My OpenAI key",
      })
    );
    expect(res.status).toBe(200);

    const data = (await jsonBody(res)) as Record<string, unknown>;
    expect(data.key).toBe("OPENAI_API_KEY");
    expect(data.description).toBe("My OpenAI key");
    expect(data.is_configured).toBe(true);
    expect(data.user_id).toBe("user-1");
    expect(data).not.toHaveProperty("encrypted_value");
    expect(data).not.toHaveProperty("value");
  });

  it("GET /api/settings/secrets lists created secrets", async () => {
    await handleApiRequest(
      makeRequest("PUT", "/api/settings/secrets/KEY_A", { value: "val-a" })
    );
    await handleApiRequest(
      makeRequest("PUT", "/api/settings/secrets/KEY_B", { value: "val-b" })
    );

    const res = await handleApiRequest(makeRequest("GET", "/api/settings/secrets"));
    expect(res.status).toBe(200);

    const data = (await jsonBody(res)) as { secrets: Array<Record<string, unknown>> };
    expect(data.secrets.length).toBe(2);

    const keys = data.secrets.map((s) => s.key).sort();
    expect(keys).toEqual(["KEY_A", "KEY_B"]);

    // Should not include encrypted values
    for (const s of data.secrets) {
      expect(s).not.toHaveProperty("encrypted_value");
      expect(s).not.toHaveProperty("value");
      expect(s.is_configured).toBe(true);
    }
  });

  it("GET /api/settings/secrets/:key returns secret metadata", async () => {
    await handleApiRequest(
      makeRequest("PUT", "/api/settings/secrets/MY_KEY", {
        value: "my-secret",
        description: "desc",
      })
    );

    const res = await handleApiRequest(
      makeRequest("GET", "/api/settings/secrets/MY_KEY")
    );
    expect(res.status).toBe(200);

    const data = (await jsonBody(res)) as Record<string, unknown>;
    expect(data.key).toBe("MY_KEY");
    expect(data.description).toBe("desc");
    expect(data.is_configured).toBe(true);
    expect(data).not.toHaveProperty("value");
  });

  it("GET /api/settings/secrets/:key?decrypt=true returns decrypted value", async () => {
    await handleApiRequest(
      makeRequest("PUT", "/api/settings/secrets/DECRYPT_KEY", {
        value: "super-secret-value",
      })
    );

    const res = await handleApiRequest(
      makeRequest("GET", "/api/settings/secrets/DECRYPT_KEY?decrypt=true")
    );
    expect(res.status).toBe(200);

    const data = (await jsonBody(res)) as Record<string, unknown>;
    expect(data.key).toBe("DECRYPT_KEY");
    expect(data.value).toBe("super-secret-value");
  });

  it("GET /api/settings/secrets/:key returns 404 when not found", async () => {
    const res = await handleApiRequest(
      makeRequest("GET", "/api/settings/secrets/NONEXISTENT")
    );
    expect(res.status).toBe(404);
  });

  it("PUT /api/settings/secrets/:key updates an existing secret", async () => {
    await handleApiRequest(
      makeRequest("PUT", "/api/settings/secrets/UPD_KEY", {
        value: "original",
        description: "v1",
      })
    );

    const res = await handleApiRequest(
      makeRequest("PUT", "/api/settings/secrets/UPD_KEY", {
        value: "updated",
        description: "v2",
      })
    );
    expect(res.status).toBe(200);

    const data = (await jsonBody(res)) as Record<string, unknown>;
    expect(data.description).toBe("v2");

    // Verify updated value
    const getRes = await handleApiRequest(
      makeRequest("GET", "/api/settings/secrets/UPD_KEY?decrypt=true")
    );
    const getData = (await jsonBody(getRes)) as Record<string, unknown>;
    expect(getData.value).toBe("updated");
  });

  it("PUT /api/settings/secrets/:key returns 400 for missing value", async () => {
    const res = await handleApiRequest(
      makeRequest("PUT", "/api/settings/secrets/BAD_KEY", {
        description: "no value",
      })
    );
    expect(res.status).toBe(400);
  });

  it("DELETE /api/settings/secrets/:key deletes a secret", async () => {
    await handleApiRequest(
      makeRequest("PUT", "/api/settings/secrets/DEL_KEY", { value: "to-delete" })
    );

    const res = await handleApiRequest(
      makeRequest("DELETE", "/api/settings/secrets/DEL_KEY")
    );
    expect(res.status).toBe(200);

    const data = (await jsonBody(res)) as Record<string, unknown>;
    expect(data.message).toBe("Secret deleted successfully");

    // Verify it's gone
    const getRes = await handleApiRequest(
      makeRequest("GET", "/api/settings/secrets/DEL_KEY")
    );
    expect(getRes.status).toBe(404);
  });

  it("DELETE /api/settings/secrets/:key returns 404 when not found", async () => {
    const res = await handleApiRequest(
      makeRequest("DELETE", "/api/settings/secrets/NONEXISTENT")
    );
    expect(res.status).toBe(404);
  });

  it("isolates secrets between users", async () => {
    await handleApiRequest(
      makeRequest("PUT", "/api/settings/secrets/SHARED", { value: "user1-val" }, "user-1")
    );
    await handleApiRequest(
      makeRequest("PUT", "/api/settings/secrets/SHARED", { value: "user2-val" }, "user-2")
    );

    const res1 = await handleApiRequest(
      makeRequest("GET", "/api/settings/secrets/SHARED?decrypt=true", undefined, "user-1")
    );
    const data1 = (await jsonBody(res1)) as Record<string, unknown>;
    expect(data1.value).toBe("user1-val");

    const res2 = await handleApiRequest(
      makeRequest("GET", "/api/settings/secrets/SHARED?decrypt=true", undefined, "user-2")
    );
    const data2 = (await jsonBody(res2)) as Record<string, unknown>;
    expect(data2.value).toBe("user2-val");
  });
});
