import { vi, describe, it, expect, beforeEach, afterEach } from "vitest";
import { KieAINode, KIE_DYNAMIC_NODES } from "../src/nodes/kie-dynamic.js";

const originalFetch = globalThis.fetch;
let mockFetch: ReturnType<typeof vi.fn>;

beforeEach(() => {
  mockFetch = vi.fn();
  globalThis.fetch = mockFetch;
  delete process.env.KIE_API_KEY;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  delete process.env.KIE_API_KEY;
});

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
    arrayBuffer: async () => new Uint8Array([116, 101, 115, 116]).buffer,
  } as unknown as Response;
}

// Model IDs that map to each output type via inferOutputType:
// - "kling" -> video (video keyword)
// - "suno" -> audio (audio keyword)
// - "flux/text2image" -> image (no video/audio keyword)

// Minimal docs with Format table for model ID extraction.
// NOTE: parseInputParams uses a regex with \Z (Python-style) which does not
// work in JS, so params are not actually parsed. Tests reflect the real behavior.
const DOCS_IMAGE_MODEL = `| **Format** | \`flux/text2image\` |`;
const DOCS_VIDEO_MODEL = `| **Format** | \`kling/video-generation\` |`;
const DOCS_AUDIO_MODEL = `| **Format** | \`suno/audio-gen\` |`;

describe("KIE_DYNAMIC_NODES export", () => {
  it("exports 1 node class", () => {
    expect(KIE_DYNAMIC_NODES).toHaveLength(1);
  });
});

describe("KieAINode static metadata", () => {
  it("has correct nodeType", () => {
    expect(KieAINode.nodeType).toBe("kie.dynamic_schema.KieAI");
  });

  it("has correct title", () => {
    expect(KieAINode.title).toBe("Kie AI");
  });

  it("has correct description", () => {
    expect(KieAINode.description).toContain("Dynamic Kie.ai node");
  });

  it("defaults returns { model_info: '' }", () => {
    const node = new KieAINode();
    expect(node.defaults()).toEqual({ model_info: "" });
  });
});

describe("KieAINode validation", () => {
  it("throws on empty model_info", async () => {
    const node = new KieAINode();
    await expect(
      node.process({ model_info: "", _secrets: { KIE_API_KEY: "k" } })
    ).rejects.toThrow("model_info is empty");
  });

  it("throws on whitespace-only model_info", async () => {
    const node = new KieAINode();
    await expect(
      node.process({ model_info: "   ", _secrets: { KIE_API_KEY: "k" } })
    ).rejects.toThrow("model_info is empty");
  });

  it("throws on missing API key", async () => {
    const node = new KieAINode();
    await expect(
      node.process({ model_info: "some docs" })
    ).rejects.toThrow("KIE_API_KEY is not configured");
  });

  it("throws when model_info has no model ID", async () => {
    const node = new KieAINode();
    await expect(
      node.process({
        model_info: "Some text without model identifier",
        _secrets: { KIE_API_KEY: "k" },
      })
    ).rejects.toThrow("Could not find model ID");
  });
});

describe("KieAINode process with mocked API", () => {
  function setupSuccessfulKieApi() {
    mockFetch.mockImplementation(async (url: string | URL) => {
      const urlStr = String(url);
      if (urlStr.includes("createTask")) {
        return jsonResponse({ code: 200, data: { taskId: "task_dyn_1" } });
      }
      if (urlStr.includes("recordInfo")) {
        return jsonResponse({
          code: 200,
          data: {
            state: "success",
            resultJson: JSON.stringify({
              resultUrls: ["https://cdn.example.com/output.png"],
            }),
          },
        });
      }
      if (urlStr.includes("cdn.example.com")) {
        return {
          ok: true,
          status: 200,
          json: async () => null,
          text: async () => "",
          arrayBuffer: async () => new Uint8Array([1, 2, 3]).buffer,
        } as unknown as Response;
      }
      return jsonResponse({ error: "unknown" }, 404);
    });
  }

  it("processes image model and returns image output", async () => {
    setupSuccessfulKieApi();
    const node = new KieAINode();
    const result = await node.process({
      model_info: DOCS_IMAGE_MODEL,
      _secrets: { KIE_API_KEY: "test-key" },
    });
    expect(result.image).toBeDefined();
    expect((result.image as { data: string }).data).toBeTruthy();
  });

  it("processes video model and returns video output", async () => {
    setupSuccessfulKieApi();
    const node = new KieAINode();
    const result = await node.process({
      model_info: DOCS_VIDEO_MODEL,
      _secrets: { KIE_API_KEY: "test-key" },
    });
    expect(result.video).toBeDefined();
    expect((result.video as { data: string }).data).toBeTruthy();
  });

  it("processes audio model and returns audio output", async () => {
    setupSuccessfulKieApi();
    const node = new KieAINode();
    const result = await node.process({
      model_info: DOCS_AUDIO_MODEL,
      _secrets: { KIE_API_KEY: "test-key" },
    });
    expect(result.audio).toBeDefined();
    expect((result.audio as { data: string }).data).toBeTruthy();
  });

  it("sends correct model ID to createTask", async () => {
    setupSuccessfulKieApi();
    const node = new KieAINode();
    await node.process({
      model_info: DOCS_IMAGE_MODEL,
      _secrets: { KIE_API_KEY: "test-key" },
    });
    const createCall = mockFetch.mock.calls.find((c: unknown[]) =>
      String(c[0]).includes("createTask")
    );
    expect(createCall).toBeDefined();
    const body = JSON.parse(createCall![1].body);
    expect(body.model).toBe("flux/text2image");
  });

  it("uses API key from env when _secrets not provided", async () => {
    setupSuccessfulKieApi();
    process.env.KIE_API_KEY = "env-api-key";
    const node = new KieAINode();
    const result = await node.process({
      model_info: DOCS_IMAGE_MODEL,
    });
    expect(result.image).toBeDefined();
  });

  it("uses API key from _secrets over env", async () => {
    setupSuccessfulKieApi();
    process.env.KIE_API_KEY = "env-key";
    const node = new KieAINode();
    const result = await node.process({
      model_info: DOCS_IMAGE_MODEL,
      _secrets: { KIE_API_KEY: "secrets-key" },
    });
    expect(result.image).toBeDefined();
    // Verify the Authorization header uses secrets-key
    const createCall = mockFetch.mock.calls.find((c: unknown[]) =>
      String(c[0]).includes("createTask")
    );
    expect(createCall![1].headers.Authorization).toBe("Bearer secrets-key");
  });

  it("result data is base64 encoded", async () => {
    setupSuccessfulKieApi();
    const node = new KieAINode();
    const result = await node.process({
      model_info: DOCS_IMAGE_MODEL,
      _secrets: { KIE_API_KEY: "k" },
    });
    const data = (result.image as { data: string }).data;
    // [1, 2, 3] -> base64 "AQID"
    expect(data).toBe("AQID");
  });

  it("model ID extracted from 'model' JSON field", async () => {
    setupSuccessfulKieApi();
    const node = new KieAINode();
    const result = await node.process({
      model_info: `"model": "dalle/text2image"`,
      _secrets: { KIE_API_KEY: "k" },
    });
    // "dalle" doesn't match video/audio keywords -> image
    expect(result.image).toBeDefined();
    const createCall = mockFetch.mock.calls.find((c: unknown[]) =>
      String(c[0]).includes("createTask")
    );
    const body = JSON.parse(createCall![1].body);
    expect(body.model).toBe("dalle/text2image");
  });
});

describe("KieAINode API error handling", () => {
  it("throws on API submit error", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ code: 401, message: "Unauthorized" })
    );
    const node = new KieAINode();
    await expect(
      node.process({
        model_info: DOCS_IMAGE_MODEL,
        _secrets: { KIE_API_KEY: "bad-key" },
      })
    ).rejects.toThrow("Unauthorized");
  });

  it("throws on task failure", async () => {
    mockFetch.mockImplementation(async (url: string | URL) => {
      const urlStr = String(url);
      if (urlStr.includes("createTask")) {
        return jsonResponse({ code: 200, data: { taskId: "task_fail" } });
      }
      if (urlStr.includes("recordInfo")) {
        return jsonResponse({
          code: 200,
          data: { state: "failed", failMsg: "Generation error" },
        });
      }
      return jsonResponse({}, 404);
    });
    const node = new KieAINode();
    await expect(
      node.process({
        model_info: DOCS_IMAGE_MODEL,
        _secrets: { KIE_API_KEY: "k" },
      })
    ).rejects.toThrow("Task failed");
  });

  it("throws on submit HTTP error", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ data: {} }, 500)
    );
    const node = new KieAINode();
    await expect(
      node.process({
        model_info: DOCS_IMAGE_MODEL,
        _secrets: { KIE_API_KEY: "k" },
      })
    ).rejects.toThrow("Submit failed");
  });

  it("throws when no taskId returned", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ code: 200, data: {} })
    );
    const node = new KieAINode();
    await expect(
      node.process({
        model_info: DOCS_IMAGE_MODEL,
        _secrets: { KIE_API_KEY: "k" },
      })
    ).rejects.toThrow("No taskId");
  });
});
