/**
 * Additional coverage tests for OllamaProvider – message conversion
 * edge cases, image handling, embedding errors, streaming errors.
 */

import { describe, it, expect, vi } from "vitest";
import { OllamaProvider } from "../../src/providers/ollama-provider.js";
import type { Message } from "../../src/providers/types.js";

function jsonResponse(payload: unknown, ok = true, status = 200): Response {
  return {
    ok,
    status,
    json: async () => payload,
    body: null,
  } as unknown as Response;
}

function streamResponse(lines: unknown[]): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const line of lines) {
        controller.enqueue(encoder.encode(`${JSON.stringify(line)}\n`));
      }
      controller.close();
    },
  });

  return {
    ok: true,
    status: 200,
    body,
    json: async () => ({}),
  } as unknown as Response;
}

describe("OllamaProvider – convertMessage edge cases", () => {
  const provider = new OllamaProvider(
    { OLLAMA_API_URL: "http://localhost:11434" },
    { fetchFn: vi.fn() as unknown as typeof fetch }
  );

  it("converts tool message", async () => {
    const result = await provider.convertMessage({
      role: "tool",
      content: "tool result",
    });
    expect(result).toEqual({ role: "tool", content: "tool result" });
  });

  it("converts tool message with object content", async () => {
    const result = await provider.convertMessage({
      role: "tool",
      content: { ok: true } as any,
    });
    expect(result).toEqual({ role: "tool", content: '{"ok":true}' });
  });

  it("converts assistant message with tool calls", async () => {
    const result = await provider.convertMessage({
      role: "assistant",
      content: "calling",
      toolCalls: [{ id: "tc1", name: "search", args: { q: "x" } }],
    });
    expect(result).toEqual({
      role: "assistant",
      content: "calling",
      tool_calls: [
        { function: { name: "search", arguments: { q: "x" } } },
      ],
    });
  });

  it("converts system message with string", async () => {
    const result = await provider.convertMessage({
      role: "system",
      content: "You are helpful",
    });
    expect(result).toEqual({ role: "system", content: "You are helpful" });
  });

  it("converts system message with array content", async () => {
    const result = await provider.convertMessage({
      role: "system",
      content: [
        { type: "text", text: "part1" },
        { type: "text", text: "part2" },
      ],
    });
    expect(result).toEqual({ role: "system", content: "part1\npart2" });
  });

  it("converts system message with null content", async () => {
    const result = await provider.convertMessage({
      role: "system",
      content: null,
    });
    expect(result).toEqual({ role: "system", content: "" });
  });

  it("throws for unsupported role", async () => {
    await expect(
      provider.convertMessage({ role: "unknown" as any, content: "x" })
    ).rejects.toThrow("Unsupported message role");
  });

  it("converts user string message", async () => {
    const result = await provider.convertMessage({
      role: "user",
      content: "hello",
    });
    expect(result).toEqual({ role: "user", content: "hello" });
  });

  it("converts user multipart text-only (no images)", async () => {
    const result = await provider.convertMessage({
      role: "user",
      content: [{ type: "text", text: "just text" }],
    });
    expect(result).toEqual({ role: "user", content: "just text" });
    expect((result as any).images).toBeUndefined();
  });
});

describe("OllamaProvider – imageToBase64 paths", () => {
  it("handles image with data URI string in data field", async () => {
    const base64 = Buffer.from("test").toString("base64");
    const provider = new OllamaProvider(
      { OLLAMA_API_URL: "http://localhost:11434" },
      { fetchFn: vi.fn() as unknown as typeof fetch }
    );

    const result = await provider.convertMessage({
      role: "user",
      content: [
        {
          type: "image",
          image: { data: `data:image/png;base64,${base64}` },
        },
      ],
    });
    expect((result as any).images[0]).toBe(base64);
  });

  it("handles image with data URI in uri field", async () => {
    const base64 = Buffer.from("test").toString("base64");
    const provider = new OllamaProvider(
      { OLLAMA_API_URL: "http://localhost:11434" },
      { fetchFn: vi.fn() as unknown as typeof fetch }
    );

    const result = await provider.convertMessage({
      role: "user",
      content: [
        {
          type: "image",
          image: { uri: `data:image/png;base64,${base64}` },
        },
      ],
    });
    expect((result as any).images[0]).toBe(base64);
  });

  it("handles image with plain base64 string data", async () => {
    const provider = new OllamaProvider(
      { OLLAMA_API_URL: "http://localhost:11434" },
      { fetchFn: vi.fn() as unknown as typeof fetch }
    );

    const result = await provider.convertMessage({
      role: "user",
      content: [
        { type: "image", image: { data: "AQID" } },
      ],
    });
    expect((result as any).images[0]).toBe("AQID");
  });

  it("fetches image from remote URI", async () => {
    const fetchFn = vi.fn().mockImplementation(async (url: string) => {
      if (url.includes("example.com")) {
        return {
          ok: true,
          arrayBuffer: async () => Uint8Array.from([1, 2, 3]).buffer,
        };
      }
      return jsonResponse({});
    });

    const provider = new OllamaProvider(
      { OLLAMA_API_URL: "http://localhost:11434" },
      { fetchFn: fetchFn as unknown as typeof fetch }
    );

    const result = await provider.convertMessage({
      role: "user",
      content: [
        { type: "image", image: { uri: "https://example.com/img.png" } },
      ],
    });
    expect((result as any).images[0]).toBeTruthy();
  });

  it("throws on fetch failure for image URI", async () => {
    const fetchFn = vi.fn().mockResolvedValue({ ok: false, status: 404 });

    const provider = new OllamaProvider(
      { OLLAMA_API_URL: "http://localhost:11434" },
      { fetchFn: fetchFn as unknown as typeof fetch }
    );

    await expect(
      provider.convertMessage({
        role: "user",
        content: [
          { type: "image", image: { uri: "https://example.com/missing.png" } },
        ],
      })
    ).rejects.toThrow("Failed to fetch image URI");
  });

  it("throws for image with no data and no uri", async () => {
    const provider = new OllamaProvider(
      { OLLAMA_API_URL: "http://localhost:11434" },
      { fetchFn: vi.fn() as unknown as typeof fetch }
    );

    await expect(
      provider.convertMessage({
        role: "user",
        content: [{ type: "image", image: {} }],
      })
    ).rejects.toThrow("Invalid image payload");
  });
});

describe("OllamaProvider – formatTools", () => {
  it("formats tools in function format", () => {
    const provider = new OllamaProvider(
      { OLLAMA_API_URL: "http://localhost:11434" },
      { fetchFn: vi.fn() as unknown as typeof fetch }
    );

    const result = provider.formatTools([
      {
        name: "search",
        description: "Search",
        inputSchema: { type: "object", properties: {} },
      },
    ]);

    expect(result).toEqual([
      {
        type: "function",
        function: {
          name: "search",
          description: "Search",
          parameters: { type: "object", properties: {} },
        },
      },
    ]);
  });
});

describe("OllamaProvider – generateMessage edge cases", () => {
  it("throws on jsonSchema", async () => {
    const provider = new OllamaProvider(
      { OLLAMA_API_URL: "http://localhost:11434" },
      { fetchFn: vi.fn() as unknown as typeof fetch }
    );

    await expect(
      provider.generateMessage({
        model: "test",
        messages: [{ role: "user", content: "hi" }],
        jsonSchema: { type: "object" },
      })
    ).rejects.toThrow("jsonSchema is not supported");
  });

  it("handles json_schema response format", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      jsonResponse({
        message: { content: '{"result": 42}' },
      })
    );

    const provider = new OllamaProvider(
      { OLLAMA_API_URL: "http://localhost:11434" },
      { fetchFn: fetchFn as unknown as typeof fetch }
    );

    const result = await provider.generateMessage({
      model: "test",
      messages: [{ role: "user", content: "json" }],
      responseFormat: {
        type: "json_schema",
        json_schema: {
          schema: { type: "object", properties: { result: { type: "number" } } },
        },
      },
    });

    expect(result.content).toBe('{"result": 42}');
    // Verify the format was set in request body
    const body = JSON.parse(fetchFn.mock.calls[0][1].body);
    expect(body.format).toBeDefined();
  });

  it("throws when json_schema has no schema", async () => {
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse({}));
    const provider = new OllamaProvider(
      { OLLAMA_API_URL: "http://localhost:11434" },
      { fetchFn: fetchFn as unknown as typeof fetch }
    );

    await expect(
      provider.generateMessage({
        model: "test",
        messages: [{ role: "user", content: "json" }],
        responseFormat: { type: "json_schema", json_schema: {} },
      })
    ).rejects.toThrow("schema is required");
  });

  it("handles tool call with string arguments", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      jsonResponse({
        message: {
          content: "",
          tool_calls: [
            { function: { name: "calc", arguments: '{"x": 1}' } },
          ],
        },
      })
    );

    const provider = new OllamaProvider(
      { OLLAMA_API_URL: "http://localhost:11434" },
      { fetchFn: fetchFn as unknown as typeof fetch }
    );

    const result = await provider.generateMessage({
      model: "test",
      messages: [{ role: "user", content: "calc" }],
      tools: [{ name: "calc" }],
    });

    expect(result.toolCalls).toEqual([
      { id: "tool_1", name: "calc", args: { x: 1 } },
    ]);
  });

  it("handles tool call with empty/missing function name", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      jsonResponse({
        message: {
          content: "ok",
          tool_calls: [{ function: {} }],
        },
      })
    );

    const provider = new OllamaProvider(
      { OLLAMA_API_URL: "http://localhost:11434" },
      { fetchFn: fetchFn as unknown as typeof fetch }
    );

    const result = await provider.generateMessage({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
    });

    // Tool call with empty name should be filtered out
    expect(result.toolCalls).toEqual([]);
  });
});

describe("OllamaProvider – streaming edge cases", () => {
  it("throws on jsonSchema in streaming", async () => {
    const provider = new OllamaProvider(
      { OLLAMA_API_URL: "http://localhost:11434" },
      { fetchFn: vi.fn() as unknown as typeof fetch }
    );

    const gen = provider.generateMessages({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
      jsonSchema: { type: "object" },
    });

    await expect(gen.next()).rejects.toThrow("jsonSchema is not supported");
  });

  it("throws on failed streaming response", async () => {
    const fetchFn = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      body: null,
    });

    const provider = new OllamaProvider(
      { OLLAMA_API_URL: "http://localhost:11434" },
      { fetchFn: fetchFn as unknown as typeof fetch }
    );

    const gen = provider.generateMessages({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
    });

    await expect(gen.next()).rejects.toThrow("API request failed");
  });
});

describe("OllamaProvider – getAvailableEmbeddingModels", () => {
  it("returns models derived from language models", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      jsonResponse({
        models: [{ model: "nomic-embed-text" }],
      })
    );

    const provider = new OllamaProvider(
      { OLLAMA_API_URL: "http://localhost:11434" },
      { fetchFn: fetchFn as unknown as typeof fetch }
    );

    const models = await provider.getAvailableEmbeddingModels();
    expect(models).toEqual([
      {
        id: "nomic-embed-text",
        name: "nomic-embed-text",
        provider: "ollama",
        dimensions: 0,
      },
    ]);
  });
});

describe("OllamaProvider – generateEmbedding errors", () => {
  it("throws on empty text", async () => {
    const provider = new OllamaProvider(
      { OLLAMA_API_URL: "http://localhost:11434" },
      { fetchFn: vi.fn() as unknown as typeof fetch }
    );

    await expect(
      provider.generateEmbedding({ text: "", model: "test" })
    ).rejects.toThrow("text must not be empty");
  });

  it("throws on empty array", async () => {
    const provider = new OllamaProvider(
      { OLLAMA_API_URL: "http://localhost:11434" },
      { fetchFn: vi.fn() as unknown as typeof fetch }
    );

    await expect(
      provider.generateEmbedding({ text: [], model: "test" })
    ).rejects.toThrow("text must not be empty");
  });
});

describe("OllamaProvider – hasToolSupport", () => {
  it("returns true for all models", () => {
    const provider = new OllamaProvider(
      { OLLAMA_API_URL: "http://localhost:11434" },
      { fetchFn: vi.fn() as unknown as typeof fetch }
    );
    expect(provider.hasToolSupport("any-model")).toBe(true);
  });
});

describe("OllamaProvider – parseDataUri non-base64 path", () => {
  it("converts image with non-base64 data URI in data field", async () => {
    const provider = new OllamaProvider(
      { OLLAMA_API_URL: "http://localhost:11434" },
      { fetchFn: vi.fn() as unknown as typeof fetch }
    );

    const result = await provider.convertMessage({
      role: "user",
      content: [
        {
          type: "image",
          image: { data: "data:image/png,hello%20world" },
        },
      ],
    });
    expect((result as any).images[0]).toBeTruthy();
  });

  it("converts image with non-base64 data URI in uri field", async () => {
    const provider = new OllamaProvider(
      { OLLAMA_API_URL: "http://localhost:11434" },
      { fetchFn: vi.fn() as unknown as typeof fetch }
    );

    const result = await provider.convertMessage({
      role: "user",
      content: [
        {
          type: "image",
          image: { uri: "data:image/png,hello%20world" },
        },
      ],
    });
    expect((result as any).images[0]).toBeTruthy();
  });
});

describe("OllamaProvider – normalizeToolArgs edge cases", () => {
  it("handles string JSON array arguments as empty object", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      jsonResponse({
        message: {
          content: "",
          tool_calls: [
            { function: { name: "calc", arguments: "[1,2,3]" } },
          ],
        },
      })
    );

    const provider = new OllamaProvider(
      { OLLAMA_API_URL: "http://localhost:11434" },
      { fetchFn: fetchFn as unknown as typeof fetch }
    );

    const result = await provider.generateMessage({
      model: "test",
      messages: [{ role: "user", content: "calc" }],
      tools: [{ name: "calc" }],
    });

    expect(result.toolCalls![0].args).toEqual({});
  });

  it("handles invalid JSON string arguments as empty object", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      jsonResponse({
        message: {
          content: "",
          tool_calls: [
            { function: { name: "calc", arguments: "not json" } },
          ],
        },
      })
    );

    const provider = new OllamaProvider(
      { OLLAMA_API_URL: "http://localhost:11434" },
      { fetchFn: fetchFn as unknown as typeof fetch }
    );

    const result = await provider.generateMessage({
      model: "test",
      messages: [{ role: "user", content: "calc" }],
      tools: [{ name: "calc" }],
    });

    expect(result.toolCalls![0].args).toEqual({});
  });

  it("handles array arguments as empty object", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      jsonResponse({
        message: {
          content: "",
          tool_calls: [
            { function: { name: "calc", arguments: [1, 2, 3] } },
          ],
        },
      })
    );

    const provider = new OllamaProvider(
      { OLLAMA_API_URL: "http://localhost:11434" },
      { fetchFn: fetchFn as unknown as typeof fetch }
    );

    const result = await provider.generateMessage({
      model: "test",
      messages: [{ role: "user", content: "calc" }],
      tools: [{ name: "calc" }],
    });

    expect(result.toolCalls![0].args).toEqual({});
  });
});

describe("OllamaProvider – postJson error path", () => {
  it("throws when postJson receives non-OK response", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      jsonResponse({}, false, 500)
    );

    const provider = new OllamaProvider(
      { OLLAMA_API_URL: "http://localhost:11434" },
      { fetchFn: fetchFn as unknown as typeof fetch }
    );

    await expect(
      provider.generateEmbedding({ text: "hello", model: "test" })
    ).rejects.toThrow("API request failed");
  });
});
