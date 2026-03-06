/**
 * Additional coverage tests for GeminiProvider – image conversion,
 * response format, error handling, edge cases.
 */

import { describe, it, expect, vi } from "vitest";
import { GeminiProvider } from "../../src/providers/gemini-provider.js";
import type { Message } from "../../src/providers/types.js";

function makeFetchResponse(body: unknown, ok = true, status = 200): Response {
  return {
    ok,
    status,
    headers: new Headers(),
    json: async () => body,
    text: async () => JSON.stringify(body),
    body: null,
  } as unknown as Response;
}

function makeSSEStream(events: unknown[]): Response {
  const lines = events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join("");
  const encoder = new TextEncoder();
  const bytes = encoder.encode(lines);

  let released = false;
  const reader = {
    async read() {
      if (released) return { done: true, value: undefined };
      released = true;
      return { done: false, value: bytes };
    },
    releaseLock() {},
  };

  return {
    ok: true,
    status: 200,
    headers: new Headers(),
    body: { getReader: () => reader },
  } as unknown as Response;
}

describe("GeminiProvider – convertMessages with images", () => {
  it("converts image with Uint8Array data", async () => {
    const provider = new GeminiProvider({ GEMINI_API_KEY: "k" });

    const result = await provider.convertMessages([
      {
        role: "user",
        content: [
          { type: "text", text: "describe" },
          { type: "image", image: { data: new Uint8Array([1, 2, 3]) } },
        ],
      },
    ]);

    expect(result.contents).toHaveLength(1);
    const parts = result.contents[0].parts;
    expect(parts).toHaveLength(2);
    expect(parts[0]).toEqual({ text: "describe" });
    expect(parts[1].inlineData).toBeDefined();
    expect(parts[1].inlineData!.mimeType).toBe("image/jpeg");
  });

  it("converts image with string data", async () => {
    const provider = new GeminiProvider({ GEMINI_API_KEY: "k" });

    const result = await provider.convertMessages([
      {
        role: "user",
        content: [
          { type: "image", image: { data: "base64string" } },
        ],
      },
    ]);

    expect(result.contents[0].parts[0].inlineData!.data).toBe("base64string");
  });

  it("converts image with data URI", async () => {
    const base64 = Buffer.from("test").toString("base64");
    const provider = new GeminiProvider({ GEMINI_API_KEY: "k" });

    const result = await provider.convertMessages([
      {
        role: "user",
        content: [
          {
            type: "image",
            image: { uri: `data:image/png;base64,${base64}` },
          },
        ],
      },
    ]);

    expect(result.contents[0].parts[0].inlineData!.mimeType).toBe("image/png");
    expect(result.contents[0].parts[0].inlineData!.data).toBe(base64);
  });

  it("fetches image from remote URI", async () => {
    const fetchFn = vi.fn().mockImplementation(async (url: string) => {
      if (url.includes("example.com")) {
        return {
          ok: true,
          headers: new Headers({ "content-type": "image/jpeg" }),
          arrayBuffer: async () => Uint8Array.from([1, 2, 3]).buffer,
        };
      }
      return makeFetchResponse({});
    });

    const provider = new GeminiProvider({ GEMINI_API_KEY: "k" }, { fetchFn });

    const result = await provider.convertMessages([
      {
        role: "user",
        content: [
          { type: "image", image: { uri: "https://example.com/img.jpg" } },
        ],
      },
    ]);

    expect(result.contents[0].parts[0].inlineData!.mimeType).toBe(
      "image/jpeg"
    );
  });

  it("handles image with no data and no URI", async () => {
    const provider = new GeminiProvider({ GEMINI_API_KEY: "k" });

    const result = await provider.convertMessages([
      {
        role: "user",
        content: [{ type: "image", image: {} }],
      },
    ]);

    expect(result.contents[0].parts[0].inlineData!.data).toBe("");
  });

  it("converts audio content with Uint8Array data to inlineData", async () => {
    const provider = new GeminiProvider({ GEMINI_API_KEY: "k" });

    const result = await provider.convertMessages([
      {
        role: "user",
        content: [
          { type: "audio", audio: { data: new Uint8Array([1, 2, 3]), mimeType: "audio/mp3" } },
        ],
      },
    ]);

    expect(result.contents[0].parts[0].inlineData).toBeDefined();
    expect(result.contents[0].parts[0].inlineData!.mimeType).toBe("audio/mp3");
    expect(result.contents[0].parts[0].inlineData!.data).toBeTruthy();
  });

  it("converts audio content with string data to inlineData", async () => {
    const provider = new GeminiProvider({ GEMINI_API_KEY: "k" });

    const result = await provider.convertMessages([
      {
        role: "user",
        content: [
          { type: "audio", audio: { data: "base64audiodata" } },
        ],
      },
    ]);

    expect(result.contents[0].parts[0].inlineData).toBeDefined();
    expect(result.contents[0].parts[0].inlineData!.data).toBe("base64audiodata");
    expect(result.contents[0].parts[0].inlineData!.mimeType).toBe("audio/mp3");
  });

  it("converts audio content with data URI", async () => {
    const base64 = Buffer.from("audiodata").toString("base64");
    const provider = new GeminiProvider({ GEMINI_API_KEY: "k" });

    const result = await provider.convertMessages([
      {
        role: "user",
        content: [
          { type: "audio", audio: { uri: `data:audio/wav;base64,${base64}` } },
        ],
      },
    ]);

    expect(result.contents[0].parts[0].inlineData).toBeDefined();
    expect(result.contents[0].parts[0].inlineData!.mimeType).toBe("audio/wav");
    expect(result.contents[0].parts[0].inlineData!.data).toBe(base64);
  });

  it("converts audio content with remote URI", async () => {
    const fetchFn = vi.fn().mockImplementation(async (url: string) => {
      if (url.includes("example.com")) {
        return {
          ok: true,
          headers: new Headers({ "content-type": "audio/mpeg" }),
          arrayBuffer: async () => Uint8Array.from([4, 5, 6]).buffer,
        };
      }
      return makeFetchResponse({});
    });

    const provider = new GeminiProvider({ GEMINI_API_KEY: "k" }, { fetchFn });

    const result = await provider.convertMessages([
      {
        role: "user",
        content: [
          { type: "audio", audio: { uri: "https://example.com/audio.mp3" } },
        ],
      },
    ]);

    expect(result.contents[0].parts[0].inlineData).toBeDefined();
    expect(result.contents[0].parts[0].inlineData!.mimeType).toBe("audio/mpeg");
    expect(fetchFn).toHaveBeenCalledWith("https://example.com/audio.mp3");
  });

  it("converts audio content with no data and no URI to empty inlineData", async () => {
    const provider = new GeminiProvider({ GEMINI_API_KEY: "k" });

    const result = await provider.convertMessages([
      {
        role: "user",
        content: [
          { type: "audio", audio: {} },
        ],
      },
    ]);

    expect(result.contents[0].parts[0].inlineData).toBeDefined();
    expect(result.contents[0].parts[0].inlineData!.data).toBe("");
    expect(result.contents[0].parts[0].inlineData!.mimeType).toBe("audio/mp3");
  });
});

describe("GeminiProvider – convertMessages system with array content", () => {
  it("extracts text from array-typed system content", async () => {
    const provider = new GeminiProvider({ GEMINI_API_KEY: "k" });

    const result = await provider.convertMessages([
      {
        role: "system",
        content: [
          { type: "text", text: "part1" },
          { type: "text", text: "part2" },
        ],
      },
      { role: "user", content: "hello" },
    ]);

    expect(result.systemInstruction).toBe("part1 part2");
  });
});

describe("GeminiProvider – convertMessages assistant with content array", () => {
  it("converts assistant array content", async () => {
    const provider = new GeminiProvider({ GEMINI_API_KEY: "k" });

    const result = await provider.convertMessages([
      {
        role: "assistant",
        content: [{ type: "text", text: "thinking..." }],
      },
    ]);

    expect(result.contents[0].role).toBe("model");
    expect(result.contents[0].parts[0].text).toBe("thinking...");
  });

  it("skips assistant with no content and no tool calls", async () => {
    const provider = new GeminiProvider({ GEMINI_API_KEY: "k" });

    const result = await provider.convertMessages([
      { role: "assistant", content: null },
    ]);

    expect(result.contents).toHaveLength(0);
  });
});

describe("GeminiProvider – formatTools deduplication", () => {
  it("deduplicates tool names with suffix", () => {
    const provider = new GeminiProvider({ GEMINI_API_KEY: "k" });

    const { geminiTools, nameMap } = provider.formatTools([
      { name: "search", description: "First" },
      { name: "search", description: "Second" },
    ]);

    expect(geminiTools[0].functionDeclarations).toHaveLength(2);
    const names = geminiTools[0].functionDeclarations.map(
      (d: any) => d.name
    );
    expect(new Set(names).size).toBe(2); // unique names
  });

  it("returns empty geminiTools for no tools", () => {
    const provider = new GeminiProvider({ GEMINI_API_KEY: "k" });
    const { geminiTools } = provider.formatTools([]);
    expect(geminiTools).toEqual([]);
  });
});

describe("GeminiProvider – generateMessage with responseFormat", () => {
  it("sets responseMimeType for responseFormat", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      makeFetchResponse({
        candidates: [{ content: { parts: [{ text: '{"ok":true}' }] } }],
      })
    );

    const provider = new GeminiProvider({ GEMINI_API_KEY: "k" }, { fetchFn });

    await provider.generateMessage({
      model: "gemini-2.0-flash",
      messages: [{ role: "user", content: "json" }],
      responseFormat: { type: "json_object" },
    });

    const body = JSON.parse(fetchFn.mock.calls[0][1].body);
    expect(body.generationConfig.responseMimeType).toBe("application/json");
  });

  it("sets responseSchema for jsonSchema", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      makeFetchResponse({
        candidates: [{ content: { parts: [{ text: '{"ok":true}' }] } }],
      })
    );

    const provider = new GeminiProvider({ GEMINI_API_KEY: "k" }, { fetchFn });

    await provider.generateMessage({
      model: "gemini-2.0-flash",
      messages: [{ role: "user", content: "json" }],
      jsonSchema: { type: "object", properties: { ok: { type: "boolean" } } },
    });

    const body = JSON.parse(fetchFn.mock.calls[0][1].body);
    expect(body.generationConfig.responseMimeType).toBe("application/json");
    expect(body.generationConfig.responseSchema).toBeDefined();
  });
});

describe("GeminiProvider – generateMessage error handling", () => {
  it("throws on API error in response body", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      makeFetchResponse({
        error: { message: "quota exceeded" },
        candidates: [{ content: { parts: [{ text: "ok" }] } }],
      })
    );

    const provider = new GeminiProvider({ GEMINI_API_KEY: "k" }, { fetchFn });

    await expect(
      provider.generateMessage({
        model: "gemini-2.0-flash",
        messages: [{ role: "user", content: "hi" }],
      })
    ).rejects.toThrow("quota exceeded");
  });

  it("throws when no candidates returned", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      makeFetchResponse({ candidates: [] })
    );

    const provider = new GeminiProvider({ GEMINI_API_KEY: "k" }, { fetchFn });

    await expect(
      provider.generateMessage({
        model: "gemini-2.0-flash",
        messages: [{ role: "user", content: "hi" }],
      })
    ).rejects.toThrow("no candidates");
  });
});

describe("GeminiProvider – streaming error handling", () => {
  it("throws on non-OK streaming response", async () => {
    const fetchFn = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      headers: new Headers(),
      text: async () => "Internal Server Error",
      body: null,
    });

    const provider = new GeminiProvider({ GEMINI_API_KEY: "k" }, { fetchFn });

    const gen = provider.generateMessages({
      model: "gemini-2.0-flash",
      messages: [{ role: "user", content: "hi" }],
    });

    await expect(gen.next()).rejects.toThrow("Gemini API error 500");
  });

  it("throws when streaming response has no body", async () => {
    const fetchFn = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: new Headers(),
      body: null,
    });

    const provider = new GeminiProvider({ GEMINI_API_KEY: "k" }, { fetchFn });

    const gen = provider.generateMessages({
      model: "gemini-2.0-flash",
      messages: [{ role: "user", content: "hi" }],
    });

    await expect(gen.next()).rejects.toThrow("no body");
  });

  it("handles malformed JSON in SSE stream gracefully", async () => {
    // Create a stream with invalid JSON
    const encoder = new TextEncoder();
    const bytes = encoder.encode("data: {invalid json}\n\ndata: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]}}]}\n\n");

    let released = false;
    const reader = {
      async read() {
        if (released) return { done: true, value: undefined };
        released = true;
        return { done: false, value: bytes };
      },
      releaseLock() {},
    };

    const fetchFn = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: new Headers(),
      body: { getReader: () => reader },
    });

    const provider = new GeminiProvider({ GEMINI_API_KEY: "k" }, { fetchFn });

    const out: unknown[] = [];
    for await (const item of provider.generateMessages({
      model: "gemini-2.0-flash",
      messages: [{ role: "user", content: "hi" }],
    })) {
      out.push(item);
    }

    // Should skip bad JSON and still parse the good one + done
    const textChunks = out.filter((o: any) => o.type === "chunk" && o.content);
    expect(textChunks.length).toBeGreaterThan(0);
  });

  it("handles [DONE] signal in SSE stream", async () => {
    const encoder = new TextEncoder();
    const bytes = encoder.encode("data: [DONE]\n\n");

    let released = false;
    const reader = {
      async read() {
        if (released) return { done: true, value: undefined };
        released = true;
        return { done: false, value: bytes };
      },
      releaseLock() {},
    };

    const fetchFn = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: new Headers(),
      body: { getReader: () => reader },
    });

    const provider = new GeminiProvider({ GEMINI_API_KEY: "k" }, { fetchFn });

    const out: unknown[] = [];
    for await (const item of provider.generateMessages({
      model: "gemini-2.0-flash",
      messages: [{ role: "user", content: "hi" }],
    })) {
      out.push(item);
    }

    // Should only have the synthetic done chunk
    expect(out).toEqual([{ type: "chunk", content: "", done: true }]);
  });
});

describe("GeminiProvider – getAvailableLanguageModels with network error", () => {
  it("returns empty on fetch throw", async () => {
    const fetchFn = vi.fn().mockRejectedValue(new Error("network error"));
    const provider = new GeminiProvider({ GEMINI_API_KEY: "k" }, { fetchFn });
    const models = await provider.getAvailableLanguageModels();
    expect(models).toEqual([]);
  });
});

describe("GeminiProvider – streaming with systemInstruction", () => {
  it("includes systemInstruction in streaming request body", async () => {
    const events = [
      {
        candidates: [{ content: { parts: [{ text: "ok" }] } }],
      },
    ];

    const fetchFn = vi.fn().mockResolvedValue(makeSSEStream(events));

    const provider = new GeminiProvider({ GEMINI_API_KEY: "k" }, { fetchFn });

    const out: unknown[] = [];
    for await (const item of provider.generateMessages({
      model: "gemini-2.0-flash",
      messages: [
        { role: "system", content: "You are helpful" },
        { role: "user", content: "hi" },
      ],
    })) {
      out.push(item);
    }

    const body = JSON.parse(fetchFn.mock.calls[0][1].body);
    expect(body.systemInstruction).toEqual({ parts: [{ text: "You are helpful" }] });
  });
});

describe("GeminiProvider – streaming with tools", () => {
  it("includes tools in streaming request body", async () => {
    const events = [
      {
        candidates: [{ content: { parts: [{ text: "ok" }] } }],
      },
    ];

    const fetchFn = vi.fn().mockResolvedValue(makeSSEStream(events));

    const provider = new GeminiProvider({ GEMINI_API_KEY: "k" }, { fetchFn });

    const out: unknown[] = [];
    for await (const item of provider.generateMessages({
      model: "gemini-2.0-flash",
      messages: [{ role: "user", content: "hi" }],
      tools: [{ name: "search", description: "Search" }],
    })) {
      out.push(item);
    }

    const body = JSON.parse(fetchFn.mock.calls[0][1].body);
    expect(body.tools).toBeDefined();
  });
});

describe("GeminiProvider – streaming with tool name reverse mapping", () => {
  it("maps sanitized tool names back to original", async () => {
    const events = [
      {
        candidates: [
          {
            content: {
              parts: [
                {
                  functionCall: { name: "my_tool_", args: { x: 1 } },
                },
              ],
            },
          },
        ],
      },
    ];

    const fetchFn = vi.fn().mockResolvedValue(makeSSEStream(events));

    const provider = new GeminiProvider({ GEMINI_API_KEY: "k" }, { fetchFn });

    const out: unknown[] = [];
    for await (const item of provider.generateMessages({
      model: "gemini-2.0-flash",
      messages: [{ role: "user", content: "use tool" }],
      tools: [{ name: "my tool!", description: "test" }],
    })) {
      out.push(item);
    }

    // The tool call name should be mapped back to original
    const tc = out.find((o: any) => o.name) as any;
    expect(tc.name).toBe("my tool!");
  });
});
