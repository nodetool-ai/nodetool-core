/**
 * Additional coverage tests for LlamaProvider – parseKeywordArgs,
 * normalizeMessagesForLlama, edge cases.
 */

import { describe, it, expect, vi } from "vitest";
import { LlamaProvider } from "../../src/providers/llama-provider.js";
import type { Message } from "../../src/providers/types.js";

function makeAsyncIterable(items: unknown[]) {
  return {
    async *[Symbol.asyncIterator]() {
      for (const item of items) {
        yield item;
      }
    },
    async close() {
      return;
    },
  };
}

describe("LlamaProvider – message normalization for strict alternation", () => {
  it("inserts empty messages to maintain user/assistant alternation", async () => {
    const create = vi.fn().mockResolvedValue({
      choices: [{ message: { content: "ok" } }],
    });

    const provider = new LlamaProvider(
      { LLAMA_CPP_URL: "http://localhost:8080" },
      { client: { chat: { completions: { create } } } as any }
    );

    // Two consecutive user messages should get an empty assistant between them
    await provider.generateMessage({
      model: "test",
      messages: [
        { role: "user", content: "first" },
        { role: "user", content: "second" },
      ],
    });

    const sentMessages = create.mock.calls[0][0].messages;
    // Should have system-injected padding for alternation
    // user -> assistant (empty) -> user
    const roles = sentMessages.map((m: any) => m.role);
    // Verify alternation is maintained
    for (let i = 1; i < roles.length; i++) {
      if (roles[i - 1] !== "system") {
        expect(roles[i]).not.toBe(roles[i - 1]);
      }
    }
  });

  it("converts system messages into a merged system message", async () => {
    const create = vi.fn().mockResolvedValue({
      choices: [{ message: { content: "ok" } }],
    });

    const provider = new LlamaProvider(
      { LLAMA_CPP_URL: "http://localhost:8080" },
      { client: { chat: { completions: { create } } } as any }
    );

    await provider.generateMessage({
      model: "test",
      messages: [
        { role: "system", content: "Be concise." },
        { role: "system", content: "Be helpful." },
        { role: "user", content: "hi" },
      ],
    });

    const sentMessages = create.mock.calls[0][0].messages;
    const systemMsg = sentMessages.find((m: any) => m.role === "system");
    expect(systemMsg.content).toContain("Be concise.");
    expect(systemMsg.content).toContain("Be helpful.");
  });

  it("converts tool messages to user messages", async () => {
    const create = vi.fn().mockResolvedValue({
      choices: [{ message: { content: "ok" } }],
    });

    const provider = new LlamaProvider(
      { LLAMA_CPP_URL: "http://localhost:8080" },
      { client: { chat: { completions: { create } } } as any }
    );

    await provider.generateMessage({
      model: "test",
      messages: [
        { role: "user", content: "calc" },
        {
          role: "assistant",
          content: "calling tool",
          toolCalls: [{ id: "tc1", name: "calc", args: {} }],
        },
        { role: "tool", content: { result: 42 }, toolCallId: "tc1" },
      ],
    });

    const sentMessages = create.mock.calls[0][0].messages;
    const toolResult = sentMessages.find((m: any) =>
      (m.content || "").includes("Tool result:")
    );
    expect(toolResult).toBeDefined();
    expect(toolResult.role).toBe("user");
  });
});

describe("LlamaProvider – convertMessage system message", () => {
  it("converts system role message", async () => {
    const provider = new LlamaProvider(
      { LLAMA_CPP_URL: "http://localhost:8080" },
      { client: {} as any }
    );

    const result = await provider.convertMessage({
      role: "system",
      content: "Be helpful",
    });
    expect(result).toEqual({ role: "system", content: "Be helpful" });
  });

  it("handles system message with array content", async () => {
    const provider = new LlamaProvider(
      { LLAMA_CPP_URL: "http://localhost:8080" },
      { client: {} as any }
    );

    const result = await provider.convertMessage({
      role: "system",
      content: [{ type: "text", text: "part1" }],
    });
    expect(result).toEqual({ role: "system", content: "part1" });
  });
});

describe("LlamaProvider – formatTools", () => {
  it("formats tools in OpenAI function format", () => {
    const provider = new LlamaProvider(
      { LLAMA_CPP_URL: "http://localhost:8080" },
      { client: {} as any }
    );

    const result = provider.formatTools([
      {
        name: "search",
        description: "Search the web",
        inputSchema: { type: "object", properties: { q: { type: "string" } } },
      },
    ]);

    expect(result).toEqual([
      {
        type: "function",
        function: {
          name: "search",
          description: "Search the web",
          parameters: { type: "object", properties: { q: { type: "string" } } },
        },
      },
    ]);
  });
});

describe("LlamaProvider – generateMessage with responseFormat", () => {
  it("passes responseFormat to request", async () => {
    const create = vi.fn().mockResolvedValue({
      choices: [{ message: { content: '{"result": 42}' } }],
    });

    const provider = new LlamaProvider(
      { LLAMA_CPP_URL: "http://localhost:8080" },
      { client: { chat: { completions: { create } } } as any }
    );

    await provider.generateMessage({
      model: "test",
      messages: [{ role: "user", content: "json" }],
      responseFormat: { type: "json_object" },
    });

    expect(create.mock.calls[0][0].response_format).toEqual({
      type: "json_object",
    });
  });

  it("throws on jsonSchema in non-streaming", async () => {
    const provider = new LlamaProvider(
      { LLAMA_CPP_URL: "http://localhost:8080" },
      { client: { chat: { completions: { create: vi.fn() } } } as any }
    );

    await expect(
      provider.generateMessage({
        model: "test",
        messages: [{ role: "user", content: "hi" }],
        jsonSchema: { type: "object" },
      })
    ).rejects.toThrow("jsonSchema is not supported");
  });
});

describe("LlamaProvider – generateMessages with native tool_calls in stream", () => {
  it("yields tool calls from stream", async () => {
    const stream = makeAsyncIterable([
      {
        choices: [
          {
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: "tc1",
                  function: { name: "search", arguments: '{"q":"x"}' },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      },
      {
        choices: [{ delta: {}, finish_reason: "tool_calls" }],
      },
    ]);

    const create = vi.fn().mockResolvedValue(stream);
    const provider = new LlamaProvider(
      { LLAMA_CPP_URL: "http://localhost:8080" },
      { client: { chat: { completions: { create } } } as any }
    );

    const out: unknown[] = [];
    for await (const item of provider.generateMessages({
      model: "test",
      messages: [{ role: "user", content: "search" }],
    })) {
      out.push(item);
    }

    expect(out).toEqual([
      { id: "tc1", name: "search", args: { q: "x" } },
    ]);
  });

  it("throws on jsonSchema in streaming", async () => {
    const provider = new LlamaProvider(
      { LLAMA_CPP_URL: "http://localhost:8080" },
      { client: { chat: { completions: { create: vi.fn() } } } as any }
    );

    const gen = provider.generateMessages({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
      jsonSchema: { type: "object" },
    });

    await expect(gen.next()).rejects.toThrow("jsonSchema is not supported");
  });
});

describe("LlamaProvider – generateMessage with native tool calls", () => {
  it("uses native tool calls when available", async () => {
    const create = vi.fn().mockResolvedValue({
      choices: [
        {
          message: {
            content: "",
            tool_calls: [
              {
                id: "tc1",
                function: { name: "calc", arguments: '{"expr":"1+1"}' },
              },
            ],
          },
        },
      ],
    });

    const provider = new LlamaProvider(
      { LLAMA_CPP_URL: "http://localhost:8080" },
      { client: { chat: { completions: { create } } } as any }
    );

    const result = await provider.generateMessage({
      model: "test",
      messages: [{ role: "user", content: "calc" }],
      tools: [{ name: "calc" }],
    });

    expect(result.toolCalls).toEqual([
      { id: "tc1", name: "calc", args: { expr: "1+1" } },
    ]);
  });
});

describe("LlamaProvider – getAvailableLanguageModels fallback", () => {
  it("uses models key as fallback", async () => {
    const fetchFn = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ models: [{ id: "model1" }] }),
    });

    const provider = new LlamaProvider(
      { LLAMA_CPP_URL: "http://localhost:8080" },
      { client: {} as any, fetchFn: fetchFn as any }
    );

    const models = await provider.getAvailableLanguageModels();
    expect(models).toEqual([
      { id: "model1", name: "model1", provider: "llama_cpp" },
    ]);
  });

  it("returns empty on failure", async () => {
    const fetchFn = vi.fn().mockResolvedValue({ ok: false });

    const provider = new LlamaProvider(
      { LLAMA_CPP_URL: "http://localhost:8080" },
      { client: {} as any, fetchFn: fetchFn as any }
    );

    expect(await provider.getAvailableLanguageModels()).toEqual([]);
  });
});
