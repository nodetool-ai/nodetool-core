import { describe, it, expect, vi } from "vitest";
import { OpenRouterProvider } from "../../src/providers/openrouter-provider.js";
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

describe("OpenRouterProvider", () => {
  it("throws if OPENROUTER_API_KEY is missing", () => {
    expect(() => new OpenRouterProvider({})).toThrow(
      "OPENROUTER_API_KEY is required"
    );
  });

  it("reports provider id as openrouter", () => {
    const provider = new OpenRouterProvider(
      { OPENROUTER_API_KEY: "k" },
      { client: {} as any }
    );
    expect(provider.provider).toBe("openrouter");
  });

  it("returns required secrets", () => {
    expect(OpenRouterProvider.requiredSecrets()).toEqual([
      "OPENROUTER_API_KEY",
    ]);
  });

  it("returns container env with OPENROUTER_API_KEY", () => {
    const provider = new OpenRouterProvider(
      { OPENROUTER_API_KEY: "test-key" },
      { client: {} as any }
    );
    expect(provider.getContainerEnv()).toEqual({
      OPENROUTER_API_KEY: "test-key",
    });
  });

  it("reports tool support with o1/o3 exceptions", async () => {
    const provider = new OpenRouterProvider(
      { OPENROUTER_API_KEY: "k" },
      { client: {} as any }
    );
    expect(await provider.hasToolSupport("gpt-4o")).toBe(true);
    expect(await provider.hasToolSupport("claude-3-opus")).toBe(true);
    expect(await provider.hasToolSupport("openai/o1-mini")).toBe(false);
    expect(await provider.hasToolSupport("openai/o3-mini")).toBe(false);
  });

  it("fetches available language models with extra headers", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        data: [
          { id: "openai/gpt-4o", name: "GPT-4o" },
          { id: "anthropic/claude-3-opus" },
        ],
      }),
    });

    const provider = new OpenRouterProvider(
      { OPENROUTER_API_KEY: "k" },
      { client: {} as any, fetchFn: mockFetch as any }
    );

    const models = await provider.getAvailableLanguageModels();
    expect(models).toEqual([
      { id: "openai/gpt-4o", name: "GPT-4o", provider: "openrouter" },
      {
        id: "anthropic/claude-3-opus",
        name: "anthropic/claude-3-opus",
        provider: "openrouter",
      },
    ]);

    expect(mockFetch).toHaveBeenCalledWith(
      "https://openrouter.ai/api/v1/models",
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: "Bearer k",
          "HTTP-Referer": "https://github.com/nodetool-ai/nodetool-core",
          "X-Title": "NodeTool",
        }),
      })
    );
  });

  it("returns empty list when model fetch fails", async () => {
    const mockFetch = vi.fn().mockResolvedValue({ ok: false });
    const provider = new OpenRouterProvider(
      { OPENROUTER_API_KEY: "k" },
      { client: {} as any, fetchFn: mockFetch as any }
    );

    const models = await provider.getAvailableLanguageModels();
    expect(models).toEqual([]);
  });

  it("generates non-streaming message via inherited OpenAI logic", async () => {
    const create = vi.fn().mockResolvedValue({
      choices: [
        {
          message: {
            content: "routed response",
            tool_calls: null,
          },
        },
      ],
    });

    const provider = new OpenRouterProvider(
      { OPENROUTER_API_KEY: "k" },
      {
        client: {
          chat: { completions: { create } },
        } as any,
      }
    );

    const messages: Message[] = [{ role: "user", content: "hello" }];
    const result = await provider.generateMessage({
      messages,
      model: "openai/gpt-4o",
    });

    expect(result.role).toBe("assistant");
    expect(result.content).toBe("routed response");
  });

  it("streams messages via inherited OpenAI logic", async () => {
    const chunks = [
      {
        choices: [
          {
            delta: { content: "streamed" },
            finish_reason: null,
          },
        ],
      },
      {
        choices: [
          {
            delta: { content: "" },
            finish_reason: "stop",
          },
        ],
      },
    ];

    const create = vi.fn().mockResolvedValue(makeAsyncIterable(chunks));

    const provider = new OpenRouterProvider(
      { OPENROUTER_API_KEY: "k" },
      {
        client: {
          chat: { completions: { create } },
        } as any,
      }
    );

    const messages: Message[] = [{ role: "user", content: "hi" }];
    const items: unknown[] = [];
    for await (const item of provider.generateMessages({
      messages,
      model: "openai/gpt-4o",
    })) {
      items.push(item);
    }

    expect(items.length).toBeGreaterThanOrEqual(1);
  });
});
