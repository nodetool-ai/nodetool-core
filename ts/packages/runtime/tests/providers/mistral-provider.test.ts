import { describe, it, expect, vi } from "vitest";
import { MistralProvider } from "../../src/providers/mistral-provider.js";
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

describe("MistralProvider", () => {
  it("throws if MISTRAL_API_KEY is missing", () => {
    expect(() => new MistralProvider({})).toThrow("MISTRAL_API_KEY is required");
  });

  it("reports provider id as mistral", () => {
    const provider = new MistralProvider(
      { MISTRAL_API_KEY: "k" },
      { client: {} as any }
    );
    expect(provider.provider).toBe("mistral");
  });

  it("returns required secrets", () => {
    expect(MistralProvider.requiredSecrets()).toEqual(["MISTRAL_API_KEY"]);
  });

  it("returns container env with MISTRAL_API_KEY", () => {
    const provider = new MistralProvider(
      { MISTRAL_API_KEY: "test-key" },
      { client: {} as any }
    );
    expect(provider.getContainerEnv()).toEqual({ MISTRAL_API_KEY: "test-key" });
  });

  it("has tool support for all models", () => {
    const provider = new MistralProvider(
      { MISTRAL_API_KEY: "k" },
      { client: {} as any }
    );
    expect(provider.hasToolSupport("mistral-large")).toBe(true);
    expect(provider.hasToolSupport("pixtral-12b")).toBe(true);
  });

  it("fetches available language models", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        data: [
          { id: "mistral-large", name: "Mistral Large" },
          { id: "mistral-small" },
        ],
      }),
    });

    const provider = new MistralProvider(
      { MISTRAL_API_KEY: "k" },
      { client: {} as any, fetchFn: mockFetch as any }
    );

    const models = await provider.getAvailableLanguageModels();
    expect(models).toEqual([
      { id: "mistral-large", name: "Mistral Large", provider: "mistral" },
      { id: "mistral-small", name: "mistral-small", provider: "mistral" },
    ]);

    expect(mockFetch).toHaveBeenCalledWith(
      "https://api.mistral.ai/v1/models",
      expect.objectContaining({
        headers: { Authorization: "Bearer k" },
      })
    );
  });

  it("returns empty list when model fetch fails", async () => {
    const mockFetch = vi.fn().mockResolvedValue({ ok: false });
    const provider = new MistralProvider(
      { MISTRAL_API_KEY: "k" },
      { client: {} as any, fetchFn: mockFetch as any }
    );

    const models = await provider.getAvailableLanguageModels();
    expect(models).toEqual([]);
  });

  it("returns embedding models", async () => {
    const provider = new MistralProvider(
      { MISTRAL_API_KEY: "k" },
      { client: {} as any }
    );

    const models = await provider.getAvailableEmbeddingModels();
    expect(models).toEqual([
      {
        id: "mistral-embed",
        name: "Mistral Embed",
        provider: "mistral",
        dimensions: 1024,
      },
    ]);
  });

  it("generates non-streaming message via inherited OpenAI logic", async () => {
    const create = vi.fn().mockResolvedValue({
      choices: [
        {
          message: {
            content: "bonjour",
            tool_calls: null,
          },
        },
      ],
    });

    const provider = new MistralProvider(
      { MISTRAL_API_KEY: "k" },
      {
        client: {
          chat: { completions: { create } },
        } as any,
      }
    );

    const messages: Message[] = [{ role: "user", content: "hello" }];
    const result = await provider.generateMessage({
      messages,
      model: "mistral-large",
    });

    expect(result.role).toBe("assistant");
    expect(result.content).toBe("bonjour");
  });

  it("streams messages via inherited OpenAI logic", async () => {
    const chunks = [
      {
        choices: [
          {
            delta: { content: "salut" },
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

    const provider = new MistralProvider(
      { MISTRAL_API_KEY: "k" },
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
      model: "mistral-large",
    })) {
      items.push(item);
    }

    expect(items.length).toBeGreaterThanOrEqual(1);
  });
});
