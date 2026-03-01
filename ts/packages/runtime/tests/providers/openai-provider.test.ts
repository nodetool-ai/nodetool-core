import { describe, it, expect, vi } from "vitest";
import { OpenAIProvider } from "../../src/providers/openai-provider.js";
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

describe("OpenAIProvider", () => {
  it("reports tool support with o1/o3 exceptions", () => {
    const provider = new OpenAIProvider(
      { OPENAI_API_KEY: "k" },
      { client: {} as any }
    );

    expect(provider.hasToolSupport("gpt-4o")).toBe(true);
    expect(provider.hasToolSupport("o1-mini")).toBe(false);
    expect(provider.hasToolSupport("o3-mini")).toBe(false);
  });

  it("resolves image/video size helpers", () => {
    const provider = new OpenAIProvider(
      { OPENAI_API_KEY: "k" },
      { client: {} as any }
    );

    expect(provider.resolveImageSize(1024, 1024)).toBe("1024x1024");
    expect(OpenAIProvider.resolveVideoSize("16:9", "720p")).toBe("1280x720");
    expect(OpenAIProvider.secondsFromParams({ numFrames: 12 })).toBe(4);
    expect(OpenAIProvider.snapToValidVideoDimensions(1920, 1080)).toBe("1280x720");
  });

  it("converts messages into OpenAI format", async () => {
    const provider = new OpenAIProvider(
      { OPENAI_API_KEY: "k" },
      { client: {} as any }
    );

    const user: Message = { role: "user", content: "hello" };
    const assistant: Message = {
      role: "assistant",
      content: "ok",
      toolCalls: [{ id: "tc1", name: "sum", args: { a: 1 } }],
    };

    await expect(provider.convertMessage(user)).resolves.toEqual({
      role: "user",
      content: "hello",
    });

    await expect(provider.convertMessage(assistant)).resolves.toEqual({
      role: "assistant",
      content: "ok",
      tool_calls: [
        {
          type: "function",
          id: "tc1",
          function: {
            name: "sum",
            arguments: "{\"a\":1}",
          },
        },
      ],
    });
  });

  it("generates non-streaming message and parses tool calls", async () => {
    const create = vi.fn().mockResolvedValue({
      choices: [
        {
          message: {
            content: "done",
            tool_calls: [
              {
                id: "tc1",
                function: { name: "sum", arguments: "{\"a\":1}" },
              },
            ],
          },
        },
      ],
    });

    const provider = new OpenAIProvider(
      { OPENAI_API_KEY: "k" },
      {
        client: {
          chat: { completions: { create } },
        } as any,
      }
    );

    const result = await provider.generateMessage({
      model: "gpt-4o",
      messages: [{ role: "user", content: "hi" }],
    });

    expect(create).toHaveBeenCalledTimes(1);
    expect(result).toEqual({
      role: "assistant",
      content: "done",
      toolCalls: [{ id: "tc1", name: "sum", args: { a: 1 } }],
    });
  });

  it("streams chunks and tool calls", async () => {
    const stream = makeAsyncIterable([
      {
        choices: [{ delta: { content: "Hello" }, finish_reason: null }],
      },
      {
        choices: [
          {
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: "tc1",
                  function: { name: "lookup", arguments: "{\"q\":" },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      },
      {
        choices: [
          {
            delta: {
              tool_calls: [
                {
                  index: 0,
                  function: { arguments: "\"x\"}" },
                },
              ],
            },
            finish_reason: "tool_calls",
          },
        ],
      },
      {
        choices: [{ delta: { content: "" }, finish_reason: "stop" }],
      },
    ]);

    const create = vi.fn().mockResolvedValue(stream);

    const provider = new OpenAIProvider(
      { OPENAI_API_KEY: "k" },
      {
        client: {
          chat: { completions: { create } },
        } as any,
      }
    );

    const out: Array<unknown> = [];
    for await (const item of provider.generateMessages({
      model: "gpt-4o",
      messages: [{ role: "user", content: "hi" }],
    })) {
      out.push(item);
    }

    expect(out).toEqual([
      { type: "chunk", content: "Hello", done: false },
      { id: "tc1", name: "lookup", args: { q: "x" } },
      { type: "chunk", content: "", done: true },
    ]);
  });

  it("returns static model lists", async () => {
    const provider = new OpenAIProvider(
      { OPENAI_API_KEY: "k" },
      { client: {} as any }
    );

    await expect(provider.getAvailableASRModels()).resolves.toHaveLength(1);
    await expect(provider.getAvailableTTSModels()).resolves.toHaveLength(2);
    await expect(provider.getAvailableImageModels()).resolves.toHaveLength(3);
    await expect(provider.getAvailableVideoModels()).resolves.toHaveLength(2);
    await expect(provider.getAvailableEmbeddingModels()).resolves.toHaveLength(3);
  });
});
