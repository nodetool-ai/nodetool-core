/**
 * Tests for BaseProvider utility methods and default behaviors.
 */

import { describe, it, expect } from "vitest";
import { BaseProvider } from "../../src/providers/base-provider.js";
import type {
  Message,
  ProviderStreamItem,
  ProviderTool,
  ToolCall,
} from "../../src/providers/types.js";

/**
 * Concrete subclass that exposes protected methods for testing.
 */
class TestProvider extends BaseProvider {
  constructor() {
    super("test");
  }

  public testParseToolCallArgs(raw: unknown): Record<string, unknown> {
    return this.parseToolCallArgs(raw);
  }

  public testBuildToolCall(
    id: string,
    name: string,
    args: unknown
  ): ToolCall {
    return this.buildToolCall(id, name, args);
  }

  async generateMessage(_args: {
    messages: Message[];
    model: string;
    tools?: ProviderTool[];
    maxTokens?: number;
    responseFormat?: Record<string, unknown>;
    jsonSchema?: Record<string, unknown>;
    temperature?: number;
    topP?: number;
    presencePenalty?: number;
    frequencyPenalty?: number;
  }): Promise<Message> {
    return { role: "assistant", content: "ok" };
  }

  async *generateMessages(_args: {
    messages: Message[];
    model: string;
    tools?: ProviderTool[];
    maxTokens?: number;
    responseFormat?: Record<string, unknown>;
    jsonSchema?: Record<string, unknown>;
    temperature?: number;
    topP?: number;
    presencePenalty?: number;
    frequencyPenalty?: number;
    audio?: Record<string, unknown>;
  }): AsyncGenerator<ProviderStreamItem> {
    yield { type: "chunk", content: "ok", done: true };
  }
}

describe("BaseProvider – parseToolCallArgs", () => {
  const provider = new TestProvider();

  it("parses valid JSON string into object", () => {
    const result = provider.testParseToolCallArgs('{"key": "value", "n": 42}');
    expect(result).toEqual({ key: "value", n: 42 });
  });

  it("returns {} for non-string input", () => {
    expect(provider.testParseToolCallArgs(123)).toEqual({});
    expect(provider.testParseToolCallArgs(null)).toEqual({});
    expect(provider.testParseToolCallArgs(undefined)).toEqual({});
    expect(provider.testParseToolCallArgs({ key: "val" })).toEqual({});
  });

  it("returns {} for invalid JSON", () => {
    expect(provider.testParseToolCallArgs("not json")).toEqual({});
    expect(provider.testParseToolCallArgs("{broken")).toEqual({});
  });

  it("returns {} for JSON that parses to array", () => {
    expect(provider.testParseToolCallArgs("[1, 2, 3]")).toEqual({});
  });

  it("returns {} for JSON that parses to primitive", () => {
    expect(provider.testParseToolCallArgs('"hello"')).toEqual({});
    expect(provider.testParseToolCallArgs("42")).toEqual({});
    expect(provider.testParseToolCallArgs("true")).toEqual({});
    expect(provider.testParseToolCallArgs("null")).toEqual({});
  });
});

describe("BaseProvider – buildToolCall", () => {
  const provider = new TestProvider();

  it("builds ToolCall with parsed args from JSON string", () => {
    const tc = provider.testBuildToolCall(
      "call-1",
      "myTool",
      '{"foo": "bar"}'
    );
    expect(tc).toEqual({
      id: "call-1",
      name: "myTool",
      args: { foo: "bar" },
    });
  });

  it("handles string args", () => {
    const tc = provider.testBuildToolCall("call-2", "tool", '{"x": 1}');
    expect(tc.id).toBe("call-2");
    expect(tc.name).toBe("tool");
    expect(tc.args).toEqual({ x: 1 });
  });

  it("handles non-string args (returns empty args)", () => {
    const tc = provider.testBuildToolCall("call-3", "tool", 999);
    expect(tc.args).toEqual({});
  });
});

describe("BaseProvider – isContextLengthError", () => {
  const provider = new TestProvider();

  it('returns true for "context length exceeded"', () => {
    expect(
      provider.isContextLengthError(
        new Error("The context length exceeded the maximum allowed")
      )
    ).toBe(true);
  });

  it('returns true for "maximum context"', () => {
    expect(
      provider.isContextLengthError("maximum context size reached")
    ).toBe(true);
  });

  it("returns false for unrelated errors", () => {
    expect(provider.isContextLengthError(new Error("rate limit"))).toBe(false);
    expect(provider.isContextLengthError("something else")).toBe(false);
    expect(provider.isContextLengthError(42)).toBe(false);
  });
});

describe("BaseProvider – default method behaviors", () => {
  const provider = new TestProvider();

  it("requiredSecrets() returns []", () => {
    expect(TestProvider.requiredSecrets()).toEqual([]);
  });

  it("getContainerEnv() returns {}", () => {
    expect(provider.getContainerEnv()).toEqual({});
  });

  it("hasToolSupport() returns true", () => {
    expect(provider.hasToolSupport("any-model")).toBe(true);
  });

  it("getAvailableLanguageModels() returns []", async () => {
    await expect(provider.getAvailableLanguageModels()).resolves.toEqual([]);
  });

  it("textToImage() throws 'does not support'", async () => {
    await expect(
      provider.textToImage({
        model: { id: "m", name: "m", provider: "test" },
        prompt: "test",
      })
    ).rejects.toThrow("does not support");
  });

  it("imageToImage() throws 'does not support'", async () => {
    await expect(
      provider.imageToImage(new Uint8Array(), {
        model: { id: "m", name: "m", provider: "test" },
        prompt: "test",
      })
    ).rejects.toThrow("does not support");
  });

  it("textToSpeech() throws 'does not support'", async () => {
    const gen = provider.textToSpeech({ text: "hi", model: "m" });
    await expect(gen.next()).rejects.toThrow("does not support");
  });

  it("automaticSpeechRecognition() throws 'does not support'", async () => {
    await expect(
      provider.automaticSpeechRecognition({
        audio: new Uint8Array(),
        model: "m",
      })
    ).rejects.toThrow("does not support");
  });

  it("generateEmbedding() throws 'does not support'", async () => {
    await expect(
      provider.generateEmbedding({ text: "hi", model: "m" })
    ).rejects.toThrow("does not support");
  });
});
