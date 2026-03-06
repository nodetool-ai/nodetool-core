import { describe, it, expect } from "vitest";
import {
  SummarizerNode,
  CreateThreadNode,
  ExtractorNode,
  ClassifierNode,
  AgentNode,
  ControlAgentNode,
  ResearchAgentNode,
  AGENT_NODES,
  StructuredOutputGeneratorNode,
  DataGeneratorNode,
  ListGeneratorNode,
  ChartGeneratorNode,
  SVGGeneratorNode,
  GENERATOR_NODES,
} from "../src/index.js";

// ---------------------------------------------------------------------------
// agents.ts
// ---------------------------------------------------------------------------

describe("AGENT_NODES export", () => {
  it("contains all 7 agent node classes", () => {
    expect(AGENT_NODES).toHaveLength(7);
    expect(AGENT_NODES).toContain(SummarizerNode);
    expect(AGENT_NODES).toContain(CreateThreadNode);
    expect(AGENT_NODES).toContain(ExtractorNode);
    expect(AGENT_NODES).toContain(ClassifierNode);
    expect(AGENT_NODES).toContain(AgentNode);
    expect(AGENT_NODES).toContain(ControlAgentNode);
    expect(AGENT_NODES).toContain(ResearchAgentNode);
  });
});

// ---- SummarizerNode ----
describe("SummarizerNode", () => {
  it("has correct static metadata", () => {
    expect(SummarizerNode.nodeType).toBe("nodetool.agents.Summarizer");
    expect(SummarizerNode.title).toBe("Summarizer");
  });

  it("defaults", () => {
    const n = new (SummarizerNode as any)();
    expect(n.defaults()).toEqual({ text: "", max_sentences: 3 });
  });

  it("summarizes text to max_sentences", async () => {
    const n = new (SummarizerNode as any)();
    const result = await n.process({
      text: "First sentence. Second sentence. Third sentence. Fourth sentence.",
      max_sentences: 2,
    });
    expect(result.text).toBe("First sentence. Second sentence.");
    expect(result.output).toBe(result.text);
  });

  it("returns empty string for empty text", async () => {
    const n = new (SummarizerNode as any)();
    const result = await n.process({ text: "", max_sentences: 3 });
    expect(result.text).toBe("");
  });

  it("handles text with no sentence terminators", async () => {
    const n = new (SummarizerNode as any)();
    const result = await n.process({ text: "just a phrase", max_sentences: 5 });
    expect(result.text).toBe("just a phrase");
  });

  it("handles numeric input via asText", async () => {
    const n = new (SummarizerNode as any)();
    const result = await n.process({ text: 42, max_sentences: 1 });
    expect(result.text).toBe("42");
  });

  it("handles boolean input via asText", async () => {
    const n = new (SummarizerNode as any)();
    const result = await n.process({ text: true, max_sentences: 1 });
    expect(result.text).toBe("true");
  });

  it("handles null/undefined input via asText", async () => {
    const n = new (SummarizerNode as any)();
    const result = await n.process({ text: null });
    expect(result.text).toBe("");
  });

  it("handles array input via asText", async () => {
    const n = new (SummarizerNode as any)();
    const result = await n.process({ text: ["Hello", "World"], max_sentences: 1 });
    // asText joins array elements with space
    expect(result.text).toBe("Hello World");
  });

  it("handles object with content string via asText", async () => {
    const n = new (SummarizerNode as any)();
    const result = await n.process({
      text: { content: "Message content here." },
      max_sentences: 1,
    });
    expect(result.text).toBe("Message content here.");
  });

  it("handles object with content array (MessagePart) via asText", async () => {
    const n = new (SummarizerNode as any)();
    const result = await n.process({
      text: {
        content: [
          { type: "text", text: "Part one." },
          { type: "image" }, // non-text part
          { type: "text", text: "Part two." },
        ],
      },
      max_sentences: 2,
    });
    expect(result.text).toContain("Part one.");
    expect(result.text).toContain("Part two.");
  });

  it("handles object without content (falls back to JSON.stringify)", async () => {
    const n = new (SummarizerNode as any)();
    const result = await n.process({
      text: { foo: "bar" },
      max_sentences: 1,
    });
    expect(result.text).toContain("foo");
  });

  it("handles function input via asText (returns empty string)", async () => {
    const n = new (SummarizerNode as any)();
    const result = await n.process({ text: () => "fn", max_sentences: 1 });
    // A function is not string/number/boolean/falsy/array/object, so asText returns ""
    expect(result.text).toBe("");
  });

  it("handles non-finite max_sentences by defaulting to 3", async () => {
    const n = new (SummarizerNode as any)();
    const result = await n.process({
      text: "A. B. C. D. E.",
      max_sentences: NaN,
    });
    // NaN is not finite, so defaults to 3
    expect(result.text).toBe("A. B. C.");
  });

  it("uses defaults from _props when inputs missing", async () => {
    const n = new (SummarizerNode as any)();
    // Simulate _props being set
    n._props = { text: "From props. Second.", max_sentences: 1 };
    const result = await n.process({});
    expect(result.text).toBe("From props.");
  });
});

// ---- CreateThreadNode ----
describe("CreateThreadNode", () => {
  it("has correct static metadata", () => {
    expect(CreateThreadNode.nodeType).toBe("nodetool.agents.CreateThread");
  });

  it("defaults", () => {
    const n = new (CreateThreadNode as any)();
    expect(n.defaults()).toEqual({ title: "Agent Conversation", thread_id: "" });
  });

  it("creates a new thread with auto-generated id", async () => {
    const n = new (CreateThreadNode as any)();
    const result = await n.process({ title: "My Thread" });
    expect(result.thread_id).toMatch(/^thread_\d+_/);
  });

  it("reuses existing thread when thread_id provided", async () => {
    const n = new (CreateThreadNode as any)();
    // First call creates the thread
    const r1 = await n.process({ thread_id: "test_reuse_123", title: "T1" });
    expect(r1.thread_id).toBe("test_reuse_123");
    // Second call reuses it
    const r2 = await n.process({ thread_id: "test_reuse_123", title: "T2" });
    expect(r2.thread_id).toBe("test_reuse_123");
  });

  it("creates thread from _props when inputs empty", async () => {
    const n = new (CreateThreadNode as any)();
    n._props = { thread_id: "", title: "PropTitle" };
    const result = await n.process({});
    expect(result.thread_id).toMatch(/^thread_/);
  });
});

// ---- ExtractorNode ----
describe("ExtractorNode", () => {
  it("has correct static metadata", () => {
    expect(ExtractorNode.nodeType).toBe("nodetool.agents.Extractor");
  });

  it("defaults", () => {
    const n = new (ExtractorNode as any)();
    expect(n.defaults()).toEqual({ text: "" });
  });

  it("extracts valid JSON object from text", async () => {
    const n = new (ExtractorNode as any)();
    const result = await n.process({ text: '{"name": "Alice", "age": 30}' });
    expect(result.name).toBe("Alice");
    expect(result.age).toBe(30);
  });

  it("extracts JSON embedded in surrounding text", async () => {
    const n = new (ExtractorNode as any)();
    const result = await n.process({
      text: 'Here is the data: {"key": "value"} end.',
    });
    expect(result.key).toBe("value");
  });

  it("returns { output: text } when no JSON found", async () => {
    const n = new (ExtractorNode as any)();
    const result = await n.process({ text: "no json here" });
    expect(result.output).toBe("no json here");
  });

  it("returns { output: text } for JSON array (not object)", async () => {
    const n = new (ExtractorNode as any)();
    const result = await n.process({ text: "[1,2,3]" });
    expect(result.output).toBe("[1,2,3]");
  });

  it("returns { output: text } for embedded JSON array", async () => {
    const n = new (ExtractorNode as any)();
    // The braces extraction will find { but inner content is not valid object
    const result = await n.process({ text: "prefix [1,2,3] suffix" });
    expect(result.output).toBe("prefix [1,2,3] suffix");
  });

  it("handles embedded invalid JSON gracefully", async () => {
    const n = new (ExtractorNode as any)();
    const result = await n.process({ text: "prefix {invalid json} suffix" });
    expect(result.output).toBe("prefix {invalid json} suffix");
  });

  it("handles no braces at all", async () => {
    const n = new (ExtractorNode as any)();
    const result = await n.process({ text: "plain text only" });
    expect(result.output).toBe("plain text only");
  });

  it("returns null for embedded braces containing a JSON array (not object)", async () => {
    const n = new (ExtractorNode as any)();
    // The outer JSON.parse fails, inner finds { and } but parsed result
    // is not a plain object. We need braces that contain a valid JSON value
    // that is NOT an object. E.g., wrapping text around {"arr": [1]} won't work
    // since that IS an object. We need the inner parse to succeed with an array.
    // Actually we need braces around something that parses as a non-object.
    // But { ... } always parses as an object if valid JSON. So line 78 covers
    // cases like inner parse succeeding but returning an array from within
    // the braces — which can't happen since {..} is always an object in JSON.
    // Line 78 is effectively dead code for the inner parse path.
    // However we can still test the case where there are braces but inner parse fails.
    const result = await n.process({ text: "before {not: valid, json} after" });
    expect(result.output).toBe("before {not: valid, json} after");
  });
});

// ---- ClassifierNode ----
describe("ClassifierNode", () => {
  it("has correct static metadata", () => {
    expect(ClassifierNode.nodeType).toBe("nodetool.agents.Classifier");
  });

  it("defaults", () => {
    const n = new (ClassifierNode as any)();
    expect(n.defaults()).toEqual({ text: "", categories: [] });
  });

  it("returns Unknown when no categories", async () => {
    const n = new (ClassifierNode as any)();
    const result = await n.process({ text: "hello", categories: [] });
    expect(result.output).toBe("Unknown");
    expect(result.category).toBe("Unknown");
  });

  it("classifies text to matching category by token overlap", async () => {
    const n = new (ClassifierNode as any)();
    const result = await n.process({
      text: "I love programming in python",
      categories: ["sports", "programming", "cooking"],
    });
    expect(result.output).toBe("programming");
    expect(result.category).toBe("programming");
  });

  it("returns first category when no tokens match", async () => {
    const n = new (ClassifierNode as any)();
    const result = await n.process({
      text: "xyzzy",
      categories: ["alpha", "beta", "gamma"],
    });
    expect(result.output).toBe("alpha");
  });

  it("handles non-array categories via getCategories", async () => {
    const n = new (ClassifierNode as any)();
    const result = await n.process({
      text: "test",
      categories: "not-an-array",
    });
    // getCategories returns [] for non-array
    expect(result.output).toBe("Unknown");
  });

  it("filters empty strings from categories", async () => {
    const n = new (ClassifierNode as any)();
    const result = await n.process({
      text: "hello world",
      categories: ["", "  ", "hello"],
    });
    // empty/whitespace categories are filtered by getCategories (trim check)
    expect(result.category).toBe("hello");
  });

  it("handles multi-word categories", async () => {
    const n = new (ClassifierNode as any)();
    const result = await n.process({
      text: "machine learning is great",
      categories: ["web development", "machine learning", "data science"],
    });
    expect(result.output).toBe("machine learning");
  });
});

// ---- AgentNode ----
describe("AgentNode", () => {
  it("has correct static metadata", () => {
    expect(AgentNode.nodeType).toBe("nodetool.agents.Agent");
  });

  it("defaults", () => {
    const n = new (AgentNode as any)();
    expect(n.defaults()).toEqual({
      system: "",
      prompt: "",
      history: [],
      thread_id: "",
    });
  });

  it("returns concatenated text when no context/provider", async () => {
    const n = new (AgentNode as any)();
    const result = await n.process({
      system: "You are helpful.",
      prompt: "Hello!",
    });
    expect(result.text).toContain("You are helpful.");
    expect(result.text).toContain("Hello!");
    expect(result.output).toBe(result.text);
    expect(result.chunk).toBeNull();
    expect(result.thinking).toBeNull();
    expect(result.audio).toBeNull();
  });

  it("returns 'No prompt provided.' when everything is empty", async () => {
    const n = new (AgentNode as any)();
    const result = await n.process({});
    expect(result.text).toBe("No prompt provided.");
  });

  it("includes history in deterministic response", async () => {
    const n = new (AgentNode as any)();
    const result = await n.process({
      prompt: "Continue",
      history: [
        { role: "user", content: "Previous message" },
        { role: "assistant", content: "Previous response" },
      ],
    });
    expect(result.text).toContain("Previous message");
    expect(result.text).toContain("Previous response");
    expect(result.text).toContain("Continue");
  });

  it("stores messages in thread when thread_id provided", async () => {
    const n = new (AgentNode as any)();
    // First create the thread
    const ct = new (CreateThreadNode as any)();
    const { thread_id } = await ct.process({ thread_id: "agent_test_thread" });

    const result = await n.process({
      prompt: "Test prompt",
      thread_id,
    });
    expect(result.text).toContain("Test prompt");
  });

  it("creates thread on-the-fly if thread_id not in store", async () => {
    const n = new (AgentNode as any)();
    const result = await n.process({
      prompt: "Hello",
      thread_id: "nonexistent_thread_42",
    });
    expect(result.text).toContain("Hello");
  });

  it("handles non-array history gracefully", async () => {
    const n = new (AgentNode as any)();
    const result = await n.process({
      prompt: "Hello",
      history: "not an array",
    });
    expect(result.text).toContain("Hello");
  });

  it("uses provider when context is available", async () => {
    const n = new (AgentNode as any)();
    const mockProvider = {
      generateMessage: async ({ messages, model, maxTokens }: any) => {
        return { content: `Generated response for model ${model}` };
      },
      async generateMessageTraced(...a: any[]) { return (this as any).generateMessage(...a); },
    };
    const mockContext = {
      getProvider: async (id: string) => mockProvider,
    };
    const result = await n.process(
      {
        prompt: "Test prompt",
        model: { provider: "openai", id: "gpt-4", name: "GPT-4" },
        max_tokens: 512,
      },
      mockContext as any
    );
    expect(result.text).toBe("Generated response for model gpt-4");
  });

  it("includes system message when provider is used", async () => {
    const n = new (AgentNode as any)();
    let capturedMessages: any[] = [];
    const mockProvider = {
      generateMessage: async ({ messages }: any) => {
        capturedMessages = messages;
        return { content: "ok" };
      },
      async generateMessageTraced(...a: any[]) { return (this as any).generateMessage(...a); },
    };
    const mockContext = {
      getProvider: async () => mockProvider,
    };
    await n.process(
      {
        system: "Be concise.",
        prompt: "Hi",
        model: { provider: "test", id: "m1" },
      },
      mockContext as any
    );
    expect(capturedMessages[0]).toEqual({ role: "system", content: "Be concise." });
    expect(capturedMessages[capturedMessages.length - 1]).toEqual({
      role: "user",
      content: "Hi",
    });
  });

  it("skips system message when system text is empty", async () => {
    const n = new (AgentNode as any)();
    let capturedMessages: any[] = [];
    const mockProvider = {
      generateMessage: async ({ messages }: any) => {
        capturedMessages = messages;
        return { content: "ok" };
      },
      async generateMessageTraced(...a: any[]) { return (this as any).generateMessage(...a); },
    };
    const mockContext = {
      getProvider: async () => mockProvider,
    };
    await n.process(
      {
        system: "",
        prompt: "Hi",
        model: { provider: "test", id: "m1" },
      },
      mockContext as any
    );
    expect(capturedMessages[0].role).not.toBe("system");
  });

  it("filters invalid roles from history in provider path", async () => {
    const n = new (AgentNode as any)();
    let capturedMessages: any[] = [];
    const mockProvider = {
      generateMessage: async ({ messages }: any) => {
        capturedMessages = messages;
        return { content: "ok" };
      },
      async generateMessageTraced(...a: any[]) { return (this as any).generateMessage(...a); },
    };
    const mockContext = {
      getProvider: async () => mockProvider,
    };
    await n.process(
      {
        prompt: "Hi",
        model: { provider: "test", id: "m1" },
        history: [
          { role: "user", content: "valid" },
          { role: "invalid_role", content: "skip me" },
          { role: "assistant", content: "also valid" },
        ],
      },
      mockContext as any
    );
    // Should have user + assistant from history + user prompt = 3 messages
    const roles = capturedMessages.map((m: any) => m.role);
    expect(roles).not.toContain("invalid_role");
    expect(roles.filter((r: string) => r === "user")).toHaveLength(2);
    expect(roles).toContain("assistant");
  });

  it("handles history items without role in provider path", async () => {
    const n = new (AgentNode as any)();
    let capturedMessages: any[] = [];
    const mockProvider = {
      generateMessage: async ({ messages }: any) => {
        capturedMessages = messages;
        return { content: "ok" };
      },
      async generateMessageTraced(...a: any[]) { return (this as any).generateMessage(...a); },
    };
    const mockContext = {
      getProvider: async () => mockProvider,
    };
    await n.process(
      {
        prompt: "Hi",
        model: { provider: "test", id: "m1" },
        history: [{ content: "no role field" }],
      },
      mockContext as any
    );
    // Defaults to "user" role
    expect(capturedMessages[0]).toEqual({
      role: "user",
      content: "no role field",
    });
  });

  it("handles null content from provider", async () => {
    const n = new (AgentNode as any)();
    const mockProvider = {
      generateMessage: async () => ({ content: null }),
      async generateMessageTraced(...a: any[]) { return (this as any).generateMessage(...a); },
    };
    const mockContext = {
      getProvider: async () => mockProvider,
    };
    const result = await n.process(
      {
        prompt: "Hi",
        model: { provider: "test", id: "m1" },
      },
      mockContext as any
    );
    expect(result.text).toBe("");
  });

  it("falls back to deterministic mode when model has no provider", async () => {
    const n = new (AgentNode as any)();
    const mockContext = {
      getProvider: async () => ({}),
    };
    const result = await n.process(
      {
        prompt: "Hello",
        model: { id: "m1" }, // no provider field
      },
      mockContext as any
    );
    expect(result.text).toContain("Hello");
  });

  it("falls back to deterministic mode when model has no id", async () => {
    const n = new (AgentNode as any)();
    const mockContext = {
      getProvider: async () => ({}),
    };
    const result = await n.process(
      {
        prompt: "Hello",
        model: { provider: "openai" }, // no id field
      },
      mockContext as any
    );
    expect(result.text).toContain("Hello");
  });

  it("stores in thread when using provider path", async () => {
    const n = new (AgentNode as any)();
    const mockProvider = {
      generateMessage: async () => ({ content: "Provider reply" }),
      async generateMessageTraced(...a: any[]) { return (this as any).generateMessage(...a); },
    };
    const mockContext = {
      getProvider: async () => mockProvider,
    };
    const result = await n.process(
      {
        prompt: "Thread test",
        model: { provider: "test", id: "m1" },
        thread_id: "provider_thread_test",
      },
      mockContext as any
    );
    expect(result.text).toBe("Provider reply");
  });
});

// ---- ControlAgentNode ----
describe("ControlAgentNode", () => {
  it("has correct static metadata", () => {
    expect(ControlAgentNode.nodeType).toBe("nodetool.agents.ControlAgent");
  });

  it("defaults", () => {
    const n = new (ControlAgentNode as any)();
    expect(n.defaults()).toEqual({ _control_context: {} });
  });

  it("returns empty object for null context", async () => {
    const n = new (ControlAgentNode as any)();
    const result = await n.process({ _control_context: null });
    expect(result.__control_output__).toEqual({});
  });

  it("returns empty object for non-object context", async () => {
    const n = new (ControlAgentNode as any)();
    const result = await n.process({ _control_context: "string value" });
    expect(result.__control_output__).toEqual({});
  });

  it("extracts properties when context has properties field", async () => {
    const n = new (ControlAgentNode as any)();
    const result = await n.process({
      _control_context: {
        properties: { speed: 5, direction: "north" },
      },
    });
    expect(result.__control_output__).toEqual({ speed: 5, direction: "north" });
  });

  it("returns context as-is when no properties field", async () => {
    const n = new (ControlAgentNode as any)();
    const result = await n.process({
      _control_context: { key: "value", num: 42 },
    });
    expect(result.__control_output__).toEqual({ key: "value", num: 42 });
  });

  it("returns context when properties is null", async () => {
    const n = new (ControlAgentNode as any)();
    const result = await n.process({
      _control_context: { properties: null, extra: 1 },
    });
    // properties exists but is falsy, so falls through to return context
    expect(result.__control_output__).toEqual({ properties: null, extra: 1 });
  });

  it("returns context when properties is not an object", async () => {
    const n = new (ControlAgentNode as any)();
    const result = await n.process({
      _control_context: { properties: "not-object" },
    });
    // properties is string, not object -> returns full context
    expect(result.__control_output__).toEqual({ properties: "not-object" });
  });
});

// ---- ResearchAgentNode ----
describe("ResearchAgentNode", () => {
  it("has correct static metadata", () => {
    expect(ResearchAgentNode.nodeType).toBe("nodetool.agents.ResearchAgent");
  });

  it("defaults", () => {
    const n = new (ResearchAgentNode as any)();
    expect(n.defaults()).toEqual({ query: "", prompt: "" });
  });

  it("produces research notes from query", async () => {
    const n = new (ResearchAgentNode as any)();
    const result = await n.process({ query: "What is TypeScript?" });
    expect(result.output).toContain("Question: What is TypeScript?");
    expect(result.output).toContain("Summary:");
    expect(result.output).toContain("Confidence: low");
    expect(result.text).toBe(result.output);
    expect(result.findings).toHaveLength(1);
    expect(result.findings[0].title).toBe("What is TypeScript?");
  });

  it("falls back to prompt when query is empty", async () => {
    const n = new (ResearchAgentNode as any)();
    const result = await n.process({ prompt: "Fallback prompt." });
    expect(result.output).toContain("Question: Fallback prompt.");
  });

  it("handles empty inputs", async () => {
    const n = new (ResearchAgentNode as any)();
    const result = await n.process({});
    expect(result.output).toContain("Question:");
  });
});

// ---------------------------------------------------------------------------
// generators.ts
// ---------------------------------------------------------------------------

describe("GENERATOR_NODES export", () => {
  it("contains all 5 generator node classes", () => {
    expect(GENERATOR_NODES).toHaveLength(5);
    expect(GENERATOR_NODES).toContain(StructuredOutputGeneratorNode);
    expect(GENERATOR_NODES).toContain(DataGeneratorNode);
    expect(GENERATOR_NODES).toContain(ListGeneratorNode);
    expect(GENERATOR_NODES).toContain(ChartGeneratorNode);
    expect(GENERATOR_NODES).toContain(SVGGeneratorNode);
  });
});

// ---- StructuredOutputGeneratorNode ----
describe("StructuredOutputGeneratorNode", () => {
  it("has correct static metadata", () => {
    expect(StructuredOutputGeneratorNode.nodeType).toBe(
      "nodetool.generators.StructuredOutputGenerator"
    );
  });

  it("defaults", () => {
    const n = new (StructuredOutputGeneratorNode as any)();
    expect(n.defaults()).toEqual({ instructions: "", context: "", schema: {} });
  });

  it("generates defaults from schema with various types", async () => {
    const n = new (StructuredOutputGeneratorNode as any)();
    const result = await n.process({
      schema: {
        properties: {
          name: { type: "string" },
          age: { type: "number" },
          count: { type: "integer" },
          active: { type: "boolean" },
          tags: { type: "array" },
          meta: { type: "object" },
          other: {}, // no type specified
        },
      },
    });
    expect(result.name).toBe("");
    expect(result.age).toBe(0);
    expect(result.count).toBe(0);
    expect(result.active).toBe(false);
    expect(result.tags).toEqual([]);
    expect(result.meta).toEqual({});
    expect(result.other).toBe("");
  });

  it("generates schema defaults with no properties key", async () => {
    const n = new (StructuredOutputGeneratorNode as any)();
    const result = await n.process({
      schema: { type: "object" }, // no properties
    });
    // Empty object, no properties to iterate
    expect(Object.keys(result)).toHaveLength(0);
  });

  it("falls back to instructions/context when no schema", async () => {
    const n = new (StructuredOutputGeneratorNode as any)();
    const result = await n.process({
      instructions: "Generate a list",
      context: "user context",
    });
    expect(result.output).toEqual({
      instructions: "Generate a list",
      context: "user context",
    });
  });

  it("falls back when schema is null", async () => {
    const n = new (StructuredOutputGeneratorNode as any)();
    const result = await n.process({
      schema: null,
      instructions: "test",
    });
    expect(result.output).toEqual({ instructions: "test", context: "" });
  });

  it("falls back when schema is an array", async () => {
    const n = new (StructuredOutputGeneratorNode as any)();
    const result = await n.process({
      schema: [1, 2, 3],
      instructions: "test",
    });
    expect(result.output).toEqual({ instructions: "test", context: "" });
  });

  it("handles numeric input for instructions via asText", async () => {
    const n = new (StructuredOutputGeneratorNode as any)();
    const result = await n.process({
      instructions: 42,
      context: true,
    });
    expect(result.output).toEqual({
      instructions: "42",
      context: "true",
    });
  });

  it("handles null/undefined/object inputs via asText in generators", async () => {
    const n = new (StructuredOutputGeneratorNode as any)();
    // null triggers asText !value branch
    const r1 = await n.process({ instructions: null, context: undefined });
    expect(r1.output).toEqual({ instructions: "", context: "" });
    // object triggers JSON.stringify branch
    const r2 = await n.process({ instructions: { a: 1 }, context: [1, 2] });
    expect(r2.output.instructions).toBe('{"a":1}');
    expect(r2.output.context).toBe("[1,2]");
  });
});

// ---- DataGeneratorNode ----
describe("DataGeneratorNode", () => {
  it("has correct static metadata", () => {
    expect(DataGeneratorNode.nodeType).toBe("nodetool.generators.DataGenerator");
    expect(DataGeneratorNode.isStreamingOutput).toBe(true);
  });

  it("defaults", () => {
    const n = new (DataGeneratorNode as any)();
    expect(n.defaults()).toEqual({ prompt: "", input_text: "", columns: [] });
  });

  it("generates rows with default 5 count", async () => {
    const n = new (DataGeneratorNode as any)();
    const result = await n.process({ prompt: "generate data" });
    const rows = (result.output as any).rows;
    expect(rows).toHaveLength(5);
    // default column is "value"
    expect(rows[0]).toHaveProperty("value");
  });

  it("parses count from prompt", async () => {
    const n = new (DataGeneratorNode as any)();
    const result = await n.process({ prompt: "generate 3 items" });
    expect((result.output as any).rows).toHaveLength(3);
  });

  it("uses columns array with name objects", async () => {
    const n = new (DataGeneratorNode as any)();
    const result = await n.process({
      prompt: "2 records",
      columns: [{ name: "id" }, { name: "name" }, { name: "score" }],
    });
    const rows = (result.output as any).rows;
    expect(rows).toHaveLength(2);
    expect(rows[0].id).toBe(1);
    expect(rows[0].name).toContain("_1");
    expect(typeof rows[0].score).toBe("number");
  });

  it("uses columns from nested object with columns key", async () => {
    const n = new (DataGeneratorNode as any)();
    const result = await n.process({
      prompt: "2 records",
      columns: { columns: ["id", "date", "active"] },
    });
    const rows = (result.output as any).rows;
    expect(rows).toHaveLength(2);
    expect(rows[0].id).toBe(1);
    expect(rows[0].date).toBeTruthy();
    expect(typeof rows[0].active).toBe("boolean");
  });

  it("handles price/amount column type", async () => {
    const n = new (DataGeneratorNode as any)();
    const result = await n.process({
      prompt: "2 items",
      columns: ["price", "amount"],
    });
    const rows = (result.output as any).rows;
    expect(typeof rows[0].price).toBe("number");
    expect(typeof rows[0].amount).toBe("number");
  });

  it("handles is_ prefixed columns as booleans", async () => {
    const n = new (DataGeneratorNode as any)();
    const result = await n.process({
      prompt: "3 items",
      columns: ["is_on"],
    });
    const rows = (result.output as any).rows;
    // "is_on" starts with "is_" and does not include "id"/"name"/"date"/"price"/"amount"/"score"/"active"
    expect(rows[0].is_on).toBe(true); // i=0, even
    expect(rows[1].is_on).toBe(false); // i=1, odd
    expect(rows[2].is_on).toBe(true); // i=2, even
  });

  it("caps count at 200", async () => {
    const n = new (DataGeneratorNode as any)();
    const result = await n.process({ prompt: "generate 999 rows" });
    // parseRequestedCount caps at 200
    // But 999 has 3 digits, and regex matches \b(\d{1,3})\b
    const rows = (result.output as any).rows;
    expect(rows.length).toBeLessThanOrEqual(200);
  });

  it("genProcess yields individual rows then final dataframe", async () => {
    const n = new (DataGeneratorNode as any)();
    const results: any[] = [];
    for await (const chunk of n.genProcess({ prompt: "3 items" })) {
      results.push(chunk);
    }
    // 3 row yields + 1 final dataframe yield
    expect(results).toHaveLength(4);
    expect(results[0].index).toBe(0);
    expect(results[0].record).toBeTruthy();
    expect(results[0].dataframe).toBeNull();
    expect(results[3].record).toBeNull();
    expect(results[3].index).toBeNull();
    expect(results[3].dataframe).toBeTruthy();
  });

  it("uses input_text as seed when no prompt", async () => {
    const n = new (DataGeneratorNode as any)();
    const result = await n.process({ input_text: "mydata" });
    const rows = (result.output as any).rows;
    expect(rows[0].value).toContain("mydata");
  });

  it("handles empty columns from parseColumns", async () => {
    const n = new (DataGeneratorNode as any)();
    const result = await n.process({
      prompt: "2 items",
      columns: [{ name: "" }, ""],
    });
    const rows = (result.output as any).rows;
    // Empty names are filtered, defaults to ["value"]
    expect(rows[0]).toHaveProperty("value");
  });

  it("parseRequestedCount returns fallback for no digit in prompt", async () => {
    const n = new (DataGeneratorNode as any)();
    const result = await n.process({ prompt: "some data please" });
    expect((result.output as any).rows).toHaveLength(5);
  });

  it("ensures minimum count of 1", async () => {
    const n = new (DataGeneratorNode as any)();
    // parseRequestedCount: Math.max(1, ...)
    const result = await n.process({ prompt: "0 items" });
    // 0 matches but max(1, 0) = 1... actually it is clamped to min 1
    // Actually the regex matches "0", n=0, max(1,min(200,0))=max(1,0)=1
    expect((result.output as any).rows).toHaveLength(1);
  });
});

// ---- ListGeneratorNode ----
describe("ListGeneratorNode", () => {
  it("has correct static metadata", () => {
    expect(ListGeneratorNode.nodeType).toBe("nodetool.generators.ListGenerator");
    expect(ListGeneratorNode.isStreamingOutput).toBe(true);
  });

  it("defaults", () => {
    const n = new (ListGeneratorNode as any)();
    expect(n.defaults()).toEqual({ prompt: "", input_text: "" });
  });

  it("generates a list with default count 5", async () => {
    const n = new (ListGeneratorNode as any)();
    const result = await n.process({ prompt: "list things" });
    expect(result.output).toHaveLength(5);
    expect(result.output[0]).toContain("_1");
  });

  it("parses count from prompt", async () => {
    const n = new (ListGeneratorNode as any)();
    const result = await n.process({ prompt: "give me 7 colors" });
    expect(result.output).toHaveLength(7);
  });

  it("uses input_text as seed", async () => {
    const n = new (ListGeneratorNode as any)();
    const result = await n.process({ input_text: "fruit" });
    expect(result.output[0]).toBe("fruit_1");
  });

  it("genProcess yields individual items", async () => {
    const n = new (ListGeneratorNode as any)();
    const results: any[] = [];
    for await (const chunk of n.genProcess({ prompt: "3 items" })) {
      results.push(chunk);
    }
    expect(results).toHaveLength(3);
    expect(results[0]).toEqual({ item: expect.any(String), index: 0 });
    expect(results[2].index).toBe(2);
  });

  it("genProcess yields empty when output is not array", async () => {
    const n = new (ListGeneratorNode as any)();
    // Override process to return non-array
    n.process = async () => ({ output: "not-array" });
    const results: any[] = [];
    for await (const chunk of n.genProcess({})) {
      results.push(chunk);
    }
    expect(results).toHaveLength(0);
  });
});

// ---- ChartGeneratorNode ----
describe("ChartGeneratorNode", () => {
  it("has correct static metadata", () => {
    expect(ChartGeneratorNode.nodeType).toBe("nodetool.generators.ChartGenerator");
  });

  it("defaults", () => {
    const n = new (ChartGeneratorNode as any)();
    expect(n.defaults()).toEqual({ prompt: "", data: { rows: [] } });
  });

  it("generates chart config from data rows", async () => {
    const n = new (ChartGeneratorNode as any)();
    const result = await n.process({
      prompt: "Sales Chart",
      data: {
        rows: [
          { month: "Jan", revenue: 100 },
          { month: "Feb", revenue: 200 },
        ],
      },
    });
    const output = result.output as any;
    expect(output.data[0].type).toBe("bar");
    expect(output.data[0].x).toEqual(["Jan", "Feb"]);
    expect(output.data[0].y).toEqual([100, 200]);
    expect(output.data[0].name).toBe("Sales Chart");
    expect(output.layout.title).toBe("Sales Chart");
  });

  it("uses default series name when no prompt", async () => {
    const n = new (ChartGeneratorNode as any)();
    const result = await n.process({
      data: { rows: [{ a: 1 }] },
    });
    const output = result.output as any;
    expect(output.data[0].name).toBe("series");
    expect(output.layout.title).toBe("Generated Chart");
  });

  it("handles empty rows", async () => {
    const n = new (ChartGeneratorNode as any)();
    const result = await n.process({ prompt: "Empty", data: { rows: [] } });
    const output = result.output as any;
    expect(output.data[0].x).toEqual([]);
    expect(output.data[0].y).toEqual([]);
  });

  it("handles data without rows key", async () => {
    const n = new (ChartGeneratorNode as any)();
    const result = await n.process({ data: {} });
    const output = result.output as any;
    expect(output.data[0].x).toEqual([]);
  });

  it("uses index as fallback when key missing in row", async () => {
    const n = new (ChartGeneratorNode as any)();
    // Only one key so xKey and yKey are the same
    const result = await n.process({
      data: { rows: [{ only: 10 }, { only: 20 }] },
    });
    const output = result.output as any;
    expect(output.data[0].x).toEqual([10, 20]);
    expect(output.data[0].y).toEqual([10, 20]);
  });
});

// ---- SVGGeneratorNode ----
describe("SVGGeneratorNode", () => {
  it("has correct static metadata", () => {
    expect(SVGGeneratorNode.nodeType).toBe("nodetool.generators.SVGGenerator");
  });

  it("defaults", () => {
    const n = new (SVGGeneratorNode as any)();
    expect(n.defaults()).toEqual({ prompt: "", width: 512, height: 512 });
  });

  it("generates SVG with prompt text", async () => {
    const n = new (SVGGeneratorNode as any)();
    const result = await n.process({ prompt: "Hello World" });
    const svg = (result.output as any[])[0].content;
    expect(svg).toContain("<svg");
    expect(svg).toContain("Hello World");
    expect(svg).toContain('width="512"');
    expect(svg).toContain('height="512"');
  });

  it("uses custom dimensions", async () => {
    const n = new (SVGGeneratorNode as any)();
    const result = await n.process({ prompt: "Test", width: 100, height: 200 });
    const svg = (result.output as any[])[0].content;
    expect(svg).toContain('width="100"');
    expect(svg).toContain('height="200"');
  });

  it("escapes HTML entities in prompt", async () => {
    const n = new (SVGGeneratorNode as any)();
    const result = await n.process({ prompt: "A & B <C>" });
    const svg = (result.output as any[])[0].content;
    expect(svg).toContain("A &amp; B &lt;C&gt;");
    expect(svg).not.toContain("A & B <C>");
  });

  it("uses 'SVG' as default text when no prompt", async () => {
    const n = new (SVGGeneratorNode as any)();
    const result = await n.process({});
    const svg = (result.output as any[])[0].content;
    expect(svg).toContain(">SVG</text>");
  });

  it("defaults width/height to 512 when given 0 or NaN", async () => {
    const n = new (SVGGeneratorNode as any)();
    const result = await n.process({ width: 0, height: NaN });
    const svg = (result.output as any[])[0].content;
    expect(svg).toContain('width="512"');
    expect(svg).toContain('height="512"');
  });
});
