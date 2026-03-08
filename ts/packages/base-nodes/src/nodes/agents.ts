import { BaseNode } from "@nodetool/node-sdk";
import type { ProcessingContext } from "@nodetool/runtime";

type MessagePart = { type?: string; text?: string };
type ProviderMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content?: string | null;
};
type MessageLike = {
  id?: string;
  thread_id?: string;
  role?: string;
  content?: string | MessagePart[];
};

type ThreadLike = { id: string; title: string; messages: MessageLike[] };
type LanguageModelLike = { provider?: string; id?: string; name?: string };
type ProviderLike = {
  generateMessage(args: {
    messages: ProviderMessage[];
    model: string;
    maxTokens?: number;
  }): Promise<{ content?: string | null }>;
  generateMessageTraced(args: {
    messages: ProviderMessage[];
    model: string;
    maxTokens?: number;
  }): Promise<{ content?: string | null }>;
};

const THREAD_STORE = new Map<string, ThreadLike>();

function asText(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (!value) return "";
  if (Array.isArray(value)) return value.map(asText).join(" ");
  if (typeof value === "object") {
    const msg = value as MessageLike;
    if (typeof msg.content === "string") return msg.content;
    if (Array.isArray(msg.content)) {
      return msg.content
        .map((part) => (part && part.type === "text" ? part.text ?? "" : ""))
        .join(" ")
        .trim();
    }
    return JSON.stringify(value);
  }
  return "";
}

function summarize(text: string, maxSentences: number): string {
  const parts = text
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  if (parts.length === 0) return "";
  return parts.slice(0, Math.max(1, maxSentences)).join(" ");
}

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .map((t) => t.trim())
    .filter((t) => t.length > 0);
}

function extractJson(text: string): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(text);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : null;
  } catch {
    const start = text.indexOf("{");
    const end = text.lastIndexOf("}");
    if (start >= 0 && end > start) {
      try {
        const parsed = JSON.parse(text.slice(start, end + 1));
        return parsed && typeof parsed === "object" && !Array.isArray(parsed)
          ? (parsed as Record<string, unknown>)
          : null;
      } catch {
        return null;
      }
    }
    return null;
  }
}

function makeThreadId(): string {
  return `thread_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
}

function getCategories(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((v) => String(v)).filter((v) => v.trim().length > 0);
}

export class SummarizerNode extends BaseNode {
  static readonly nodeType = "nodetool.agents.Summarizer";
  static readonly title = "Summarizer";
  static readonly description = "Create a concise summary from text.";

  defaults() {
    return { text: "", max_sentences: 3 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = asText(inputs.text ?? this._props.text ?? "");
    const maxSentences = Number(inputs.max_sentences ?? this._props.max_sentences ?? 3);
    const result = summarize(text, Number.isFinite(maxSentences) ? maxSentences : 3);
    return { text: result, output: result };
  }
}

export class CreateThreadNode extends BaseNode {
  static readonly nodeType = "nodetool.agents.CreateThread";
  static readonly title = "Create Thread";
  static readonly description = "Create or reuse an in-memory conversation thread.";

  defaults() {
    return { title: "Agent Conversation", thread_id: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const requested = String(inputs.thread_id ?? this._props.thread_id ?? "").trim();
    if (requested) {
      if (!THREAD_STORE.has(requested)) {
        THREAD_STORE.set(requested, {
          id: requested,
          title: String(inputs.title ?? this._props.title ?? "Agent Conversation"),
          messages: [],
        });
      }
      return { thread_id: requested };
    }

    const id = makeThreadId();
    THREAD_STORE.set(id, {
      id,
      title: String(inputs.title ?? this._props.title ?? "Agent Conversation"),
      messages: [],
    });
    return { thread_id: id };
  }
}

export class ExtractorNode extends BaseNode {
  static readonly nodeType = "nodetool.agents.Extractor";
  static readonly title = "Extractor";
  static readonly description = "Extract structured JSON data from text.";

  defaults() {
    return { text: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = asText(inputs.text ?? this._props.text ?? "");
    const parsed = extractJson(text);
    if (parsed) return parsed;
    return { output: text };
  }
}

export class ClassifierNode extends BaseNode {
  static readonly nodeType = "nodetool.agents.Classifier";
  static readonly title = "Classifier";
  static readonly description = "Classify text to the closest category.";

  defaults() {
    return { text: "", categories: [] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = asText(inputs.text ?? this._props.text ?? "");
    const categories = getCategories(inputs.categories ?? this._props.categories);
    if (categories.length === 0) {
      return { output: "Unknown", category: "Unknown" };
    }

    const tokens = tokenize(text);
    let best = categories[0];
    let bestScore = -1;
    for (const category of categories) {
      const catTokens = tokenize(category);
      let score = 0;
      for (const token of catTokens) {
        if (tokens.includes(token)) score += 1;
      }
      if (score > bestScore) {
        best = category;
        bestScore = score;
      }
    }
    return { output: best, category: best };
  }
}

export class AgentNode extends BaseNode {
  static readonly nodeType = "nodetool.agents.Agent";
  static readonly title = "Agent";
  static readonly description = "Generate a deterministic text response.";

  defaults() {
    return {
      system: "",
      prompt: "",
      history: [],
      thread_id: "",
    };
  }

  async process(
    inputs: Record<string, unknown>,
    context?: ProcessingContext
  ): Promise<Record<string, unknown>> {
    const prompt = asText(inputs.prompt ?? this._props.prompt ?? "");
    const system = asText(inputs.system ?? this._props.system ?? "");
    const history = Array.isArray(inputs.history ?? this._props.history)
      ? (inputs.history ?? this._props.history) as unknown[]
      : [];
    const model = ((inputs.model ?? this._props.model ?? {}) as LanguageModelLike) ?? {};
    const providerId = typeof model.provider === "string" ? model.provider : "";
    const modelId = typeof model.id === "string" ? model.id : "";

    let response: string;
    const providerSupported =
      !!context && typeof context.getProvider === "function" && providerId && modelId;

    if (providerSupported) {
      const provider = (await context.getProvider(providerId)) as ProviderLike;
      const messages: ProviderMessage[] = [];

      if (system.trim().length > 0) {
        messages.push({ role: "system", content: system });
      }
      for (const item of history) {
        const msg = item as { role?: unknown; content?: unknown };
        const role =
          msg && typeof msg === "object" && typeof msg.role === "string"
            ? msg.role
            : "user";
        if (!["system", "user", "assistant", "tool"].includes(role)) {
          continue;
        }
        messages.push({
          role: role as ProviderMessage["role"],
          content: asText(msg.content),
        });
      }
      messages.push({ role: "user", content: prompt });

      const generated = await provider.generateMessageTraced({
        model: modelId,
        messages,
        maxTokens: Number(inputs.max_tokens ?? this._props.max_tokens ?? 1024),
      });
      response = asText(generated.content ?? "");
    } else {
      const prior = history
        .map((h) => asText(h))
        .filter((v) => v.length > 0)
        .join("\n");
      const text = [system, prior, prompt]
        .map((s) => s.trim())
        .filter((s) => s.length > 0)
        .join("\n")
        .trim();
      response = text.length > 0 ? text : "No prompt provided.";
    }

    const threadId = String(inputs.thread_id ?? this._props.thread_id ?? "").trim();
    if (threadId) {
      const thread = THREAD_STORE.get(threadId) ?? {
        id: threadId,
        title: "Agent Conversation",
        messages: [],
      };
      thread.messages.push({
        role: "user",
        content: prompt,
      });
      thread.messages.push({
        role: "assistant",
        content: response,
      });
      THREAD_STORE.set(threadId, thread);
    }

    return { text: response, output: response, chunk: null, thinking: null, audio: null };
  }
}

export class ControlAgentNode extends BaseNode {
  static readonly nodeType = "nodetool.agents.ControlAgent";
  static readonly title = "Control Agent";
  static readonly description = "Generate control parameters from context.";

  defaults() {
    return { _control_context: {} };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const value = inputs._control_context ?? this._props._control_context ?? {};
    if (!value || typeof value !== "object") return { __control_output__: {} };
    const context = value as Record<string, unknown>;
    if ("properties" in context && context.properties && typeof context.properties === "object") {
      return { __control_output__: context.properties as Record<string, unknown> };
    }
    return { __control_output__: context };
  }
}

export class ResearchAgentNode extends BaseNode {
  static readonly nodeType = "nodetool.agents.ResearchAgent";
  static readonly title = "Research Agent";
  static readonly description = "Produce lightweight research notes for a query.";

  defaults() {
    return { query: "", prompt: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const query = asText(inputs.query ?? this._props.query ?? this._props.prompt ?? inputs.prompt ?? "");
    const summary = summarize(query, 2);
    const notes = [
      `Question: ${query}`,
      `Summary: ${summary}`,
      "Confidence: low (offline placeholder implementation)",
    ];
    return {
      output: notes.join("\n"),
      text: notes.join("\n"),
      findings: [{ title: query, summary }],
    };
  }
}

export const AGENT_NODES = [
  SummarizerNode,
  CreateThreadNode,
  ExtractorNode,
  ClassifierNode,
  AgentNode,
  ControlAgentNode,
  ResearchAgentNode,
] as const;
