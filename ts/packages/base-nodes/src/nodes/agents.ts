import { createLogger } from "@nodetool/config";
import { BaseNode } from "@nodetool/node-sdk";
import type {
  BaseProvider,
  Message,
  MessageAudioContent,
  MessageContent,
  MessageImageContent,
  ProcessingContext,
  ProviderStreamItem,
  ToolCall,
} from "@nodetool/runtime";
import type { Chunk } from "@nodetool/protocol";

type MessagePart = { type?: string; text?: string };
type ThreadLike = { id: string; title: string; messages: Message[] };
type LanguageModelLike = { provider?: string; id?: string; name?: string };
type ToolLike = {
  name: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
  process?: (context: ProcessingContext, params: Record<string, unknown>) => Promise<unknown>;
  toProviderTool?: () => { name: string; description?: string; inputSchema?: Record<string, unknown> };
};

const THREAD_STORE = new Map<string, ThreadLike>();
const log = createLogger("nodetool.base-nodes.agents");
const DEFAULT_SYSTEM_PROMPT = "You are a friendly assistant";
const EXTRACTOR_SYSTEM_PROMPT = [
  "You are a precise structured data extractor.",
  "Return exactly one JSON object and no additional prose.",
  "Use only information present in the input.",
].join(" ");
const CLASSIFIER_SYSTEM_PROMPT = [
  "You are a precise classifier.",
  "Choose exactly one category from the allowed list.",
  'Return only JSON matching {"category":"<allowed-category>"} with no extra text.',
].join(" ");
const SUMMARIZER_SYSTEM_PROMPT =
  "You are an expert summarizer. Produce a concise, accurate summary.";
const CONTROL_AGENT_SYSTEM_PROMPT = [
  "You are a control flow agent that determines parameter values for downstream nodes.",
  "Return only a JSON object containing parameter names and values.",
].join(" ");
const RESEARCH_AGENT_SYSTEM_PROMPT = [
  "You are a research assistant.",
  "Synthesize a concise answer and key findings from the objective.",
  "Return JSON only.",
].join(" ");

function asText(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (!value) return "";
  if (Array.isArray(value)) return value.map(asText).join(" ");
  if (typeof value === "object") {
    const msg = value as { content?: string | MessagePart[] };
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

function getModelConfig(
  inputs: Record<string, unknown>,
  props: Record<string, unknown>
): { providerId: string; modelId: string } {
  const model = ((inputs.model ?? props.model ?? {}) as LanguageModelLike) ?? {};
  return {
    providerId: typeof model.provider === "string" ? model.provider : "",
    modelId: typeof model.id === "string" ? model.id : "",
  };
}

function hasProviderSupport(
  context: ProcessingContext | undefined,
  providerId: string,
  modelId: string
): context is ProcessingContext & { getProvider(providerId: string): Promise<BaseProvider> } {
  return !!context && typeof context.getProvider === "function" && !!providerId && !!modelId;
}

async function generateProviderMessage(
  provider: BaseProvider,
  args: {
    messages: Message[];
    model: string;
    maxTokens?: number;
    responseFormat?: Record<string, unknown>;
  }
): Promise<string> {
  const call =
    typeof provider.generateMessageTraced === "function"
      ? provider.generateMessageTraced.bind(provider)
      : provider.generateMessage.bind(provider);
  const result = await call(args);
  return messageContentText(result.content);
}

function normalizeProviderStreamItem(item: ProviderStreamItem): ProviderStreamItem {
  if (!item || typeof item !== "object" || !("type" in item) || (item as Chunk).type !== "chunk") {
    return item;
  }

  const chunk = item as Chunk;
  if (typeof chunk.content_type === "string" && chunk.content_type.length > 0) {
    return chunk;
  }

  return {
    ...chunk,
    content_type: "text",
  } as Chunk;
}

async function* streamProviderMessages(
  provider: BaseProvider,
  args: Parameters<BaseProvider["generateMessages"]>[0]
): AsyncGenerator<ProviderStreamItem> {
  const request = {
    ...args,
    messages: [...args.messages],
    tools: args.tools ? [...args.tools] : undefined,
  };
  if (typeof provider.generateMessagesTraced === "function") {
    for await (const item of provider.generateMessagesTraced(request)) {
      yield normalizeProviderStreamItem(item);
    }
    return;
  }
  if (typeof provider.generateMessages === "function") {
    for await (const item of provider.generateMessages(request)) {
      yield normalizeProviderStreamItem(item);
    }
    return;
  }
  const result = await provider.generateMessage(request);
  const content = messageContentText(result.content);
  if (content || (result.toolCalls?.length ?? 0) === 0) {
    yield {
      type: "chunk",
      content,
      content_type: "text",
      done: true,
    } as Chunk;
  }
  for (const toolCall of result.toolCalls ?? []) {
    yield toolCall;
  }
}

function parseCategory(raw: string, categories: string[]): string {
  if (categories.length === 0) return "Unknown";

  const parsed = extractJson(raw);
  const categoryValue = typeof parsed?.category === "string" ? parsed.category : "";
  for (const category of categories) {
    if (categoryValue.trim().toLowerCase() === category.trim().toLowerCase()) {
      return category;
    }
  }

  const lowered = raw.toLowerCase();
  for (const category of categories) {
    if (category.toLowerCase() && lowered.includes(category.toLowerCase())) {
      return category;
    }
  }

  for (const fallback of ["other", "unknown"]) {
    for (const category of categories) {
      if (category.trim().toLowerCase() === fallback) return category;
    }
  }

  return categories[0];
}

function messageContentText(content: Message["content"] | unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return asText(content);
  return content
    .map((part) => {
      if (!part || typeof part !== "object") return asText(part);
      if ((part as { type?: string }).type === "text") {
        return String((part as { text?: unknown }).text ?? "");
      }
      return "";
    })
    .join("")
    .trim();
}

function normalizeRole(role: unknown): Message["role"] | null {
  if (role === "system" || role === "user" || role === "assistant" || role === "tool") {
    return role;
  }
  return null;
}

function normalizeBinaryRef(
  value: unknown
): { uri?: string; data?: Uint8Array | string; mimeType?: string } | null {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  const out: { uri?: string; data?: Uint8Array | string; mimeType?: string } = {};
  if (typeof record.uri === "string" && record.uri) out.uri = record.uri;
  if (record.data instanceof Uint8Array || typeof record.data === "string") out.data = record.data;
  if (typeof record.mimeType === "string" && record.mimeType) out.mimeType = record.mimeType;
  if (typeof record.mime_type === "string" && record.mime_type) out.mimeType = record.mime_type;
  return out.uri || out.data ? out : null;
}

function normalizeMessageContent(value: unknown): Message["content"] {
  if (value == null || typeof value === "string") return value ?? null;
  if (!Array.isArray(value)) return asText(value);
  const parts: MessageContent[] = [];
  for (const part of value) {
    if (!part || typeof part !== "object") {
      const text = asText(part);
      if (text) parts.push({ type: "text", text });
      continue;
    }
    const record = part as Record<string, unknown>;
    const kind = typeof record.type === "string" ? record.type : "";
    if (kind === "text") {
      parts.push({ type: "text", text: asText(record.text ?? "") });
      continue;
    }
    if (kind === "image" || kind === "image_url") {
      const image = normalizeBinaryRef(record.image ?? record.image_url ?? record.imageUrl);
      if (image) parts.push({ type: "image", image } satisfies MessageImageContent);
      continue;
    }
    if (kind === "audio") {
      const audio = normalizeBinaryRef(record.audio);
      if (audio) parts.push({ type: "audio", audio } satisfies MessageAudioContent);
      continue;
    }
    const text = asText(part);
    if (text) parts.push({ type: "text", text });
  }
  return parts;
}

function normalizeToolCalls(value: unknown): ToolCall[] | null {
  if (!Array.isArray(value)) return null;
  const toolCalls = value
    .filter((item): item is Record<string, unknown> => !!item && typeof item === "object")
    .map((item, index) => ({
      id: typeof item.id === "string" && item.id ? item.id : `tool_${index + 1}`,
      name: typeof item.name === "string" ? item.name : "",
      args: item.args && typeof item.args === "object" ? (item.args as Record<string, unknown>) : {},
    }))
    .filter((item) => item.name.length > 0);
  return toolCalls.length > 0 ? toolCalls : null;
}

function normalizeMessage(value: unknown): Message | null {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  const role = normalizeRole(record.role);
  if (!role) return null;
  return {
    role,
    content: normalizeMessageContent(record.content),
    toolCalls: normalizeToolCalls(record.toolCalls ?? record.tool_calls),
    toolCallId:
      typeof record.toolCallId === "string"
        ? record.toolCallId
        : typeof record.tool_call_id === "string"
          ? record.tool_call_id
          : null,
    threadId:
      typeof record.threadId === "string"
        ? record.threadId
        : typeof record.thread_id === "string"
          ? record.thread_id
          : null,
  };
}

function threadMessages(threadId: string): Message[] {
  const thread = THREAD_STORE.get(threadId);
  if (!thread) return [];
  return thread.messages.map((message) => ({ ...message }));
}

function logThreadWarning(message: string, error: unknown, details: Record<string, unknown>): void {
  if (process.env["NODE_ENV"] === "test") return;
  console.warn(`[AgentNode] ${message}`, {
    ...details,
    error: String(error),
  });
}

function buildUserMessage(prompt: string, image: unknown, audio: unknown): Message {
  const content: MessageContent[] = [{ type: "text", text: prompt }];
  const imageRef = normalizeBinaryRef(image);
  if (imageRef) {
    content.push({ type: "image", image: imageRef });
  }
  const audioRef = normalizeBinaryRef(audio);
  if (audioRef) {
    content.push({ type: "audio", audio: audioRef });
  }
  return { role: "user", content };
}

async function loadThreadMessages(
  context: ProcessingContext | undefined,
  threadId: string
): Promise<Message[]> {
  if (!threadId) return [];
  const threadedContext = context as
    | (ProcessingContext & {
        get_messages?: (
          threadId: string,
          limit?: number,
          startKey?: string | null,
          reverse?: boolean
        ) => Promise<{ messages: Array<Record<string, unknown>> }>;
        getThreadMessages?: (
          threadId: string,
          limit?: number,
          startKey?: string | null,
          reverse?: boolean
        ) => Promise<{ messages: Array<Record<string, unknown>> }>;
      })
    | undefined;
  const getMessages =
    threadedContext?.get_messages?.bind(threadedContext) ??
    threadedContext?.getThreadMessages?.bind(threadedContext);
  if (getMessages) {
    try {
      const result = await getMessages(threadId, 1000, null, false);
      const messages = (result.messages ?? [])
        .map((item: Record<string, unknown>) => normalizeMessage(item))
        .filter((message: Message | null): message is Message => message !== null && message.role !== "system");
      log.info("Agent thread history loaded from context", {
        threadId,
        messageCount: messages.length,
      });
      return messages;
    } catch (error) {
      logThreadWarning("Failed to load thread messages", error, { threadId });
    }
  }
  const fallbackMessages = threadMessages(threadId).filter((message) => message.role !== "system");
  log.info("Agent thread history loaded from fallback store", {
    threadId,
    messageCount: fallbackMessages.length,
  });
  return fallbackMessages;
}

async function saveThreadMessage(
  context: ProcessingContext | undefined,
  threadId: string,
  message: Message
): Promise<void> {
  if (!threadId) return;
  const threadedContext = context as
    | (ProcessingContext & {
        create_message?: (req: Record<string, unknown>) => Promise<unknown>;
        createMessage?: (req: Record<string, unknown>) => Promise<unknown>;
      })
    | undefined;
  const createMessage =
    threadedContext?.create_message?.bind(threadedContext) ??
    threadedContext?.createMessage?.bind(threadedContext);
  if (createMessage) {
    try {
      await createMessage({
        thread_id: threadId,
        role: message.role,
        content: message.content ?? null,
        tool_calls: message.toolCalls ?? null,
        tool_call_id: message.toolCallId ?? null,
      });
      log.info("Agent thread message saved via context", {
        threadId,
        role: message.role,
        hasToolCalls: (message.toolCalls?.length ?? 0) > 0,
        textLength: messageContentText(message.content).length,
      });
      return;
    } catch (error) {
      logThreadWarning("Failed to save thread message", error, {
        threadId,
        role: message.role,
      });
    }
  }

  const thread = THREAD_STORE.get(threadId) ?? {
    id: threadId,
    title: "Agent Conversation",
    messages: [],
  };
  thread.messages.push({
    ...message,
    threadId,
  });
  THREAD_STORE.set(threadId, thread);
  log.info("Agent thread message saved via fallback store", {
    threadId,
    role: message.role,
    threadSize: thread.messages.length,
    hasToolCalls: (message.toolCalls?.length ?? 0) > 0,
    textLength: messageContentText(message.content).length,
  });
}

function isChunkItem(item: ProviderStreamItem): item is Chunk {
  return !!item && typeof item === "object" && "type" in item && (item as Chunk).type === "chunk";
}

function isToolCallItem(item: ProviderStreamItem): item is ToolCall {
  return !!item && typeof item === "object" && "id" in item && "name" in item && !("type" in item);
}

function normalizeTools(value: unknown): ToolLike[] {
  if (!Array.isArray(value)) return [];
  return value.filter(
    (tool): tool is ToolLike =>
      !!tool && typeof tool === "object" && typeof (tool as { name?: unknown }).name === "string"
  );
}

function toProviderTools(tools: ToolLike[]): Array<{ name: string; description?: string; inputSchema?: Record<string, unknown> }> {
  return tools.map((tool) =>
    typeof tool.toProviderTool === "function"
      ? tool.toProviderTool()
      : {
          name: tool.name,
          description: tool.description,
          inputSchema: tool.inputSchema,
        }
  );
}

function serializeToolResult(value: unknown): unknown {
  if (value == null) return value;
  if (Array.isArray(value)) return value.map(serializeToolResult);
  if (typeof value !== "object") return value;
  if (value instanceof Uint8Array) {
    return Buffer.from(value).toString("base64");
  }
  const record = value as Record<string, unknown>;
  return Object.fromEntries(Object.entries(record).map(([key, item]) => [key, serializeToolResult(item)]));
}

function getStructuredOutputSchema(node: BaseNode): Record<string, unknown> | null {
  const outputs = (node as { _dynamic_outputs?: unknown })._dynamic_outputs;
  if (!outputs || typeof outputs !== "object" || Array.isArray(outputs)) return null;
  const properties: Record<string, unknown> = {};
  const required: string[] = [];
  for (const [name, spec] of Object.entries(outputs as Record<string, unknown>)) {
    required.push(name);
    const value = spec && typeof spec === "object" ? (spec as Record<string, unknown>) : {};
    const declared = typeof value.type === "string" ? value.type.toLowerCase() : "str";
    let type = "string";
    if (["int", "integer"].includes(declared)) type = "integer";
    else if (["float", "number"].includes(declared)) type = "number";
    else if (["bool", "boolean"].includes(declared)) type = "boolean";
    else if (["list", "array"].includes(declared)) type = "array";
    else if (["dict", "object"].includes(declared)) type = "object";
    properties[name] = { type };
  }
  return {
    type: "object",
    additionalProperties: false,
    required,
    properties,
  };
}

function hasContentType(message: Message | undefined, type: MessageContent["type"]): boolean {
  return Array.isArray(message?.content)
    ? message!.content.some((part: MessageContent) => part.type === type)
    : false;
}

function inferControlValue(value: unknown): unknown {
  if (!value || typeof value !== "object" || Array.isArray(value)) return value;
  const record = value as Record<string, unknown>;
  if ("value" in record && record.value !== undefined) return record.value;
  if ("default" in record && record.default !== undefined) return record.default;
  return value;
}

function inferControlParams(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  const record = value as Record<string, unknown>;
  const properties =
    "properties" in record && record.properties && typeof record.properties === "object"
      ? (record.properties as Record<string, unknown>)
      : record;

  const entries = Object.entries(properties).map(([key, propValue]) => [
    key,
    inferControlValue(propValue),
  ]);
  return Object.fromEntries(entries);
}

function parseControlOutput(raw: string): Record<string, unknown> {
  const parsed = extractJson(raw);
  if (!parsed) return {};

  if (Object.keys(parsed).length === 1) {
    const [key] = Object.keys(parsed);
    const value = parsed[key];
    if (
      ["result", "output", "json", "data", "properties", "response"].includes(
        key.toLowerCase()
      ) &&
      value &&
      typeof value === "object" &&
      !Array.isArray(value)
    ) {
      return value as Record<string, unknown>;
    }
  }

  if (
    "properties" in parsed &&
    parsed.properties &&
    typeof parsed.properties === "object" &&
    !Array.isArray(parsed.properties)
  ) {
    return parsed.properties as Record<string, unknown>;
  }

  return parsed;
}

function parseResearchOutput(raw: string, query: string): {
  text: string;
  output: string;
  findings: Array<{ title: string; summary: string; source?: string }>;
} {
  const parsed = extractJson(raw);
  if (parsed) {
    const summary =
      typeof parsed.summary === "string"
        ? parsed.summary
        : typeof parsed.output === "string"
          ? parsed.output
          : "";
    const findings = Array.isArray(parsed.findings)
      ? parsed.findings
          .filter((item): item is Record<string, unknown> => !!item && typeof item === "object")
          .map((item) => ({
            title:
              typeof item.title === "string" && item.title.trim().length > 0
                ? item.title
                : query,
            summary: typeof item.summary === "string" ? item.summary : asText(item),
            source: typeof item.source === "string" ? item.source : undefined,
          }))
      : [];

    const text = summary || raw;
    return { text, output: text, findings };
  }

  return {
    text: raw,
    output: raw,
    findings: query ? [{ title: query, summary: raw }] : [],
  };
}

export class SummarizerNode extends BaseNode {
  static readonly nodeType = "nodetool.agents.Summarizer";
  static readonly title = "Summarizer";
  static readonly description = "Create a concise summary from text.";

  defaults() {
    return { text: "", max_sentences: 3 };
  }

  async process(
    inputs: Record<string, unknown>,
    context?: ProcessingContext
  ): Promise<Record<string, unknown>> {
    const text = asText(inputs.text ?? this._props.text ?? "");
    const maxSentences = Number(inputs.max_sentences ?? this._props.max_sentences ?? 3);
    const { providerId, modelId } = getModelConfig(inputs, this._props);
    if (hasProviderSupport(context, providerId, modelId)) {
      const provider = await context.getProvider(providerId);
      const result = await generateProviderMessage(provider, {
        model: modelId,
        maxTokens: Number.isFinite(maxSentences) ? Math.max(64, maxSentences * 128) : 384,
        messages: [
          { role: "system", content: SUMMARIZER_SYSTEM_PROMPT },
          {
            role: "user",
            content: `Summarize the following text in about ${Math.max(1, maxSentences)} sentence(s):\n\n${text}`,
          },
        ],
      });
      return { text: result, output: result };
    }
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

  async process(
    inputs: Record<string, unknown>,
    context?: ProcessingContext
  ): Promise<Record<string, unknown>> {
    const text = asText(inputs.text ?? this._props.text ?? "");
    const { providerId, modelId } = getModelConfig(inputs, this._props);
    if (hasProviderSupport(context, providerId, modelId)) {
      const provider = await context.getProvider(providerId);
      const raw = await generateProviderMessage(provider, {
        model: modelId,
        maxTokens: Number(inputs.max_tokens ?? this._props.max_tokens ?? 1024),
        responseFormat: { type: "json_object" },
        messages: [
          { role: "system", content: EXTRACTOR_SYSTEM_PROMPT },
          { role: "user", content: text },
        ],
      });
      const parsed = extractJson(raw);
      if (parsed) return parsed;
    }
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

  async process(
    inputs: Record<string, unknown>,
    context?: ProcessingContext
  ): Promise<Record<string, unknown>> {
    const text = asText(inputs.text ?? this._props.text ?? "");
    const categories = getCategories(inputs.categories ?? this._props.categories);
    if (categories.length < 2) {
      throw new Error("At least 2 categories are required");
    }

    const { providerId, modelId } = getModelConfig(inputs, this._props);
    if (hasProviderSupport(context, providerId, modelId)) {
      const provider = await context.getProvider(providerId);
      const raw = await generateProviderMessage(provider, {
        model: modelId,
        maxTokens: Number(inputs.max_tokens ?? this._props.max_tokens ?? 256),
        responseFormat: {
          type: "json_schema",
          json_schema: {
            name: "classification_result",
            schema: {
              type: "object",
              additionalProperties: false,
              required: ["category"],
              properties: {
                category: {
                  type: "string",
                  enum: categories,
                },
              },
            },
          },
        },
        messages: [
          { role: "system", content: CLASSIFIER_SYSTEM_PROMPT },
          {
            role: "user",
            content: `Allowed categories: ${categories.join(", ")}\n\nText: ${text}`,
          },
        ],
      });
      const category = parseCategory(raw, categories);
      return { output: category, category };
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
  static readonly description = "Generate natural language responses using LLM providers and stream output.";
  static readonly isStreamingOutput = true;

  defaults() {
    return {
      model: {},
      system: DEFAULT_SYSTEM_PROMPT,
      prompt: "",
      tools: [],
      image: {},
      audio: {},
      history: [],
      thread_id: "",
      max_tokens: 8192,
    };
  }

  async *genProcess(
    inputs: Record<string, unknown>,
    context?: ProcessingContext
  ): AsyncGenerator<Record<string, unknown>> {
    const { providerId, modelId } = getModelConfig(inputs, this._props);
    log.info("AgentNode starting", {
      nodeId: this._props.__node_id ?? null,
      providerId,
      modelId,
      hasContext: Boolean(context),
      hasGetProvider: Boolean(context && typeof context.getProvider === "function"),
      propKeys: Object.keys(this._props),
      inputKeys: Object.keys(inputs),
    });
    if (!providerId || !modelId) {
      log.error("AgentNode missing model selection", {
        nodeId: this._props.__node_id ?? null,
        providerId,
        modelId,
        modelInput: inputs.model ?? null,
        modelProp: this._props.model ?? null,
      });
      throw new Error("Select a model");
    }
    if (!context || typeof context.getProvider !== "function") {
      log.error("AgentNode missing processing context or provider access", {
        nodeId: this._props.__node_id ?? null,
        providerId,
        modelId,
      });
      throw new Error("Processing context is required");
    }

    const prompt = asText(inputs.prompt ?? this._props.prompt ?? "");
    const system = asText(inputs.system ?? this._props.system ?? DEFAULT_SYSTEM_PROMPT);
    const image = inputs.image ?? this._props.image;
    const audio = inputs.audio ?? this._props.audio;
    const historyInput = inputs.history ?? this._props.history;
    const history = Array.isArray(historyInput)
      ? historyInput.map((item) => normalizeMessage(item)).filter((item): item is Message => item !== null)
      : [];
    const threadId = String(inputs.thread_id ?? this._props.thread_id ?? "").trim();
    const maxTokens = Number(inputs.max_tokens ?? this._props.max_tokens ?? 8192);
    const tools = normalizeTools(inputs.tools ?? this._props.tools);
    const structuredSchema = getStructuredOutputSchema(this);
    const responseFormat = structuredSchema
      ? {
          type: "json_schema",
          json_schema: {
            name: "agent_structured_output",
            schema: structuredSchema,
            strict: true,
          },
        }
      : undefined;

    const messages: Message[] = [
      { role: "system", content: system },
      ...(await loadThreadMessages(context, threadId)),
      ...history,
      buildUserMessage(prompt, image, audio),
    ];
    log.info("AgentNode prepared messages", {
      nodeId: this._props.__node_id ?? null,
      providerId,
      modelId,
      threadId: threadId || null,
      promptLength: prompt.length,
      historyCount: history.length,
      toolCount: tools.length,
      messageCount: messages.length,
      hasImage: hasContentType(messages[messages.length - 1], "image"),
      hasAudio: hasContentType(messages[messages.length - 1], "audio"),
      responseFormat: responseFormat?.type ?? null,
    });

    if (threadId) {
      await saveThreadMessage(context, threadId, messages[messages.length - 1]);
    }

    let shouldContinue = false;
    let firstIteration = true;
    let lastTextOutput: string | null = null;
    const providerTools = tools.length > 0 ? toProviderTools(tools) : undefined;

    while (firstIteration || shouldContinue) {
      firstIteration = false;
      shouldContinue = false;
      log.info("AgentNode provider iteration starting", {
        nodeId: this._props.__node_id ?? null,
        providerId,
        modelId,
        threadId: threadId || null,
        messageCount: messages.length,
      });
      const provider = await context.getProvider(providerId);
      const assistantToolCalls: ToolCall[] = [];
      let assistantText = "";
      let chunkCount = 0;
      let thinkingCount = 0;
      let audioChunkCount = 0;

      for await (const item of streamProviderMessages(provider, {
        messages,
        model: modelId,
        tools: providerTools,
        maxTokens,
        responseFormat,
      })) {
        if (isChunkItem(item)) {
          chunkCount += 1;
          if (item.thinking) {
            thinkingCount += 1;
            log.debug("AgentNode received thinking chunk", {
              nodeId: this._props.__node_id ?? null,
              providerId,
              modelId,
              contentLength: (item.content ?? "").length,
              done: Boolean(item.done),
            });
            yield { chunk: null, thinking: item, text: null, audio: null };
            continue;
          }
          if (item.content_type === "audio") {
            audioChunkCount += 1;
            log.debug("AgentNode received audio chunk", {
              nodeId: this._props.__node_id ?? null,
              providerId,
              modelId,
              contentLength: (item.content ?? "").length,
              done: Boolean(item.done),
            });
            yield { chunk: item, thinking: null, text: null, audio: null };
            const audioBytes = item.content ? Buffer.from(item.content, "base64") : Buffer.alloc(0);
            yield {
              chunk: null,
              thinking: null,
              text: null,
              audio: { data: new Uint8Array(audioBytes) },
            };
          } else {
            assistantText += item.content ?? "";
            log.debug("AgentNode received text chunk", {
              nodeId: this._props.__node_id ?? null,
              providerId,
              modelId,
              chunkLength: (item.content ?? "").length,
              accumulatedLength: assistantText.length,
              done: Boolean(item.done),
            });
            yield { chunk: item, thinking: null, text: null, audio: null };
          }
          continue;
        }
        if (isToolCallItem(item)) {
          assistantToolCalls.push(item);
          log.info("AgentNode received tool call", {
            nodeId: this._props.__node_id ?? null,
            providerId,
            modelId,
            toolCallId: item.id,
            toolName: item.name,
            argKeys: Object.keys(item.args ?? {}),
          });
        }
      }

      log.info("AgentNode provider iteration completed", {
        nodeId: this._props.__node_id ?? null,
        providerId,
        modelId,
        chunkCount,
        thinkingCount,
        audioChunkCount,
        toolCallCount: assistantToolCalls.length,
        assistantTextLength: assistantText.length,
      });

      if (assistantText) {
        lastTextOutput = assistantText;
        log.info("AgentNode yielding final text", {
          nodeId: this._props.__node_id ?? null,
          providerId,
          modelId,
          textLength: assistantText.length,
        });
        yield { chunk: null, thinking: null, text: assistantText, audio: null };
      }

      if (assistantText || assistantToolCalls.length > 0) {
        const assistantMessage: Message = {
          role: "assistant",
          content: [{ type: "text", text: assistantText }],
          toolCalls: assistantToolCalls.length > 0 ? assistantToolCalls : null,
        };
        messages.push(assistantMessage);
        await saveThreadMessage(context, threadId, assistantMessage);
      }

      for (const toolCall of assistantToolCalls) {
        const tool = tools.find((candidate) => candidate.name === toolCall.name);
        if (!tool || typeof tool.process !== "function") {
          log.warn("AgentNode tool call had no matching executable tool", {
            nodeId: this._props.__node_id ?? null,
            toolCallId: toolCall.id,
            toolName: toolCall.name,
            availableTools: tools.map((candidate) => candidate.name),
          });
          continue;
        }
        log.info("AgentNode executing tool", {
          nodeId: this._props.__node_id ?? null,
          toolCallId: toolCall.id,
          toolName: toolCall.name,
        });
        const result = await tool.process(context, toolCall.args);
        const toolMessage: Message = {
          role: "tool",
          toolCallId: toolCall.id,
          content: JSON.stringify(serializeToolResult(result)),
        };
        messages.push(toolMessage);
        await saveThreadMessage(context, threadId, toolMessage);
        shouldContinue = true;
        log.info("AgentNode tool execution completed", {
          nodeId: this._props.__node_id ?? null,
          toolCallId: toolCall.id,
          toolName: toolCall.name,
          resultLength: String(toolMessage.content ?? "").length,
        });
      }
    }

    if (structuredSchema) {
      if (!lastTextOutput) {
        log.error("AgentNode structured output missing text payload", {
          nodeId: this._props.__node_id ?? null,
          providerId,
          modelId,
        });
        throw new Error("Agent did not return structured output text");
      }
      const parsed = extractJson(lastTextOutput);
      if (!parsed) {
        log.error("AgentNode structured output was not valid JSON", {
          nodeId: this._props.__node_id ?? null,
          providerId,
          modelId,
          textPreview: lastTextOutput.slice(0, 200),
        });
        throw new Error("Agent returned invalid structured output");
      }
      const required = Array.isArray((structuredSchema as { required?: unknown }).required)
        ? ((structuredSchema as { required: string[] }).required ?? [])
        : [];
      for (const name of required) {
        if (!(name in parsed)) {
          log.error("AgentNode structured output missing required field", {
            nodeId: this._props.__node_id ?? null,
            missingField: name,
            parsedKeys: Object.keys(parsed),
          });
          throw new Error(`Agent structured output is missing required field '${name}'`);
        }
      }
      log.info("AgentNode yielding structured output", {
        nodeId: this._props.__node_id ?? null,
        keys: Object.keys(parsed),
      });
      yield parsed;
    }

    log.info("AgentNode completed", {
      nodeId: this._props.__node_id ?? null,
      providerId,
      modelId,
      finalTextLength: lastTextOutput?.length ?? 0,
      returnedStructured: Boolean(structuredSchema),
    });
  }

  async process(
    inputs: Record<string, unknown>,
    context?: ProcessingContext
  ): Promise<Record<string, unknown>> {
    let lastText = "";
    let lastAudio: Record<string, unknown> | null = null;
    let structuredResult: Record<string, unknown> | null = null;

    for await (const item of this.genProcess(inputs, context)) {
      if (
        "chunk" in item ||
        "thinking" in item ||
        "text" in item ||
        "audio" in item
      ) {
        if (typeof item.text === "string") {
          lastText = item.text;
        }
        if (item.audio && typeof item.audio === "object") {
          lastAudio = item.audio as Record<string, unknown>;
        }
      } else {
        structuredResult = item;
      }
    }

    if (structuredResult) {
      log.info("AgentNode process() returning structured result", {
        nodeId: this._props.__node_id ?? null,
        keys: Object.keys(structuredResult),
      });
      return structuredResult;
    }

    log.info("AgentNode process() returning aggregate result", {
      nodeId: this._props.__node_id ?? null,
      textLength: lastText.length,
      hasAudio: Boolean(lastAudio),
    });
    return {
      text: lastText,
      output: lastText,
      chunk: null,
      thinking: null,
      audio: lastAudio,
    };
  }
}

export class ControlAgentNode extends BaseNode {
  static readonly nodeType = "nodetool.agents.ControlAgent";
  static readonly title = "Control Agent";
  static readonly description = "Generate control parameters from context.";

  defaults() {
    return { _control_context: {} };
  }

  async process(
    inputs: Record<string, unknown>,
    context?: ProcessingContext
  ): Promise<Record<string, unknown>> {
    const value = inputs._control_context ?? this._props._control_context ?? {};
    const legacyContext = asText(inputs.context ?? this._props.context ?? "");
    const schemaDescription = asText(
      inputs.schema_description ?? this._props.schema_description ?? ""
    );
    const { providerId, modelId } = getModelConfig(inputs, this._props);
    if (hasProviderSupport(context, providerId, modelId)) {
      const provider = await context.getProvider(providerId);
      const userPrompt =
        value && typeof value === "object"
          ? `_control_context:\n${JSON.stringify(value, null, 2)}`
          : `Context:\n${legacyContext}\n\nSchema description:\n${schemaDescription}`;
      const raw = await generateProviderMessage(provider, {
        model: modelId,
        maxTokens: Number(inputs.max_tokens ?? this._props.max_tokens ?? 2048),
        responseFormat: { type: "json_object" },
        messages: [
          { role: "system", content: CONTROL_AGENT_SYSTEM_PROMPT },
          { role: "user", content: userPrompt },
        ],
      });
      return { __control_output__: parseControlOutput(raw) };
    }
    return { __control_output__: inferControlParams(value) };
  }
}

export class ResearchAgentNode extends BaseNode {
  static readonly nodeType = "nodetool.agents.ResearchAgent";
  static readonly title = "Research Agent";
  static readonly description = "Produce lightweight research notes for a query.";

  defaults() {
    return { query: "", prompt: "" };
  }

  async process(
    inputs: Record<string, unknown>,
    context?: ProcessingContext
  ): Promise<Record<string, unknown>> {
    const query = asText(inputs.query ?? this._props.query ?? this._props.prompt ?? inputs.prompt ?? "");
    const { providerId, modelId } = getModelConfig(inputs, this._props);
    if (hasProviderSupport(context, providerId, modelId)) {
      const provider = await context.getProvider(providerId);
      const raw = await generateProviderMessage(provider, {
        model: modelId,
        maxTokens: Number(inputs.max_tokens ?? this._props.max_tokens ?? 2048),
        responseFormat: {
          type: "json_schema",
          json_schema: {
            name: "research_result",
            schema: {
              type: "object",
              additionalProperties: false,
              required: ["summary", "findings"],
              properties: {
                summary: { type: "string" },
                findings: {
                  type: "array",
                  items: {
                    type: "object",
                    additionalProperties: false,
                    required: ["title", "summary"],
                    properties: {
                      title: { type: "string" },
                      summary: { type: "string" },
                      source: { type: "string" },
                    },
                  },
                },
              },
            },
          },
        },
        messages: [
          { role: "system", content: RESEARCH_AGENT_SYSTEM_PROMPT },
          { role: "user", content: `Research objective: ${query}` },
        ],
      });
      return parseResearchOutput(raw, query);
    }
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
