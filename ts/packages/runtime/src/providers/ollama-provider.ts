import type { Chunk } from "@nodetool/protocol";
import { createLogger } from "@nodetool/config";
import { BaseProvider } from "./base-provider.js";

const log = createLogger("nodetool.runtime.providers.ollama");
import type {
  EmbeddingModel,
  LanguageModel,
  Message,
  MessageContent,
  MessageImageContent,
  MessageTextContent,
  ProviderStreamItem,
  ProviderTool,
  ToolCall,
} from "./types.js";

interface OllamaProviderOptions {
  fetchFn?: typeof fetch;
}

type OllamaToolCall = {
  function?: {
    name?: string;
    arguments?: unknown;
  };
};

type OllamaChatMessage = {
  role?: string;
  content?: string;
  images?: string[];
  tool_calls?: OllamaToolCall[];
};

function parseDataUri(uri: string): { mime: string; base64: string } {
  const idx = uri.indexOf(",");
  if (idx < 0) {
    throw new Error("Invalid data URI");
  }
  const header = uri.slice(5, idx);
  const payload = uri.slice(idx + 1);
  const isBase64 = header.includes(";base64");
  const mime = header.split(";")[0] || "application/octet-stream";
  if (isBase64) {
    return { mime, base64: payload };
  }
  return {
    mime,
    base64: Buffer.from(decodeURIComponent(payload), "utf8").toString("base64"),
  };
}

function normalizeToolArgs(raw: unknown): Record<string, unknown> {
  if (!raw) return {};
  if (typeof raw === "string") {
    try {
      const parsed = JSON.parse(raw) as unknown;
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
      return {};
    } catch {
      return {};
    }
  }
  if (typeof raw === "object" && !Array.isArray(raw)) {
    return raw as Record<string, unknown>;
  }
  return {};
}

function asTextParts(content: MessageContent[]): string {
  return content
    .filter((part): part is MessageTextContent => part.type === "text")
    .map((part) => part.text)
    .join("\n");
}

export class OllamaProvider extends BaseProvider {
  static requiredSecrets(): string[] {
    return ["OLLAMA_API_URL"];
  }

  readonly apiUrl: string;
  private _fetch: typeof fetch;

  constructor(secrets: { OLLAMA_API_URL?: string }, options: OllamaProviderOptions = {}) {
    super("ollama");
    const apiUrl = secrets.OLLAMA_API_URL ?? process.env.OLLAMA_API_URL;
    if (!apiUrl || !apiUrl.trim()) {
      throw new Error("OLLAMA_API_URL is required");
    }
    this.apiUrl = apiUrl.replace(/\/+$/, "");
    this._fetch = options.fetchFn ?? globalThis.fetch.bind(globalThis);
  }

  getContainerEnv(): Record<string, string> {
    return { OLLAMA_API_URL: this.apiUrl };
  }

  hasToolSupport(_model: string): boolean {
    // Ollama exposes native tools for supported models; leave enabled by default.
    return true;
  }

  private async imageToBase64(image: MessageImageContent["image"]): Promise<string> {
    if (typeof image.data === "string" && image.data.startsWith("data:")) {
      return parseDataUri(image.data).base64;
    }
    if (typeof image.uri === "string" && image.uri.startsWith("data:")) {
      return parseDataUri(image.uri).base64;
    }
    if (typeof image.data === "string") {
      return image.data;
    }
    if (image.data instanceof Uint8Array) {
      return Buffer.from(image.data).toString("base64");
    }
    if (image.uri) {
      const response = await this._fetch(image.uri);
      if (!response.ok) {
        throw new Error(`Failed to fetch image URI: ${response.status}`);
      }
      const bytes = new Uint8Array(await response.arrayBuffer());
      return Buffer.from(bytes).toString("base64");
    }
    throw new Error("Invalid image payload: expected uri or data");
  }

  async convertMessage(message: Message): Promise<Record<string, unknown>> {
    if (message.role === "tool") {
      const content =
        typeof message.content === "string"
          ? message.content
          : JSON.stringify(message.content ?? null);
      return { role: "tool", content };
    }

    if (message.role === "assistant") {
      const out: Record<string, unknown> = {
        role: "assistant",
        content: typeof message.content === "string" ? message.content : "",
      };

      const toolCalls = message.toolCalls ?? [];
      if (toolCalls.length > 0) {
        out.tool_calls = toolCalls.map((tc) => ({
          function: {
            name: tc.name,
            arguments: tc.args,
          },
        }));
      }
      return out;
    }

    if (message.role === "system") {
      if (typeof message.content === "string") {
        return { role: "system", content: message.content };
      }
      if (Array.isArray(message.content)) {
        return { role: "system", content: asTextParts(message.content) };
      }
      return { role: "system", content: "" };
    }

    if (message.role !== "user") {
      throw new Error(`Unsupported message role: ${message.role}`);
    }

    if (typeof message.content === "string") {
      return { role: "user", content: message.content };
    }

    const parts = message.content ?? [];
    const text = asTextParts(parts);
    const images = await Promise.all(
      parts
        .filter((part): part is MessageImageContent => part.type === "image")
        .map((part) => this.imageToBase64(part.image))
    );

    const out: Record<string, unknown> = {
      role: "user",
      content: text,
    };
    if (images.length > 0) {
      out.images = images;
    }
    return out;
  }

  formatTools(tools: ProviderTool[]): Array<Record<string, unknown>> {
    return tools.map((tool) => ({
      type: "function",
      function: {
        name: tool.name,
        description: tool.description ?? "",
        parameters: tool.inputSchema ?? { type: "object", properties: {} },
      },
    }));
  }

  private async postJson<T>(path: string, body: Record<string, unknown>): Promise<T> {
    const response = await this._fetch(`${this.apiUrl}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      throw new Error(`Ollama API request failed (${response.status})`);
    }
    return (await response.json()) as T;
  }

  async getAvailableLanguageModels(): Promise<LanguageModel[]> {
    const response = await this._fetch(`${this.apiUrl}/api/tags`);
    if (!response.ok) return [];
    const payload = (await response.json()) as {
      models?: Array<{ name?: string; model?: string }>;
    };
    const rows = payload.models ?? [];
    return rows
      .map((m) => m.model ?? m.name)
      .filter((id): id is string => typeof id === "string" && id.length > 0)
      .map((id) => ({
        id,
        name: id,
        provider: "ollama",
      }));
  }

  async getAvailableEmbeddingModels(): Promise<EmbeddingModel[]> {
    const models = await this.getAvailableLanguageModels();
    return models.map((m) => ({
      id: m.id,
      name: m.name,
      provider: "ollama",
      dimensions: 0,
    }));
  }

  private toToolCalls(toolCalls: OllamaToolCall[] | undefined): ToolCall[] {
    if (!Array.isArray(toolCalls)) return [];
    return toolCalls
      .map((tc, idx) => {
        const fn = tc.function ?? {};
        const name = typeof fn.name === "string" ? fn.name : "";
        if (!name) return null;
        return {
          id: `tool_${idx + 1}`,
          name,
          args: normalizeToolArgs(fn.arguments),
        } satisfies ToolCall;
      })
      .filter((tc): tc is ToolCall => tc !== null);
  }

  private async buildChatRequest(args: {
    messages: Message[];
    model: string;
    tools?: ProviderTool[];
    maxTokens?: number;
    responseFormat?: Record<string, unknown>;
  }): Promise<Record<string, unknown>> {
    const request: Record<string, unknown> = {
      model: args.model,
      messages: await Promise.all(args.messages.map((m) => this.convertMessage(m))),
      options: {
        num_predict: args.maxTokens ?? 8192,
      },
    };

    if ((args.tools ?? []).length > 0 && this.hasToolSupport(args.model)) {
      request.tools = this.formatTools(args.tools ?? []);
    }

    if (args.responseFormat?.type === "json_schema") {
      const schema = (args.responseFormat.json_schema as { schema?: unknown } | undefined)?.schema;
      if (!schema) {
        throw new Error("schema is required in json_schema response format");
      }
      request.format = schema;
    }

    return request;
  }

  async generateMessage(args: {
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
    if (args.jsonSchema) {
      throw new Error("Ollama provider expects responseFormat; jsonSchema is not supported directly");
    }

    const request = await this.buildChatRequest({
      messages: args.messages,
      model: args.model,
      tools: args.tools,
      maxTokens: args.maxTokens,
      responseFormat: args.responseFormat,
    });
    request.stream = false;

    log.debug("Ollama request", { model: args.model });

    const response = await this.postJson<{ message?: OllamaChatMessage }>("/api/chat", request);
    const message = response.message ?? {};
    const content = typeof message.content === "string" ? message.content : "";
    const toolCalls = this.toToolCalls(message.tool_calls);

    return {
      role: "assistant",
      content,
      toolCalls,
    };
  }

  async *generateMessages(args: {
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
    if (args.jsonSchema) {
      throw new Error("Ollama provider expects responseFormat; jsonSchema is not supported directly");
    }

    const request = await this.buildChatRequest({
      messages: args.messages,
      model: args.model,
      tools: args.tools,
      maxTokens: args.maxTokens,
      responseFormat: args.responseFormat,
    });
    request.stream = true;

    log.debug("Ollama request", { model: args.model });

    const response = await this._fetch(`${this.apiUrl}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(request),
    });

    if (!response.ok || !response.body) {
      throw new Error(`Ollama API request failed (${response.status})`);
    }

    const decoder = new TextDecoder();
    let buffer = "";
    const reader = response.body.getReader();

    try {
      while (true) {
        const read = await reader.read();
        if (read.done) break;
        buffer += decoder.decode(read.value, { stream: true });

        while (true) {
          const idx = buffer.indexOf("\n");
          if (idx < 0) break;
          const line = buffer.slice(0, idx).trim();
          buffer = buffer.slice(idx + 1);
          if (!line) continue;

          const event = JSON.parse(line) as {
            message?: OllamaChatMessage;
            done?: boolean;
          };
          const message = event.message ?? {};

          for (const tc of this.toToolCalls(message.tool_calls)) {
            yield tc;
          }

          const content = typeof message.content === "string" ? message.content : "";
          if (content.length > 0 || event.done) {
            const chunk: Chunk = {
              type: "chunk",
              content,
              done: event.done ?? false,
            };
            yield chunk;
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  async generateEmbedding(args: {
    text: string | string[];
    model: string;
    dimensions?: number;
  }): Promise<number[][]> {
    const values = Array.isArray(args.text) ? args.text : [args.text];
    if (values.length === 0 || values.some((v) => typeof v !== "string" || v.length === 0)) {
      throw new Error("text must not be empty");
    }
    const response = await this.postJson<{ embeddings?: number[][] }>(
      "/api/embed",
      {
        model: args.model,
        input: values,
      }
    );
    return response.embeddings ?? [];
  }

  isContextLengthError(error: unknown): boolean {
    const msg = String(error).toLowerCase();
    return (
      msg.includes("context length") ||
      msg.includes("context window") ||
      msg.includes("token limit") ||
      msg.includes("request too large") ||
      msg.includes("413")
    );
  }
}

