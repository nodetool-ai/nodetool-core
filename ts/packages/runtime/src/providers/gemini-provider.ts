import type { Chunk } from "@nodetool/protocol";
import { createLogger } from "@nodetool/config";
import { BaseProvider } from "./base-provider.js";

const log = createLogger("nodetool.runtime.providers.gemini");
import type {
  LanguageModel,
  Message,
  MessageContent,
  MessageAudioContent,
  MessageImageContent,
  MessageTextContent,
  ProviderStreamItem,
  ProviderTool,
  ToolCall,
} from "./types.js";

const GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta";

interface GeminiProviderOptions {
  fetchFn?: typeof fetch;
}

/** A Gemini content part. */
interface GeminiPart {
  text?: string;
  inlineData?: { mimeType: string; data: string };
  functionCall?: { name: string; args?: Record<string, unknown> };
  functionResponse?: { name: string; response: unknown };
}

/** A Gemini content entry (role + parts). */
interface GeminiContent {
  role: "user" | "model";
  parts: GeminiPart[];
}

/** Shape of Gemini generateContent / streamGenerateContent request body. */
interface GeminiRequest {
  contents: GeminiContent[];
  systemInstruction?: { parts: Array<{ text: string }> };
  tools?: Array<{ functionDeclarations: Array<Record<string, unknown>> }>;
  generationConfig?: Record<string, unknown>;
}

/** A single candidate in a Gemini response. */
interface GeminiCandidate {
  content?: { parts?: GeminiPart[] };
  finishReason?: string;
}

/** Top-level Gemini response shape. */
interface GeminiResponse {
  candidates?: GeminiCandidate[];
  error?: { message?: string };
}

/** Shape of a model entry from the Gemini models list API. */
interface GeminiModelEntry {
  name?: string;
  displayName?: string;
  supportedGenerationMethods?: string[];
}

function sanitizeToolName(name: string): string {
  let sanitized = (name ?? "").trim();
  sanitized = sanitized.replace(/[^a-zA-Z0-9_.:-]/g, "_");
  sanitized = sanitized.replace(/_+/g, "_");
  if (!sanitized) sanitized = "_tool";
  if (!/^[a-zA-Z_]/.test(sanitized)) sanitized = `_${sanitized}`;
  if (sanitized.length > 64) sanitized = sanitized.slice(0, 64);
  if (!sanitized) sanitized = "_tool";
  return sanitized;
}

export class GeminiProvider extends BaseProvider {
  static requiredSecrets(): string[] {
    return ["GEMINI_API_KEY"];
  }

  readonly apiKey: string;
  private _fetch: typeof fetch;

  constructor(secrets: { GEMINI_API_KEY?: string }, options: GeminiProviderOptions = {}) {
    super("gemini");

    const apiKey = secrets.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY is required");
    }

    this.apiKey = apiKey;
    this._fetch = options.fetchFn ?? globalThis.fetch.bind(globalThis);
  }

  getContainerEnv(): Record<string, string> {
    return { GEMINI_API_KEY: this.apiKey };
  }

  hasToolSupport(_model: string): boolean {
    return true;
  }

  // ---------------------------------------------------------------------------
  // Model listing
  // ---------------------------------------------------------------------------

  async getAvailableLanguageModels(): Promise<LanguageModel[]> {
    const url = `${GEMINI_API_BASE}/models?key=${this.apiKey}`;

    let response: Response;
    try {
      response = await this._fetch(url);
    } catch {
      return [];
    }

    if (!response.ok) return [];

    const payload = (await response.json()) as { models?: GeminiModelEntry[] };
    const items = payload.models ?? [];

    return items
      .filter((m) => (m.supportedGenerationMethods ?? []).includes("generateContent"))
      .filter((m) => !!m.name)
      .map((m) => {
        const id = (m.name as string).split("/").pop() as string;
        return {
          id,
          name: m.displayName ?? id,
          provider: "gemini",
        };
      });
  }

  // ---------------------------------------------------------------------------
  // Message conversion helpers
  // ---------------------------------------------------------------------------

  private async messageContentToGeminiPart(content: MessageContent): Promise<GeminiPart> {
    if (content.type === "text") {
      return { text: (content as MessageTextContent).text };
    }

    if (content.type === "image") {
      const img = (content as MessageImageContent).image;
      let base64Data: string;
      let mimeType = img.mimeType ?? "image/jpeg";

      if (img.data) {
        if (typeof img.data === "string") {
          base64Data = img.data;
        } else {
          base64Data = Buffer.from(img.data).toString("base64");
        }
      } else if (img.uri) {
        if (img.uri.startsWith("data:")) {
          const idx = img.uri.indexOf(",");
          const header = img.uri.slice(5, idx);
          mimeType = header.split(";")[0] || mimeType;
          base64Data = img.uri.slice(idx + 1);
        } else {
          const resp = await this._fetch(img.uri);
          if (!resp.ok) throw new Error(`Failed to fetch image: ${resp.status}`);
          mimeType = resp.headers.get("content-type") ?? mimeType;
          base64Data = Buffer.from(await resp.arrayBuffer()).toString("base64");
        }
      } else {
        base64Data = "";
      }

      return { inlineData: { mimeType, data: base64Data } };
    }

    if (content.type === "audio") {
      const aud = (content as MessageAudioContent).audio;
      let base64Data: string;
      let mimeType = aud.mimeType ?? "audio/mp3";

      if (aud.data) {
        if (typeof aud.data === "string") {
          base64Data = aud.data;
        } else {
          base64Data = Buffer.from(aud.data).toString("base64");
        }
      } else if (aud.uri) {
        if (aud.uri.startsWith("data:")) {
          const idx = aud.uri.indexOf(",");
          const header = aud.uri.slice(5, idx);
          mimeType = header.split(";")[0] || mimeType;
          base64Data = aud.uri.slice(idx + 1);
        } else {
          const resp = await this._fetch(aud.uri);
          if (!resp.ok) throw new Error(`Failed to fetch audio: ${resp.status}`);
          mimeType = resp.headers.get("content-type") ?? mimeType;
          base64Data = Buffer.from(await resp.arrayBuffer()).toString("base64");
        }
      } else {
        base64Data = "";
      }

      return { inlineData: { mimeType, data: base64Data } };
    }

    return { text: "[unsupported content type]" };
  }

  /**
   * Convert our Message array into Gemini contents + optional system instruction.
   */
  async convertMessages(
    messages: Message[]
  ): Promise<{ contents: GeminiContent[]; systemInstruction?: string }> {
    let systemInstruction: string | undefined;
    const contents: GeminiContent[] = [];

    for (const msg of messages) {
      if (msg.role === "system") {
        systemInstruction = typeof msg.content === "string"
          ? msg.content
          : (msg.content ?? [])
              .filter((c): c is MessageTextContent => c.type === "text")
              .map((c) => c.text)
              .join(" ");
        continue;
      }

      if (msg.role === "tool") {
        // Tool result → model role with functionResponse part
        const responseText =
          typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content);

        contents.push({
          role: "user",
          parts: [
            {
              functionResponse: {
                name: msg.toolCallId ?? "unknown",
                response: { result: responseText },
              },
            },
          ],
        });
        continue;
      }

      if (msg.role === "assistant") {
        const parts: GeminiPart[] = [];

        // Tool calls
        if (msg.toolCalls && msg.toolCalls.length > 0) {
          for (const tc of msg.toolCalls) {
            parts.push({
              functionCall: { name: tc.name, args: tc.args },
            });
          }
        }

        // Text / content
        if (typeof msg.content === "string" && msg.content) {
          parts.push({ text: msg.content });
        } else if (Array.isArray(msg.content)) {
          for (const c of msg.content) {
            parts.push(await this.messageContentToGeminiPart(c));
          }
        }

        if (parts.length > 0) {
          contents.push({ role: "model", parts });
        }
        continue;
      }

      // user
      const parts: GeminiPart[] = [];
      if (typeof msg.content === "string") {
        parts.push({ text: msg.content });
      } else if (Array.isArray(msg.content)) {
        for (const c of msg.content) {
          parts.push(await this.messageContentToGeminiPart(c));
        }
      }
      if (parts.length > 0) {
        contents.push({ role: "user", parts });
      }
    }

    return { contents, systemInstruction };
  }

  formatTools(
    tools: ProviderTool[]
  ): {
    geminiTools: Array<{ functionDeclarations: Array<Record<string, unknown>> }>;
    nameMap: Map<string, string>;
    reverseMap: Map<string, string>;
  } {
    const nameMap = new Map<string, string>();
    const reverseMap = new Map<string, string>();
    const usedNames = new Set<string>();
    const declarations: Array<Record<string, unknown>> = [];

    for (const tool of tools) {
      const original = tool.name;
      let unique = sanitizeToolName(original);

      let suffix = 2;
      while (usedNames.has(unique)) {
        const sfx = `_${suffix}`;
        unique = `${sanitizeToolName(original).slice(0, 64 - sfx.length)}${sfx}`;
        suffix++;
      }

      usedNames.add(unique);
      nameMap.set(original, unique);
      reverseMap.set(unique, original);

      declarations.push({
        name: unique,
        description: tool.description ?? "",
        parameters: tool.inputSchema ?? { type: "object", properties: {} },
      });
    }

    return {
      geminiTools: declarations.length > 0 ? [{ functionDeclarations: declarations }] : [],
      nameMap,
      reverseMap,
    };
  }

  // ---------------------------------------------------------------------------
  // Non-streaming generation
  // ---------------------------------------------------------------------------

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
    const {
      model,
      tools = [],
      maxTokens = 16384,
      responseFormat,
      jsonSchema,
      temperature,
      topP,
    } = args;

    const { contents, systemInstruction } = await this.convertMessages(args.messages);
    const { geminiTools, reverseMap } = this.formatTools(tools);

    const body: GeminiRequest = { contents };

    if (systemInstruction) {
      body.systemInstruction = { parts: [{ text: systemInstruction }] };
    }

    if (geminiTools.length > 0) {
      body.tools = geminiTools;
    }

    const generationConfig: Record<string, unknown> = {
      maxOutputTokens: maxTokens,
    };
    if (temperature != null) generationConfig.temperature = temperature;
    if (topP != null) generationConfig.topP = topP;
    if (responseFormat || jsonSchema) {
      generationConfig.responseMimeType = "application/json";
      if (jsonSchema) generationConfig.responseSchema = jsonSchema;
    }
    body.generationConfig = generationConfig;

    log.debug("Gemini request", { model });

    const url = `${GEMINI_API_BASE}/models/${model}:generateContent?key=${this.apiKey}`;
    const response = await this._fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const text = await response.text();
      log.error("Gemini request failed", { model, error: `${response.status}: ${text.slice(0, 200)}` });
      throw new Error(`Gemini API error ${response.status}: ${text}`);
    }

    const data = (await response.json()) as GeminiResponse;

    if (data.error) {
      throw new Error(`Gemini API error: ${data.error.message}`);
    }

    const candidate = data.candidates?.[0];
    if (!candidate?.content?.parts) {
      throw new Error("Gemini returned no candidates");
    }

    return this.extractMessage(candidate.content.parts, reverseMap);
  }

  // ---------------------------------------------------------------------------
  // Streaming generation
  // ---------------------------------------------------------------------------

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
    const {
      model,
      tools = [],
      maxTokens = 16384,
      temperature,
      topP,
    } = args;

    const { contents, systemInstruction } = await this.convertMessages(args.messages);
    const { geminiTools, reverseMap } = this.formatTools(tools);

    const body: GeminiRequest = { contents };

    if (systemInstruction) {
      body.systemInstruction = { parts: [{ text: systemInstruction }] };
    }

    if (geminiTools.length > 0) {
      body.tools = geminiTools;
    }

    const generationConfig: Record<string, unknown> = {
      maxOutputTokens: maxTokens,
    };
    if (temperature != null) generationConfig.temperature = temperature;
    if (topP != null) generationConfig.topP = topP;
    body.generationConfig = generationConfig;

    log.debug("Gemini request", { model });

    const url = `${GEMINI_API_BASE}/models/${model}:streamGenerateContent?alt=sse&key=${this.apiKey}`;
    const response = await this._fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const text = await response.text();
      log.error("Gemini request failed", { model, error: `${response.status}: ${text.slice(0, 200)}` });
      throw new Error(`Gemini API error ${response.status}: ${text}`);
    }

    if (!response.body) {
      throw new Error("Gemini streaming response has no body");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const jsonStr = line.slice(6).trim();
          if (!jsonStr || jsonStr === "[DONE]") continue;

          let event: GeminiResponse;
          try {
            event = JSON.parse(jsonStr) as GeminiResponse;
          } catch {
            continue;
          }

          const parts = event.candidates?.[0]?.content?.parts;
          if (!parts) continue;

          for (const part of parts) {
            if (part.text !== undefined) {
              const chunk: Chunk = {
                type: "chunk",
                content: part.text,
                done: false,
              };
              yield chunk;
            } else if (part.functionCall) {
              const originalName = reverseMap.get(part.functionCall.name) ?? part.functionCall.name;
              const toolCall: ToolCall = {
                id: `call_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
                name: originalName,
                args: part.functionCall.args ?? {},
              };
              yield toolCall;
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }

    // Emit synthetic done chunk
    const doneChunk: Chunk = {
      type: "chunk",
      content: "",
      done: true,
    };
    yield doneChunk;
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  private extractMessage(
    parts: GeminiPart[],
    reverseMap: Map<string, string>
  ): Message {
    const textParts: string[] = [];
    const toolCalls: ToolCall[] = [];

    for (const part of parts) {
      if (part.text !== undefined) {
        textParts.push(part.text);
      } else if (part.functionCall) {
        const originalName = reverseMap.get(part.functionCall.name) ?? part.functionCall.name;
        toolCalls.push({
          id: `call_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
          name: originalName,
          args: part.functionCall.args ?? {},
        });
      }
    }

    return {
      role: "assistant",
      content: textParts.join("") || null,
      toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
    };
  }

  isContextLengthError(error: unknown): boolean {
    const msg = String(error).toLowerCase();
    return (
      msg.includes("context length") ||
      msg.includes("maximum context") ||
      msg.includes("too long") ||
      msg.includes("token limit")
    );
  }
}
