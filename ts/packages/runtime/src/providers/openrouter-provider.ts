import OpenAI from "openai";
import { OpenAIProvider } from "./openai-provider.js";
import type { LanguageModel } from "./types.js";

interface OpenRouterProviderOptions {
  client?: OpenAI;
  clientFactory?: (apiKey: string) => OpenAI;
  fetchFn?: typeof fetch;
}

export class OpenRouterProvider extends OpenAIProvider {
  static override requiredSecrets(): string[] {
    return ["OPENROUTER_API_KEY"];
  }

  private _routerFetch: typeof fetch;

  constructor(
    secrets: { OPENROUTER_API_KEY?: string },
    options: OpenRouterProviderOptions = {}
  ) {
    const apiKey = secrets.OPENROUTER_API_KEY;
    if (!apiKey) {
      throw new Error("OPENROUTER_API_KEY is required");
    }

    const fetchFn = options.fetchFn ?? globalThis.fetch.bind(globalThis);

    super(
      { OPENAI_API_KEY: apiKey },
      {
        client: options.client,
        clientFactory:
          options.clientFactory ??
          ((key) =>
            new OpenAI({
              apiKey: key,
              baseURL: "https://openrouter.ai/api/v1",
              defaultHeaders: {
                "HTTP-Referer": "https://github.com/nodetool-ai/nodetool-core",
                "X-Title": "NodeTool",
              },
            })),
        fetchFn,
      }
    );

    (this as { provider: string }).provider = "openrouter";
    this._routerFetch = fetchFn;
  }

  override getContainerEnv(): Record<string, string> {
    return { OPENROUTER_API_KEY: this.apiKey };
  }

  override hasToolSupport(model: string): boolean {
    const lower = model.toLowerCase();
    if (lower.includes("o1") || lower.includes("o3")) {
      return false;
    }
    return true;
  }

  override async getAvailableLanguageModels(): Promise<LanguageModel[]> {
    const response = await this._routerFetch(
      "https://openrouter.ai/api/v1/models",
      {
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          "HTTP-Referer": "https://github.com/nodetool-ai/nodetool-core",
          "X-Title": "NodeTool",
        },
      }
    );

    if (!response.ok) {
      return [];
    }

    const payload = (await response.json()) as {
      data?: Array<{ id?: string; name?: string }>;
    };
    const rows = payload.data ?? [];
    return rows
      .filter(
        (row): row is { id: string; name?: string } =>
          typeof row.id === "string" && row.id.length > 0
      )
      .map((row) => ({
        id: row.id,
        name: row.name ?? row.id,
        provider: "openrouter",
      }));
  }
}
