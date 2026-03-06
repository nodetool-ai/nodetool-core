/**
 * Provider factory for the chat CLI.
 * Creates the right BaseProvider instance based on name + available API keys.
 */

import type { BaseProvider } from "@nodetool/runtime";
import {
  AnthropicProvider,
  OpenAIProvider,
  OllamaProvider,
  GeminiProvider,
  MistralProvider,
  GroqProvider,
  OllamaProvider as LlamaProvider,
} from "@nodetool/runtime";

export const KNOWN_PROVIDERS = ["anthropic", "openai", "ollama", "gemini", "mistral", "groq"] as const;
export type KnownProvider = (typeof KNOWN_PROVIDERS)[number];

/** Default models for each provider. */
export const DEFAULT_MODELS: Record<string, string> = {
  anthropic: "claude-sonnet-4-6",
  openai: "gpt-4o",
  ollama: "llama3.2",
  gemini: "gemini-2.0-flash",
  mistral: "mistral-large-latest",
  groq: "llama-3.3-70b-versatile",
};

export function createProvider(providerId: string): BaseProvider {
  switch (providerId.toLowerCase()) {
    case "anthropic":
      return new AnthropicProvider({ ANTHROPIC_API_KEY: process.env["ANTHROPIC_API_KEY"] });
    case "openai":
      return new OpenAIProvider({ OPENAI_API_KEY: process.env["OPENAI_API_KEY"] });
    case "ollama":
      return new OllamaProvider({ OLLAMA_API_URL: process.env["OLLAMA_API_URL"] });
    case "gemini":
      return new GeminiProvider({ GEMINI_API_KEY: process.env["GEMINI_API_KEY"] });
    case "mistral":
      return new MistralProvider({ MISTRAL_API_KEY: process.env["MISTRAL_API_KEY"] });
    case "groq":
      return new GroqProvider({ GROQ_API_KEY: process.env["GROQ_API_KEY"] });
    default:
      // Attempt to use Ollama as a catch-all local provider
      return new OllamaProvider({ OLLAMA_API_URL: process.env["OLLAMA_API_URL"] });
  }
}

/** Check which providers have API keys configured. */
export function availableProviders(): string[] {
  const available: string[] = [];
  if (process.env["ANTHROPIC_API_KEY"]) available.push("anthropic");
  if (process.env["OPENAI_API_KEY"]) available.push("openai");
  if (process.env["GEMINI_API_KEY"]) available.push("gemini");
  if (process.env["MISTRAL_API_KEY"]) available.push("mistral");
  if (process.env["GROQ_API_KEY"]) available.push("groq");
  available.push("ollama"); // always available (local)
  return available;
}
