import type {
  ASRModel,
  EmbeddingModel,
  ImageModel,
  ImageToImageParams,
  ImageToVideoParams,
  LanguageModel,
  Message,
  ProviderId,
  ProviderStreamItem,
  ProviderTool,
  StreamingAudioChunk,
  TextToImageParams,
  TextToVideoParams,
  ToolCall,
  TTSModel,
  VideoModel,
} from "./types.js";
import { CostCalculator } from "./cost-calculator.js";
import type { UsageInfo } from "./cost-calculator.js";

export abstract class BaseProvider {
  readonly provider: ProviderId;
  private _cost = 0;

  protected constructor(provider: ProviderId) {
    this.provider = provider;
  }

  static requiredSecrets(): string[] {
    return [];
  }

  getContainerEnv(): Record<string, string> {
    return {};
  }

  hasToolSupport(_model: string): boolean {
    return true;
  }

  trackUsage(model: string, usage: UsageInfo): number {
    const cost = CostCalculator.calculate(model, usage, this.provider);
    this._cost += cost;
    return cost;
  }

  getTotalCost(): number {
    return this._cost;
  }

  resetCost(): void {
    this._cost = 0;
  }

  async getAvailableLanguageModels(): Promise<LanguageModel[]> {
    return [];
  }

  async getAvailableImageModels(): Promise<ImageModel[]> {
    return [];
  }

  async getAvailableVideoModels(): Promise<VideoModel[]> {
    return [];
  }

  async getAvailableTTSModels(): Promise<TTSModel[]> {
    return [];
  }

  async getAvailableASRModels(): Promise<ASRModel[]> {
    return [];
  }

  async getAvailableEmbeddingModels(): Promise<EmbeddingModel[]> {
    return [];
  }

  abstract generateMessage(args: {
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
  }): Promise<Message>;

  abstract generateMessages(args: {
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
  }): AsyncGenerator<ProviderStreamItem>;

  async textToImage(_params: TextToImageParams): Promise<Uint8Array> {
    throw new Error(`${this.provider} does not support textToImage`);
  }

  async imageToImage(_image: Uint8Array, _params: ImageToImageParams): Promise<Uint8Array> {
    throw new Error(`${this.provider} does not support imageToImage`);
  }

  async *textToSpeech(_args: {
    text: string;
    model: string;
    voice?: string;
    speed?: number;
  }): AsyncGenerator<StreamingAudioChunk> {
    throw new Error(`${this.provider} does not support textToSpeech`);
  }

  async automaticSpeechRecognition(_args: {
    audio: Uint8Array;
    model: string;
    language?: string;
    prompt?: string;
    temperature?: number;
  }): Promise<string> {
    throw new Error(`${this.provider} does not support automaticSpeechRecognition`);
  }

  async textToVideo(_params: TextToVideoParams): Promise<Uint8Array> {
    throw new Error(`${this.provider} does not support textToVideo`);
  }

  async imageToVideo(_image: Uint8Array, _params: ImageToVideoParams): Promise<Uint8Array> {
    throw new Error(`${this.provider} does not support imageToVideo`);
  }

  async generateEmbedding(_args: {
    text: string | string[];
    model: string;
    dimensions?: number;
  }): Promise<number[][]> {
    throw new Error(`${this.provider} does not support generateEmbedding`);
  }

  isContextLengthError(error: unknown): boolean {
    const msg = String(error).toLowerCase();
    return msg.includes("context length") || msg.includes("maximum context");
  }

  protected parseToolCallArgs(raw: unknown): Record<string, unknown> {
    if (typeof raw !== "string") {
      return {};
    }
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

  protected buildToolCall(id: string, name: string, args: unknown): ToolCall {
    return {
      id,
      name,
      args: this.parseToolCallArgs(args),
    };
  }
}
