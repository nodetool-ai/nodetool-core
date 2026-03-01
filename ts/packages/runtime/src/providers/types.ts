import type { Chunk } from "@nodetool/protocol";

export type ProviderId = "openai" | "anthropic" | "ollama" | "llama_cpp" | string;

export interface LanguageModel {
  id: string;
  name: string;
  provider: ProviderId;
}

export interface ImageModel {
  id: string;
  name: string;
  provider: ProviderId;
  supportedTasks?: string[];
}

export interface VideoModel {
  id: string;
  name: string;
  provider: ProviderId;
  supportedTasks?: string[];
}

export interface TTSModel {
  id: string;
  name: string;
  provider: ProviderId;
  voices?: string[];
}

export interface ASRModel {
  id: string;
  name: string;
  provider: ProviderId;
}

export interface EmbeddingModel {
  id: string;
  name: string;
  provider: ProviderId;
  dimensions?: number;
}

export interface ToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
}

export interface ProviderTool {
  name: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
  type?: "function" | "code_interpreter";
}

export interface MessageTextContent {
  type: "text";
  text: string;
}

export interface MessageImageContent {
  type: "image";
  image: {
    uri?: string;
    data?: Uint8Array | string;
    mimeType?: string;
  };
}

export interface MessageAudioContent {
  type: "audio";
  audio: {
    uri?: string;
    data?: Uint8Array | string;
    mimeType?: string;
  };
}

export type MessageContent =
  | MessageTextContent
  | MessageImageContent
  | MessageAudioContent;

export interface Message {
  role: "system" | "user" | "assistant" | "tool";
  content?: string | MessageContent[] | null;
  toolCalls?: ToolCall[] | null;
  toolCallId?: string | null;
  threadId?: string | null;
}

export interface TextToImageParams {
  model: ImageModel;
  prompt: string;
  negativePrompt?: string | null;
  width?: number;
  height?: number;
  quality?: string | null;
}

export interface ImageToImageParams {
  model: ImageModel;
  prompt: string;
  negativePrompt?: string | null;
  targetWidth?: number | null;
  targetHeight?: number | null;
  quality?: string | null;
}

export interface TextToVideoParams {
  model: VideoModel;
  prompt: string;
  negativePrompt?: string | null;
  numFrames?: number | null;
  aspectRatio?: string | null;
  resolution?: string | null;
}

export interface ImageToVideoParams {
  model: VideoModel;
  prompt?: string | null;
  negativePrompt?: string | null;
  numFrames?: number | null;
  aspectRatio?: string | null;
  resolution?: string | null;
}

export type ProviderStreamItem = Chunk | ToolCall;

export interface StreamingAudioChunk {
  samples: Int16Array;
}
