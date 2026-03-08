import { BaseNode } from "@nodetool/node-sdk";
import type { NodeClass } from "@nodetool/node-sdk";

const OPENAI_API_BASE = "https://api.openai.com/v1";

function getApiKey(inputs: Record<string, unknown>): string {
  const key =
    (inputs._secrets as Record<string, string>)?.OPENAI_API_KEY ||
    process.env.OPENAI_API_KEY ||
    "";
  if (!key) throw new Error("OPENAI_API_KEY is not configured");
  return key;
}

function authHeaders(apiKey: string): Record<string, string> {
  return {
    Authorization: `Bearer ${apiKey}`,
    "Content-Type": "application/json",
  };
}

// ---------------------------------------------------------------------------
// 1. Embedding
// ---------------------------------------------------------------------------
export class EmbeddingNode extends BaseNode {
  static readonly nodeType = "openai.text.Embedding";
  static readonly title = "Embedding";
  static readonly description =
    "Generate vector representations of text for semantic analysis using OpenAI embedding models.";

  defaults() {
    return {
      input: "",
      model: "text-embedding-3-small",
      chunk_size: 4096,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const text = String(inputs.input ?? "");
    const model = String(inputs.model ?? "text-embedding-3-small");
    const chunkSize = Number(inputs.chunk_size ?? 4096);

    // Chunk input text
    const chunks: string[] = [];
    for (let i = 0; i < text.length; i += chunkSize) {
      chunks.push(text.slice(i, i + chunkSize));
    }
    if (chunks.length === 0) chunks.push("");

    const res = await fetch(`${OPENAI_API_BASE}/embeddings`, {
      method: "POST",
      headers: authHeaders(apiKey),
      body: JSON.stringify({ input: chunks, model }),
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`OpenAI Embedding API error ${res.status}: ${err}`);
    }
    const data = (await res.json()) as {
      data: Array<{ embedding: number[] }>;
    };

    // Average embeddings across chunks
    const allEmbeddings = data.data.map((d) => d.embedding);
    const dim = allEmbeddings[0]?.length ?? 0;
    const avg = new Array(dim).fill(0);
    for (const emb of allEmbeddings) {
      for (let i = 0; i < dim; i++) {
        avg[i] += emb[i] / allEmbeddings.length;
      }
    }

    return { output: avg };
  }
}

// ---------------------------------------------------------------------------
// 2. WebSearch
// ---------------------------------------------------------------------------
export class WebSearchNode extends BaseNode {
  static readonly nodeType = "openai.text.WebSearch";
  static readonly title = "Web Search";
  static readonly description =
    "Search the web using OpenAI's web search capabilities with gpt-4o-search-preview.";

  defaults() {
    return { query: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const query = String(inputs.query ?? "");
    if (!query) throw new Error("Search query cannot be empty");

    const res = await fetch(`${OPENAI_API_BASE}/chat/completions`, {
      method: "POST",
      headers: authHeaders(apiKey),
      body: JSON.stringify({
        model: "gpt-4o-search-preview",
        web_search_options: {},
        messages: [{ role: "user", content: query }],
      }),
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`OpenAI WebSearch API error ${res.status}: ${err}`);
    }
    const data = (await res.json()) as Record<string, unknown>;

    let content: string;
    if (data.choices && Array.isArray(data.choices)) {
      const first = data.choices[0] as Record<string, unknown> | undefined;
      const msg = first?.message as Record<string, unknown> | undefined;
      content = String(msg?.content ?? JSON.stringify(data));
    } else {
      content = JSON.stringify(data);
    }

    return { output: content };
  }
}

// ---------------------------------------------------------------------------
// 3. Moderation
// ---------------------------------------------------------------------------
export class ModerationNode extends BaseNode {
  static readonly nodeType = "openai.text.Moderation";
  static readonly title = "Moderation";
  static readonly description =
    "Check text content for potential policy violations using OpenAI's moderation API.";

  defaults() {
    return {
      input: "",
      model: "omni-moderation-latest",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const text = String(inputs.input ?? "");
    const model = String(inputs.model ?? "omni-moderation-latest");
    if (!text) throw new Error("Input text cannot be empty");

    const res = await fetch(`${OPENAI_API_BASE}/moderations`, {
      method: "POST",
      headers: authHeaders(apiKey),
      body: JSON.stringify({ input: text, model }),
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`OpenAI Moderation API error ${res.status}: ${err}`);
    }
    const data = (await res.json()) as Record<string, unknown>;

    const results = data.results as Array<Record<string, unknown>> | undefined;
    if (results && results.length > 0) {
      const result = results[0];
      return {
        flagged: result.flagged ?? false,
        categories: result.categories ?? {},
        category_scores: result.category_scores ?? {},
      };
    }

    return { flagged: false, categories: {}, category_scores: {} };
  }
}

// ---------------------------------------------------------------------------
// 4. CreateImage
// ---------------------------------------------------------------------------
export class CreateImageNode extends BaseNode {
  static readonly nodeType = "openai.image.CreateImage";
  static readonly title = "Create Image";
  static readonly description =
    "Generate images from textual descriptions using OpenAI's image generation models.";

  defaults() {
    return {
      prompt: "",
      model: "gpt-image-1",
      size: "1024x1024",
      background: "auto",
      quality: "high",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");

    const model = String(inputs.model ?? "gpt-image-1");
    const size = String(inputs.size ?? "1024x1024");
    const quality = String(inputs.quality ?? "high");
    const background = String(inputs.background ?? "auto");

    const res = await fetch(`${OPENAI_API_BASE}/images/generations`, {
      method: "POST",
      headers: authHeaders(apiKey),
      body: JSON.stringify({
        prompt,
        model,
        n: 1,
        size,
        quality,
        background,
        response_format: "b64_json",
      }),
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`OpenAI CreateImage API error ${res.status}: ${err}`);
    }
    const data = (await res.json()) as {
      data: Array<{ b64_json?: string; url?: string }>;
    };

    const item = data.data[0];
    if (item.b64_json) {
      return { output: { data: `data:image/png;base64,${item.b64_json}` } };
    } else if (item.url) {
      return { output: { uri: item.url } };
    }
    throw new Error("No image data in response");
  }
}

// ---------------------------------------------------------------------------
// 5. EditImage
// ---------------------------------------------------------------------------
export class EditImageNode extends BaseNode {
  static readonly nodeType = "openai.image.EditImage";
  static readonly title = "Edit Image";
  static readonly description =
    "Edit images using OpenAI's gpt-image-1 model with text prompts and optional masks.";

  defaults() {
    return {
      image: {},
      mask: {},
      prompt: "",
      model: "gpt-image-1",
      size: "1024x1024",
      quality: "high",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Edit prompt cannot be empty");

    const image = inputs.image as Record<string, unknown> | undefined;
    if (!image || (!image.data && !image.uri)) {
      throw new Error("Input image is required");
    }

    const model = String(inputs.model ?? "gpt-image-1");
    const size = String(inputs.size ?? "1024x1024");
    const quality = String(inputs.quality ?? "high");

    // Build multipart form data
    const formData = new FormData();
    formData.append("prompt", prompt);
    formData.append("model", model);
    formData.append("size", size);
    formData.append("quality", quality);
    formData.append("response_format", "b64_json");

    // Convert image ref to blob
    const imageBlob = await refToBlob(image);
    formData.append("image", imageBlob, "image.png");

    // Optional mask
    const mask = inputs.mask as Record<string, unknown> | undefined;
    if (mask && (mask.data || mask.uri)) {
      const maskBlob = await refToBlob(mask);
      formData.append("mask", maskBlob, "mask.png");
    }

    const res = await fetch(`${OPENAI_API_BASE}/images/edits`, {
      method: "POST",
      headers: { Authorization: `Bearer ${apiKey}` },
      body: formData,
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`OpenAI EditImage API error ${res.status}: ${err}`);
    }
    const data = (await res.json()) as {
      data: Array<{ b64_json?: string; url?: string }>;
    };

    const item = data.data[0];
    if (item.b64_json) {
      return { output: { data: `data:image/png;base64,${item.b64_json}` } };
    } else if (item.url) {
      return { output: { uri: item.url } };
    }
    throw new Error("No image data in response");
  }
}

/** Convert an image/audio ref object to a Blob for multipart upload. */
async function refToBlob(ref: Record<string, unknown>): Promise<Blob> {
  if (ref.data && typeof ref.data === "string") {
    const dataStr = ref.data as string;
    // Handle data: URI
    if (dataStr.startsWith("data:")) {
      const commaIdx = dataStr.indexOf(",");
      const b64 = dataStr.slice(commaIdx + 1);
      const buf = Buffer.from(b64, "base64");
      return new Blob([buf]);
    }
    // Assume raw base64
    const buf = Buffer.from(dataStr, "base64");
    return new Blob([buf]);
  }
  if (ref.uri && typeof ref.uri === "string") {
    const r = await fetch(ref.uri as string);
    return await r.blob();
  }
  throw new Error("Cannot convert ref to blob: no data or uri");
}

// ---------------------------------------------------------------------------
// 6. TextToSpeech
// ---------------------------------------------------------------------------
export class TextToSpeechNode extends BaseNode {
  static readonly nodeType = "openai.audio.TextToSpeech";
  static readonly title = "Text to Speech";
  static readonly description =
    "Convert text to speech using OpenAI TTS models.";

  defaults() {
    return {
      model: "tts-1",
      voice: "alloy",
      input: "",
      speed: 1.0,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const text = String(inputs.input ?? "");
    const model = String(inputs.model ?? "tts-1");
    const voice = String(inputs.voice ?? "alloy");
    const speed = Number(inputs.speed ?? 1.0);

    const res = await fetch(`${OPENAI_API_BASE}/audio/speech`, {
      method: "POST",
      headers: authHeaders(apiKey),
      body: JSON.stringify({ model, input: text, voice, speed, response_format: "mp3" }),
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`OpenAI TTS API error ${res.status}: ${err}`);
    }

    const arrayBuf = await res.arrayBuffer();
    const b64 = Buffer.from(arrayBuf).toString("base64");
    return { output: { data: `data:audio/mp3;base64,${b64}` } };
  }
}

// ---------------------------------------------------------------------------
// 7. Translate
// ---------------------------------------------------------------------------
export class TranslateNode extends BaseNode {
  static readonly nodeType = "openai.audio.Translate";
  static readonly title = "Translate";
  static readonly description =
    "Translate speech in audio to English text using OpenAI Whisper.";

  defaults() {
    return {
      audio: {},
      temperature: 0.0,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const audio = inputs.audio as Record<string, unknown> | undefined;
    if (!audio || (!audio.data && !audio.uri)) {
      throw new Error("Audio input is required");
    }
    const temperature = Number(inputs.temperature ?? 0.0);

    const audioBlob = await refToBlob(audio);
    const formData = new FormData();
    formData.append("file", audioBlob, "audio.mp3");
    formData.append("model", "whisper-1");
    formData.append("temperature", String(temperature));

    const res = await fetch(`${OPENAI_API_BASE}/audio/translations`, {
      method: "POST",
      headers: { Authorization: `Bearer ${apiKey}` },
      body: formData,
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`OpenAI Translate API error ${res.status}: ${err}`);
    }
    const data = (await res.json()) as { text: string };
    return { output: data.text };
  }
}

// ---------------------------------------------------------------------------
// 8. Transcribe
// ---------------------------------------------------------------------------
export class TranscribeNode extends BaseNode {
  static readonly nodeType = "openai.audio.Transcribe";
  static readonly title = "Transcribe";
  static readonly description =
    "Convert speech to text using OpenAI's speech-to-text API (Whisper / GPT-4o transcribe).";

  defaults() {
    return {
      model: "whisper-1",
      audio: {},
      language: "auto_detect",
      timestamps: false,
      prompt: "",
      temperature: 0,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const audio = inputs.audio as Record<string, unknown> | undefined;
    if (!audio || (!audio.data && !audio.uri)) {
      throw new Error("Audio input is required");
    }

    const model = String(inputs.model ?? "whisper-1");
    const language = String(inputs.language ?? "auto_detect");
    const timestamps = Boolean(inputs.timestamps ?? false);
    const promptText = String(inputs.prompt ?? "");
    const temperature = Number(inputs.temperature ?? 0);

    const isNewModel = model === "gpt-4o-transcribe" || model === "gpt-4o-mini-transcribe";

    const audioBlob = await refToBlob(audio);
    const formData = new FormData();
    formData.append("file", audioBlob, "audio.mp3");
    formData.append("model", model);
    formData.append("temperature", String(temperature));

    if (timestamps) {
      if (isNewModel) {
        throw new Error("New transcription models do not support verbose_json timestamps");
      }
      formData.append("response_format", "verbose_json");
      formData.append("timestamp_granularities[]", "segment");
      formData.append("timestamp_granularities[]", "word");
    } else {
      formData.append("response_format", "json");
    }

    if (language !== "auto_detect") {
      formData.append("language", language);
    }
    if (promptText) {
      formData.append("prompt", promptText);
    }

    const res = await fetch(`${OPENAI_API_BASE}/audio/transcriptions`, {
      method: "POST",
      headers: { Authorization: `Bearer ${apiKey}` },
      body: formData,
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`OpenAI Transcribe API error ${res.status}: ${err}`);
    }
    const data = (await res.json()) as Record<string, unknown>;

    const text = String(data.text ?? "");
    const words: Array<{ timestamp: [number, number]; text: string }> = [];
    const segments: Array<{ timestamp: [number, number]; text: string }> = [];

    if (timestamps && !isNewModel) {
      const rawSegments = data.segments as Array<Record<string, unknown>> | undefined;
      if (rawSegments) {
        for (const seg of rawSegments) {
          segments.push({
            timestamp: [Number(seg.start), Number(seg.end)],
            text: String(seg.text),
          });
        }
      }
      const rawWords = data.words as Array<Record<string, unknown>> | undefined;
      if (rawWords) {
        for (const w of rawWords) {
          words.push({
            timestamp: [Number(w.start), Number(w.end)],
            text: String(w.word),
          });
        }
      }
    }

    return { text, words, segments };
  }
}

// ---------------------------------------------------------------------------
// 9. RealtimeAgent (stub — WebSocket not yet implemented)
// ---------------------------------------------------------------------------
export class RealtimeAgentNode extends BaseNode {
  static readonly nodeType = "openai.agents.RealtimeAgent";
  static readonly title = "Realtime Agent";
  static readonly description =
    "Stream responses using OpenAI Realtime API with optional audio input and text output. (WebSocket-based — not yet implemented in TS runtime.)";

  defaults() {
    return {
      model: "gpt-4o-mini-realtime-preview",
      system: "",
      chunk: {},
      voice: "alloy",
      speed: 1.0,
      temperature: 0.8,
    };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    throw new Error(
      "RealtimeAgent is not yet implemented in the TS runtime. " +
        "It requires WebSocket-based streaming which is only available in the Python runtime."
    );
  }
}

// ---------------------------------------------------------------------------
// 10. RealtimeTranscription (stub — WebSocket not yet implemented)
// ---------------------------------------------------------------------------
export class RealtimeTranscriptionNode extends BaseNode {
  static readonly nodeType = "openai.agents.RealtimeTranscription";
  static readonly title = "Realtime Transcription";
  static readonly description =
    "Stream microphone or audio input to OpenAI Realtime and emit transcription. (WebSocket-based — not yet implemented in TS runtime.)";

  defaults() {
    return {
      model: {},
      system: "",
      temperature: 0.8,
    };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    throw new Error(
      "RealtimeTranscription is not yet implemented in the TS runtime. " +
        "It requires WebSocket-based streaming which is only available in the Python runtime."
    );
  }
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------
export const OPENAI_NODES: readonly NodeClass[] = [
  EmbeddingNode,
  WebSearchNode,
  ModerationNode,
  CreateImageNode,
  EditImageNode,
  TextToSpeechNode,
  TranslateNode,
  TranscribeNode,
  RealtimeAgentNode,
  RealtimeTranscriptionNode,
];
