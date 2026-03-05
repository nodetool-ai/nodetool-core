import { BaseNode } from "@nodetool/node-sdk";
import type { NodeClass } from "@nodetool/node-sdk";

const GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta";

function getGeminiApiKey(inputs: Record<string, unknown>): string {
  const key =
    (inputs._secrets as Record<string, string>)?.GEMINI_API_KEY ||
    process.env.GEMINI_API_KEY ||
    "";
  if (!key) throw new Error("GEMINI_API_KEY is not configured");
  return key;
}

function getAudioBytes(audio: Record<string, unknown>): Uint8Array {
  if (typeof audio.data === "string") {
    return Uint8Array.from(Buffer.from(audio.data, "base64"));
  }
  if (audio.data instanceof Uint8Array) {
    return audio.data;
  }
  throw new Error("Audio data is required");
}

function getImageBytes(image: Record<string, unknown>): Uint8Array | null {
  if (typeof image.data === "string" && image.data.length > 0) {
    return Uint8Array.from(Buffer.from(image.data, "base64"));
  }
  if (image.data instanceof Uint8Array) {
    return image.data;
  }
  return null;
}

function isRefSet(ref: unknown): boolean {
  if (!ref || typeof ref !== "object") return false;
  const r = ref as Record<string, unknown>;
  return Boolean(r.data || r.uri || r.asset_id);
}

// ── Text nodes ──────────────────────────────────────────────────────────────

export class GroundedSearchNode extends BaseNode {
  static readonly nodeType = "gemini.text.GroundedSearch";
  static readonly title = "Grounded Search";
  static readonly description =
    "Search the web using Google's Gemini API with grounding capabilities. " +
    "Returns structured results with source information.";

  defaults() {
    return {
      query: "",
      model: "gemini-2.0-flash",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getGeminiApiKey(inputs);
    const query = String(inputs.query ?? this._props.query ?? "");
    const model = String(inputs.model ?? this._props.model ?? "gemini-2.0-flash");

    if (!query) throw new Error("Search query is required");

    const url = `${GEMINI_API_BASE}/models/${model}:generateContent?key=${apiKey}`;
    const body = {
      contents: [{ parts: [{ text: query }] }],
      tools: [{ google_search: {} }],
      generationConfig: {
        responseMimeType: "text/plain",
      },
    };

    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Gemini API error ${res.status}: ${errText}`);
    }

    const data = (await res.json()) as Record<string, unknown>;

    const candidates = data.candidates as Array<Record<string, unknown>> | undefined;
    if (!candidates || candidates.length === 0) {
      throw new Error("No response received from Gemini API");
    }

    const candidate = candidates[0];
    const content = candidate.content as Record<string, unknown> | undefined;
    if (!content) throw new Error("Invalid response format from Gemini API");

    const parts = content.parts as Array<Record<string, unknown>> | undefined;
    const results: string[] = [];
    if (parts) {
      for (const part of parts) {
        if (typeof part.text === "string") {
          results.push(part.text);
        }
      }
    }

    const sources: Array<{ title: string; url: string }> = [];
    const groundingMetadata = candidate.groundingMetadata as Record<string, unknown> | undefined;
    if (groundingMetadata) {
      const chunks = groundingMetadata.groundingChunks as Array<Record<string, unknown>> | undefined;
      if (chunks) {
        for (const chunk of chunks) {
          const web = chunk.web as Record<string, unknown> | undefined;
          if (web) {
            const source = {
              title: String(web.title ?? ""),
              url: String(web.uri ?? ""),
            };
            if (source.url && !sources.some((s) => s.url === source.url)) {
              sources.push(source);
            }
          }
        }
      }
    }

    return { results, sources };
  }
}

export class EmbeddingNode extends BaseNode {
  static readonly nodeType = "gemini.text.Embedding";
  static readonly title = "Embedding";
  static readonly description =
    "Generate vector representations of text for semantic analysis using " +
    "Google's Gemini API embedding models.";

  defaults() {
    return {
      input: "",
      model: "text-embedding-004",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getGeminiApiKey(inputs);
    const input = String(inputs.input ?? this._props.input ?? "");
    const model = String(inputs.model ?? this._props.model ?? "text-embedding-004");

    if (!input) throw new Error("Input text is required for embedding generation");

    const url = `${GEMINI_API_BASE}/models/${model}:embedContent?key=${apiKey}`;
    const body = {
      content: { parts: [{ text: input }] },
    };

    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Gemini API error ${res.status}: ${errText}`);
    }

    const data = (await res.json()) as Record<string, unknown>;
    const embedding = data.embedding as Record<string, unknown> | undefined;
    if (!embedding || !embedding.values) {
      throw new Error("No embedding generated from the input text");
    }

    return { output: embedding.values };
  }
}

// ── Image nodes ─────────────────────────────────────────────────────────────

export class ImageGenerationNode extends BaseNode {
  static readonly nodeType = "gemini.image.ImageGeneration";
  static readonly title = "Image Generation";
  static readonly description =
    "Generate an image using Google's Imagen or Gemini image-capable models " +
    "via the Gemini API.";

  defaults() {
    return {
      prompt: "",
      model: "imagen-3.0-generate-002",
      image: { uri: "", data: null, asset_id: null },
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getGeminiApiKey(inputs);
    const prompt = String(inputs.prompt ?? this._props.prompt ?? "");
    const model = String(inputs.model ?? this._props.model ?? "imagen-3.0-generate-002");
    const image = (inputs.image ?? this._props.image ?? {}) as Record<string, unknown>;

    if (!prompt) throw new Error("The input prompt cannot be empty.");

    // Gemini image-capable models use generateContent with IMAGE+TEXT modalities
    if (model.startsWith("gemini-")) {
      const contentParts: Array<Record<string, unknown>> = [{ text: prompt }];

      // Add optional input image
      const imageBytes = getImageBytes(image);
      if (imageBytes) {
        contentParts.push({
          inline_data: {
            mime_type: "image/png",
            data: Buffer.from(imageBytes).toString("base64"),
          },
        });
      }

      const url = `${GEMINI_API_BASE}/models/${model}:generateContent?key=${apiKey}`;
      const body = {
        contents: [{ parts: contentParts }],
        generationConfig: {
          responseModalities: ["IMAGE", "TEXT"],
        },
      };

      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Gemini API error ${res.status}: ${errText}`);
      }

      const data = (await res.json()) as Record<string, unknown>;
      const candidates = data.candidates as Array<Record<string, unknown>> | undefined;
      if (!candidates || candidates.length === 0) {
        throw new Error("No response received from Gemini API");
      }

      const candidate = candidates[0];

      if (candidate.finishReason === "PROHIBITED_CONTENT") {
        throw new Error("Prohibited content in the input prompt");
      }

      const content = candidate.content as Record<string, unknown> | undefined;
      const parts = content?.parts as Array<Record<string, unknown>> | undefined;
      if (!parts) throw new Error("Invalid response format from Gemini API");

      for (const part of parts) {
        const inlineData = part.inlineData as Record<string, unknown> | undefined;
        // Also check snake_case variant from API
        const inlineData2 = part.inline_data as Record<string, unknown> | undefined;
        const d = inlineData ?? inlineData2;
        if (d && typeof d.data === "string") {
          return { output: { type: "image", data: d.data } };
        }
      }

      throw new Error("No image bytes returned in response");
    }

    // Imagen models use the generateImages endpoint
    const url = `${GEMINI_API_BASE}/models/${model}:predict?key=${apiKey}`;
    const body = {
      instances: [{ prompt }],
      parameters: { sampleCount: 1 },
    };

    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Gemini API error ${res.status}: ${errText}`);
    }

    const data = (await res.json()) as Record<string, unknown>;
    const predictions = data.predictions as Array<Record<string, unknown>> | undefined;
    if (!predictions || predictions.length === 0) {
      throw new Error("No images generated");
    }

    const bytesBase64 = predictions[0].bytesBase64Encoded as string | undefined;
    if (!bytesBase64) throw new Error("No image bytes in response");

    return { output: { type: "image", data: bytesBase64 } };
  }
}

// ── Video nodes ─────────────────────────────────────────────────────────────

export class TextToVideoGeminiNode extends BaseNode {
  static readonly nodeType = "gemini.video.TextToVideo";
  static readonly title = "Text To Video";
  static readonly description =
    "Generate videos from text prompts using Google's Veo models. " +
    "Supports 720p resolution at 24fps with 8-second duration and native audio generation.";

  defaults() {
    return {
      prompt: "",
      model: "veo-3.1-generate-preview",
      aspect_ratio: "16:9",
      negative_prompt: "",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getGeminiApiKey(inputs);
    const prompt = String(inputs.prompt ?? this._props.prompt ?? "");
    const model = String(inputs.model ?? this._props.model ?? "veo-3.1-generate-preview");
    const aspectRatio = String(inputs.aspect_ratio ?? this._props.aspect_ratio ?? "16:9");
    const negativePrompt = String(inputs.negative_prompt ?? this._props.negative_prompt ?? "");

    if (!prompt) throw new Error("Video generation prompt is required");

    // Start the long-running video generation operation
    const url = `${GEMINI_API_BASE}/models/${model}:generateVideos?key=${apiKey}`;
    const config: Record<string, unknown> = {};
    if (aspectRatio) config.aspectRatio = aspectRatio;
    if (negativePrompt) config.negativePrompt = negativePrompt;

    const body: Record<string, unknown> = {
      prompt,
    };
    if (Object.keys(config).length > 0) body.config = config;

    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Gemini API error ${res.status}: ${errText}`);
    }

    const operation = (await res.json()) as Record<string, unknown>;
    const videoData = await pollVideoOperation(apiKey, operation);
    return { output: { data: videoData } };
  }
}

export class ImageToVideoGeminiNode extends BaseNode {
  static readonly nodeType = "gemini.video.ImageToVideo";
  static readonly title = "Image To Video";
  static readonly description =
    "Generate videos from images using Google's Veo models. " +
    "Animate static images into dynamic videos.";

  defaults() {
    return {
      image: { uri: "", data: null, asset_id: null },
      prompt: "",
      model: "veo-3.1-generate-preview",
      aspect_ratio: "16:9",
      negative_prompt: "",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getGeminiApiKey(inputs);
    const image = (inputs.image ?? this._props.image ?? {}) as Record<string, unknown>;
    const prompt = String(inputs.prompt ?? this._props.prompt ?? "Animate this image");
    const model = String(inputs.model ?? this._props.model ?? "veo-3.1-generate-preview");
    const aspectRatio = String(inputs.aspect_ratio ?? this._props.aspect_ratio ?? "16:9");
    const negativePrompt = String(inputs.negative_prompt ?? this._props.negative_prompt ?? "");

    if (!isRefSet(image)) throw new Error("Input image is required");

    const imageBytes = getImageBytes(image);
    if (!imageBytes) throw new Error("Image data is required");

    const config: Record<string, unknown> = {};
    if (aspectRatio) config.aspectRatio = aspectRatio;
    if (negativePrompt) config.negativePrompt = negativePrompt;

    const url = `${GEMINI_API_BASE}/models/${model}:generateVideos?key=${apiKey}`;
    const body: Record<string, unknown> = {
      prompt: prompt || "Animate this image",
      image: {
        imageBytes: Buffer.from(imageBytes).toString("base64"),
        mimeType: "image/png",
      },
    };
    if (Object.keys(config).length > 0) body.config = config;

    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Gemini API error ${res.status}: ${errText}`);
    }

    const operation = (await res.json()) as Record<string, unknown>;
    const videoData = await pollVideoOperation(apiKey, operation);
    return { output: { data: videoData } };
  }
}

async function pollVideoOperation(
  apiKey: string,
  operation: Record<string, unknown>
): Promise<string> {
  // If the operation already has the result, return it
  if (operation.done) {
    return extractVideoFromResponse(operation);
  }

  // Otherwise poll the operation
  const opName = operation.name as string | undefined;
  if (!opName) throw new Error("No operation name in response");

  const maxAttempts = 120; // 10 minutes at 5s intervals
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise((resolve) => setTimeout(resolve, 5000));

    const pollUrl = `https://generativelanguage.googleapis.com/v1beta/${opName}?key=${apiKey}`;
    const pollRes = await fetch(pollUrl);
    if (!pollRes.ok) {
      const errText = await pollRes.text();
      throw new Error(`Gemini poll error ${pollRes.status}: ${errText}`);
    }

    const pollData = (await pollRes.json()) as Record<string, unknown>;
    if (pollData.done) {
      return extractVideoFromResponse(pollData);
    }
  }

  throw new Error("Video generation timed out");
}

function extractVideoFromResponse(data: Record<string, unknown>): string {
  const response = (data.response ?? data) as Record<string, unknown>;
  const generatedVideos = response.generatedVideos as Array<Record<string, unknown>> | undefined;
  if (!generatedVideos || generatedVideos.length === 0) {
    throw new Error("No video generated");
  }

  const video = generatedVideos[0].video as Record<string, unknown> | undefined;
  if (!video) throw new Error("No video in response");

  const videoBytes = video.videoBytes as string | undefined;
  if (!videoBytes) throw new Error("No video bytes in response");

  return videoBytes;
}

// ── Audio nodes ─────────────────────────────────────────────────────────────

export class TextToSpeechGeminiNode extends BaseNode {
  static readonly nodeType = "gemini.audio.TextToSpeech";
  static readonly title = "Text To Speech";
  static readonly description =
    "Generate speech audio from text using Google's Gemini text-to-speech models. " +
    "Supports multiple voices and speech styles.";

  defaults() {
    return {
      text: "",
      model: "gemini-2.5-pro-preview-tts",
      voice_name: "kore",
      style_prompt: "",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getGeminiApiKey(inputs);
    const text = String(inputs.text ?? this._props.text ?? "");
    const model = String(inputs.model ?? this._props.model ?? "gemini-2.5-pro-preview-tts");
    const voiceName = String(inputs.voice_name ?? this._props.voice_name ?? "kore").toLowerCase();
    const stylePrompt = String(inputs.style_prompt ?? this._props.style_prompt ?? "");

    if (!text) throw new Error("The input text cannot be empty.");

    let content = text;
    if (stylePrompt) {
      content = `${stylePrompt}: ${text}`;
    }

    const url = `${GEMINI_API_BASE}/models/${model}:generateContent?key=${apiKey}`;
    const body = {
      contents: [{ parts: [{ text: content }] }],
      generationConfig: {
        responseModalities: ["AUDIO"],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: {
              voiceName: voiceName,
            },
          },
        },
      },
    };

    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Gemini API error ${res.status}: ${errText}`);
    }

    const data = (await res.json()) as Record<string, unknown>;
    const candidates = data.candidates as Array<Record<string, unknown>> | undefined;
    if (!candidates || candidates.length === 0) {
      throw new Error("No audio generated from the text-to-speech request");
    }

    const candidate = candidates[0];
    const contentObj = candidate.content as Record<string, unknown> | undefined;
    const parts = contentObj?.parts as Array<Record<string, unknown>> | undefined;
    if (!parts) throw new Error("No audio generated from the text-to-speech request");

    for (const part of parts) {
      const inlineData =
        (part.inlineData as Record<string, unknown> | undefined) ??
        (part.inline_data as Record<string, unknown> | undefined);
      if (inlineData && typeof inlineData.data === "string") {
        // The API returns raw PCM audio at 24kHz, 16-bit mono
        // Encode as WAV for the audio ref
        const pcmBase64 = inlineData.data;
        const pcmBytes = Buffer.from(pcmBase64, "base64");
        const wavBytes = encodeWav(pcmBytes, 24000, 1, 16);
        return {
          output: {
            uri: "",
            data: Buffer.from(wavBytes).toString("base64"),
          },
        };
      }
    }

    throw new Error("No audio data found in the response");
  }
}

export class TranscribeGeminiNode extends BaseNode {
  static readonly nodeType = "gemini.audio.Transcribe";
  static readonly title = "Transcribe";
  static readonly description =
    "Transcribe audio to text using Google's Gemini multimodal models. " +
    "Supports various audio formats and provides accurate speech-to-text transcription.";

  defaults() {
    return {
      audio: { uri: "", data: null, asset_id: null },
      model: "gemini-2.5-flash",
      prompt:
        "Transcribe the following audio accurately. Return only the transcription text without any additional commentary.",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getGeminiApiKey(inputs);
    const audio = (inputs.audio ?? this._props.audio ?? {}) as Record<string, unknown>;
    const model = String(inputs.model ?? this._props.model ?? "gemini-2.5-flash");
    const prompt = String(
      inputs.prompt ??
        this._props.prompt ??
        "Transcribe the following audio accurately. Return only the transcription text without any additional commentary."
    );

    if (!isRefSet(audio)) throw new Error("Audio file is required for transcription");

    const audioBytes = getAudioBytes(audio);

    // Detect MIME type from first bytes (simple magic-number check)
    const mimeType = detectAudioMime(audioBytes);

    const url = `${GEMINI_API_BASE}/models/${model}:generateContent?key=${apiKey}`;
    const body = {
      contents: [
        {
          parts: [
            { text: prompt },
            {
              inline_data: {
                mime_type: mimeType,
                data: Buffer.from(audioBytes).toString("base64"),
              },
            },
          ],
        },
      ],
      generationConfig: {
        responseModalities: ["TEXT"],
      },
    };

    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Gemini API error ${res.status}: ${errText}`);
    }

    const data = (await res.json()) as Record<string, unknown>;
    const candidates = data.candidates as Array<Record<string, unknown>> | undefined;
    if (!candidates || candidates.length === 0) {
      throw new Error("No transcription generated from the audio");
    }

    const candidate = candidates[0];
    const content = candidate.content as Record<string, unknown> | undefined;
    const parts = content?.parts as Array<Record<string, unknown>> | undefined;
    if (!parts) throw new Error("No transcription generated from the audio");

    const transcriptionParts: string[] = [];
    for (const part of parts) {
      if (typeof part.text === "string") {
        transcriptionParts.push(part.text);
      }
    }

    return { output: transcriptionParts.join("") };
  }
}

// ── Utility functions ───────────────────────────────────────────────────────

function detectAudioMime(bytes: Uint8Array): string {
  if (bytes.length < 4) return "audio/mpeg";
  // WAV: RIFF header
  if (bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46) {
    return "audio/wav";
  }
  // FLAC
  if (bytes[0] === 0x66 && bytes[1] === 0x4c && bytes[2] === 0x61 && bytes[3] === 0x43) {
    return "audio/flac";
  }
  // OGG
  if (bytes[0] === 0x4f && bytes[1] === 0x67 && bytes[2] === 0x67 && bytes[3] === 0x53) {
    return "audio/ogg";
  }
  // MP3 frame sync or ID3 tag
  if ((bytes[0] === 0xff && (bytes[1] & 0xe0) === 0xe0) || (bytes[0] === 0x49 && bytes[1] === 0x44 && bytes[2] === 0x33)) {
    return "audio/mpeg";
  }
  return "audio/mpeg";
}

function encodeWav(
  pcmData: Uint8Array | Buffer,
  sampleRate: number,
  channels: number,
  bitsPerSample: number
): Uint8Array {
  const byteRate = (sampleRate * channels * bitsPerSample) / 8;
  const blockAlign = (channels * bitsPerSample) / 8;
  const dataSize = pcmData.length;
  const headerSize = 44;
  const buffer = new ArrayBuffer(headerSize + dataSize);
  const view = new DataView(buffer);
  const out = new Uint8Array(buffer);

  // RIFF header
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, "WAVE");

  // fmt chunk
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true); // chunk size
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);

  // data chunk
  writeString(view, 36, "data");
  view.setUint32(40, dataSize, true);
  out.set(pcmData instanceof Buffer ? new Uint8Array(pcmData) : pcmData, headerSize);

  return out;
}

function writeString(view: DataView, offset: number, str: string): void {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

// ── Exports ─────────────────────────────────────────────────────────────────

export const GEMINI_NODES: readonly NodeClass[] = [
  GroundedSearchNode,
  EmbeddingNode,
  ImageGenerationNode,
  TextToVideoGeminiNode,
  ImageToVideoGeminiNode,
  TextToSpeechGeminiNode,
  TranscribeGeminiNode,
];
