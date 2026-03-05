import { BaseNode } from "@nodetool/node-sdk";
import type { NodeClass } from "@nodetool/node-sdk";

type ImageRefLike = { data?: string | Uint8Array; uri?: string };

function getApiKey(inputs: Record<string, unknown>): string {
  const key =
    (inputs._secrets as Record<string, string>)?.MISTRAL_API_KEY ||
    process.env.MISTRAL_API_KEY ||
    "";
  if (!key) throw new Error("Mistral API key not configured");
  return key;
}

function imageToDataUri(image: ImageRefLike): string {
  if (typeof image.data === "string") {
    return `data:image/png;base64,${image.data}`;
  }
  if (image.data instanceof Uint8Array) {
    return `data:image/png;base64,${Buffer.from(image.data).toString("base64")}`;
  }
  if (image.uri) return image.uri;
  throw new Error("Image must include data or uri");
}

async function mistralPost(
  apiKey: string,
  endpoint: string,
  body: Record<string, unknown>
): Promise<Record<string, unknown>> {
  const res = await fetch(`https://api.mistral.ai/v1/${endpoint}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Mistral API error ${res.status}: ${text}`);
  }
  return (await res.json()) as Record<string, unknown>;
}

// ── Chat Completion ─────────────────────────────────────────────────────────

export class ChatComplete extends BaseNode {
  static readonly nodeType = "mistral.text.ChatComplete";
  static readonly title = "Mistral Chat";
  static readonly description =
    "Generate text using Mistral AI's chat completion models. " +
    "Supports multiple models including Mistral Large, Small, Pixtral, and Codestral.";

  defaults() {
    return {
      model: "mistral-small-latest",
      prompt: "",
      system_prompt: "",
      temperature: 0.7,
      max_tokens: 1024,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? this._props.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");

    const model = String(inputs.model ?? this._props.model ?? "mistral-small-latest");
    const systemPrompt = String(inputs.system_prompt ?? this._props.system_prompt ?? "");
    const temperature = Number(inputs.temperature ?? this._props.temperature ?? 0.7);
    const maxTokens = Number(inputs.max_tokens ?? this._props.max_tokens ?? 1024);

    const messages: Record<string, unknown>[] = [];
    if (systemPrompt) messages.push({ role: "system", content: systemPrompt });
    messages.push({ role: "user", content: prompt });

    const data = await mistralPost(apiKey, "chat/completions", {
      model,
      messages,
      temperature,
      max_tokens: maxTokens,
    });

    const choices = data.choices as { message: { content: string | null } }[] | undefined;
    if (!choices || choices.length === 0) throw new Error("No response from Mistral API");

    return { output: choices[0].message.content ?? "" };
  }
}

// ── Code Completion ─────────────────────────────────────────────────────────

export class CodeComplete extends BaseNode {
  static readonly nodeType = "mistral.text.CodeComplete";
  static readonly title = "Mistral Code";
  static readonly description =
    "Generate code using Mistral AI's Codestral model. " +
    "Supports fill-in-the-middle (FIM) for code completion tasks.";

  defaults() {
    return {
      prompt: "",
      suffix: "",
      temperature: 0.0,
      max_tokens: 2048,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? this._props.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");

    const suffix = String(inputs.suffix ?? this._props.suffix ?? "");
    const temperature = Number(inputs.temperature ?? this._props.temperature ?? 0.0);
    const maxTokens = Number(inputs.max_tokens ?? this._props.max_tokens ?? 2048);
    const model = "codestral-latest";

    let data: Record<string, unknown>;

    if (suffix) {
      data = await mistralPost(apiKey, "fim/completions", {
        model,
        prompt,
        suffix,
        temperature,
        max_tokens: maxTokens,
      });
    } else {
      data = await mistralPost(apiKey, "chat/completions", {
        model,
        messages: [{ role: "user", content: prompt }],
        temperature,
        max_tokens: maxTokens,
      });
    }

    const choices = data.choices as { message: { content: string | null } }[] | undefined;
    if (!choices || choices.length === 0) throw new Error("No response from Mistral API");

    return { output: choices[0].message.content ?? "" };
  }
}

// ── Embeddings ──────────────────────────────────────────────────────────────

export class Embedding extends BaseNode {
  static readonly nodeType = "mistral.embeddings.Embedding";
  static readonly title = "Mistral Embedding";
  static readonly description =
    "Generate vector embeddings using Mistral AI. " +
    "Creates dense vector representations of text for semantic search, clustering, and similarity.";

  defaults() {
    return {
      input: "",
      model: "mistral-embed",
      chunk_size: 4096,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const input = String(inputs.input ?? this._props.input ?? "");
    if (!input) throw new Error("Input text cannot be empty");

    const model = String(inputs.model ?? this._props.model ?? "mistral-embed");
    const chunkSize = Number(inputs.chunk_size ?? this._props.chunk_size ?? 4096);

    // Chunk the input
    const chunks: string[] = [];
    for (let i = 0; i < input.length; i += chunkSize) {
      chunks.push(input.slice(i, i + chunkSize));
    }

    const data = await mistralPost(apiKey, "embeddings", {
      model,
      input: chunks,
    });

    const embData = data.data as { embedding: number[] }[] | undefined;
    if (!embData || embData.length === 0) throw new Error("No embeddings from Mistral API");

    // Average embeddings if multiple chunks
    const dim = embData[0].embedding.length;
    const avg = new Array<number>(dim).fill(0);
    for (const item of embData) {
      for (let i = 0; i < dim; i++) {
        avg[i] += item.embedding[i];
      }
    }
    if (embData.length > 1) {
      for (let i = 0; i < dim; i++) {
        avg[i] /= embData.length;
      }
    }

    return { output: { data: avg, shape: [dim], dtype: "float32" } };
  }
}

// ── Image to Text ───────────────────────────────────────────────────────────

export class ImageToText extends BaseNode {
  static readonly nodeType = "mistral.vision.ImageToText";
  static readonly title = "Mistral Image to Text";
  static readonly description =
    "Analyze images and generate text descriptions using Mistral AI's Pixtral models. " +
    "Can perform OCR, image analysis, and answer questions about images.";

  defaults() {
    return {
      image: {},
      prompt: "Describe this image in detail.",
      model: "pixtral-large-latest",
      temperature: 0.3,
      max_tokens: 1024,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const image = (inputs.image ?? this._props.image ?? {}) as ImageRefLike;
    if (!image.data && !image.uri) throw new Error("Image is required");

    const prompt = String(inputs.prompt ?? this._props.prompt ?? "Describe this image in detail.");
    if (!prompt) throw new Error("Prompt cannot be empty");

    const model = String(inputs.model ?? this._props.model ?? "pixtral-large-latest");
    const temperature = Number(inputs.temperature ?? this._props.temperature ?? 0.3);
    const maxTokens = Number(inputs.max_tokens ?? this._props.max_tokens ?? 1024);

    const dataUri = imageToDataUri(image);

    const data = await mistralPost(apiKey, "chat/completions", {
      model,
      messages: [
        {
          role: "user",
          content: [
            { type: "image_url", image_url: { url: dataUri } },
            { type: "text", text: prompt },
          ],
        },
      ],
      temperature,
      max_tokens: maxTokens,
    });

    const choices = data.choices as { message: { content: string | null } }[] | undefined;
    if (!choices || choices.length === 0) throw new Error("No response from Mistral API");

    return { output: choices[0].message.content ?? "" };
  }
}

// ── OCR ─────────────────────────────────────────────────────────────────────

export class OCR extends BaseNode {
  static readonly nodeType = "mistral.vision.OCR";
  static readonly title = "Mistral OCR";
  static readonly description =
    "Extract text from images using Mistral AI's Pixtral models. " +
    "Optimized for OCR on documents, screenshots, and printed materials.";

  defaults() {
    return {
      image: {},
      model: "pixtral-large-latest",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const image = (inputs.image ?? this._props.image ?? {}) as ImageRefLike;
    if (!image.data && !image.uri) throw new Error("Image is required");

    const model = String(inputs.model ?? this._props.model ?? "pixtral-large-latest");
    const dataUri = imageToDataUri(image);

    const data = await mistralPost(apiKey, "chat/completions", {
      model,
      messages: [
        {
          role: "user",
          content: [
            { type: "image_url", image_url: { url: dataUri } },
            {
              type: "text",
              text:
                "Extract and return all text visible in this image. " +
                "Preserve the original formatting and structure as much as possible. " +
                "Return only the extracted text without any additional commentary.",
            },
          ],
        },
      ],
      temperature: 0.0,
      max_tokens: 8192,
    });

    const choices = data.choices as { message: { content: string | null } }[] | undefined;
    if (!choices || choices.length === 0) throw new Error("No response from Mistral API");

    return { output: choices[0].message.content ?? "" };
  }
}

// ── Export ───────────────────────────────────────────────────────────────────

export const MISTRAL_NODES: readonly NodeClass[] = [
  ChatComplete,
  CodeComplete,
  Embedding,
  ImageToText,
  OCR,
];
