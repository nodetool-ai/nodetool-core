import { vi, describe, it, expect, beforeEach, afterEach } from "vitest";
import {
  GroundedSearchNode,
  EmbeddingNode,
  ImageGenerationNode,
  TextToVideoGeminiNode,
  ImageToVideoGeminiNode,
  TextToSpeechGeminiNode,
  TranscribeGeminiNode,
} from "../src/nodes/gemini.js";

const originalFetch = globalThis.fetch;
let mockFetch: ReturnType<typeof vi.fn>;

beforeEach(() => {
  mockFetch = vi.fn();
  globalThis.fetch = mockFetch;
  delete process.env.GEMINI_API_KEY;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  delete process.env.GEMINI_API_KEY;
});

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as unknown as Response;
}

const secrets = { _secrets: { GEMINI_API_KEY: "test-key" } };

// ── GroundedSearchNode ─────────────────────────────────────────────────────

describe("GroundedSearchNode", () => {
  it("returns results and sources", async () => {
    const node = new GroundedSearchNode();
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        candidates: [
          {
            content: { parts: [{ text: "Search result text" }] },
            groundingMetadata: {
              groundingChunks: [
                { web: { title: "Source 1", uri: "https://example.com" } },
              ],
            },
          },
        ],
      })
    );
    const result = await node.process({ query: "test query", ...secrets });
    expect(result.results).toEqual(["Search result text"]);
    expect(result.sources).toEqual([
      { title: "Source 1", url: "https://example.com" },
    ]);
  });

  it("deduplicates sources by url", async () => {
    const node = new GroundedSearchNode();
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        candidates: [
          {
            content: { parts: [{ text: "text" }] },
            groundingMetadata: {
              groundingChunks: [
                { web: { title: "A", uri: "https://a.com" } },
                { web: { title: "A dup", uri: "https://a.com" } },
                { web: { title: "B", uri: "https://b.com" } },
              ],
            },
          },
        ],
      })
    );
    const result = await node.process({ query: "test", ...secrets });
    expect((result.sources as unknown[]).length).toBe(2);
  });

  it("handles missing groundingMetadata", async () => {
    const node = new GroundedSearchNode();
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        candidates: [{ content: { parts: [{ text: "result" }] } }],
      })
    );
    const result = await node.process({ query: "test", ...secrets });
    expect(result.results).toEqual(["result"]);
    expect(result.sources).toEqual([]);
  });

  it("throws on empty query", async () => {
    const node = new GroundedSearchNode();
    await expect(
      node.process({ query: "", ...secrets })
    ).rejects.toThrow("Search query is required");
  });

  it("throws when API key missing", async () => {
    const node = new GroundedSearchNode();
    await expect(node.process({ query: "test" })).rejects.toThrow(
      "GEMINI_API_KEY is not configured"
    );
  });

  it("throws on API error", async () => {
    const node = new GroundedSearchNode();
    mockFetch.mockResolvedValueOnce(jsonResponse("err", 500));
    await expect(
      node.process({ query: "test", ...secrets })
    ).rejects.toThrow("Gemini API error 500");
  });

  it("throws on no candidates", async () => {
    const node = new GroundedSearchNode();
    mockFetch.mockResolvedValueOnce(jsonResponse({ candidates: [] }));
    await expect(
      node.process({ query: "test", ...secrets })
    ).rejects.toThrow("No response received from Gemini API");
  });

  it("throws on invalid content format", async () => {
    const node = new GroundedSearchNode();
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ candidates: [{}] })
    );
    await expect(
      node.process({ query: "test", ...secrets })
    ).rejects.toThrow("Invalid response format from Gemini API");
  });
});

// ── EmbeddingNode ──────────────────────────────────────────────────────────

describe("EmbeddingNode (Gemini)", () => {
  it("returns embedding values", async () => {
    const node = new EmbeddingNode();
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        embedding: { values: [0.1, 0.2, 0.3] },
      })
    );
    const result = await node.process({ input: "test", ...secrets });
    expect(result.output).toEqual([0.1, 0.2, 0.3]);
  });

  it("throws on empty input", async () => {
    const node = new EmbeddingNode();
    await expect(
      node.process({ input: "", ...secrets })
    ).rejects.toThrow("Input text is required for embedding generation");
  });

  it("throws when no embedding in response", async () => {
    const node = new EmbeddingNode();
    mockFetch.mockResolvedValueOnce(jsonResponse({}));
    await expect(
      node.process({ input: "test", ...secrets })
    ).rejects.toThrow("No embedding generated from the input text");
  });

  it("throws on API error", async () => {
    const node = new EmbeddingNode();
    mockFetch.mockResolvedValueOnce(jsonResponse("err", 400));
    await expect(
      node.process({ input: "test", ...secrets })
    ).rejects.toThrow("Gemini API error 400");
  });
});

// ── ImageGenerationNode ────────────────────────────────────────────────────

describe("ImageGenerationNode", () => {
  it("generates image with Imagen model (predict endpoint)", async () => {
    const node = new ImageGenerationNode();
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        predictions: [{ bytesBase64Encoded: "base64imagedata" }],
      })
    );
    const result = await node.process({
      prompt: "a cat",
      model: "imagen-3.0-generate-002",
      ...secrets,
    });
    expect(result.output).toEqual({
      type: "image",
      data: "base64imagedata",
    });
  });

  it("generates image with Gemini model (generateContent)", async () => {
    const node = new ImageGenerationNode();
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        candidates: [
          {
            content: {
              parts: [
                { inlineData: { data: "geminiimgdata", mimeType: "image/png" } },
              ],
            },
          },
        ],
      })
    );
    const result = await node.process({
      prompt: "a cat",
      model: "gemini-2.0-flash",
      ...secrets,
    });
    expect(result.output).toEqual({
      type: "image",
      data: "geminiimgdata",
    });
  });

  it("handles snake_case inline_data from API", async () => {
    const node = new ImageGenerationNode();
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        candidates: [
          {
            content: {
              parts: [
                { inline_data: { data: "snakedata" } },
              ],
            },
          },
        ],
      })
    );
    const result = await node.process({
      prompt: "a cat",
      model: "gemini-2.0-flash",
      ...secrets,
    });
    expect(result.output).toEqual({ type: "image", data: "snakedata" });
  });

  it("includes optional input image for Gemini model", async () => {
    const node = new ImageGenerationNode();
    const b64 = Buffer.from("test-image").toString("base64");
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        candidates: [
          {
            content: {
              parts: [{ inlineData: { data: "edited" } }],
            },
          },
        ],
      })
    );
    await node.process({
      prompt: "edit this",
      model: "gemini-2.0-flash",
      image: { data: b64 },
      ...secrets,
    });
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.contents[0].parts.length).toBe(2);
  });

  it("throws on prohibited content", async () => {
    const node = new ImageGenerationNode();
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        candidates: [{ finishReason: "PROHIBITED_CONTENT", content: { parts: [] } }],
      })
    );
    await expect(
      node.process({ prompt: "bad", model: "gemini-2.0-flash", ...secrets })
    ).rejects.toThrow("Prohibited content");
  });

  it("throws when no image bytes in Gemini response", async () => {
    const node = new ImageGenerationNode();
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        candidates: [
          { content: { parts: [{ text: "no image here" }] } },
        ],
      })
    );
    await expect(
      node.process({ prompt: "cat", model: "gemini-2.0-flash", ...secrets })
    ).rejects.toThrow("No image bytes returned in response");
  });

  it("throws on empty prompt", async () => {
    const node = new ImageGenerationNode();
    await expect(
      node.process({ prompt: "", ...secrets })
    ).rejects.toThrow("The input prompt cannot be empty");
  });

  it("throws when no predictions from Imagen", async () => {
    const node = new ImageGenerationNode();
    mockFetch.mockResolvedValueOnce(jsonResponse({ predictions: [] }));
    await expect(
      node.process({ prompt: "cat", model: "imagen-3.0-generate-002", ...secrets })
    ).rejects.toThrow("No images generated");
  });

  it("throws when no bytesBase64 in predictions", async () => {
    const node = new ImageGenerationNode();
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ predictions: [{}] })
    );
    await expect(
      node.process({ prompt: "cat", model: "imagen-3.0-generate-002", ...secrets })
    ).rejects.toThrow("No image bytes in response");
  });

  it("throws on API error for Imagen", async () => {
    const node = new ImageGenerationNode();
    mockFetch.mockResolvedValueOnce(jsonResponse("err", 500));
    await expect(
      node.process({ prompt: "cat", model: "imagen-3.0-generate-002", ...secrets })
    ).rejects.toThrow("Gemini API error 500");
  });

  it("throws on no candidates for Gemini model", async () => {
    const node = new ImageGenerationNode();
    mockFetch.mockResolvedValueOnce(jsonResponse({ candidates: [] }));
    await expect(
      node.process({ prompt: "cat", model: "gemini-2.0-flash", ...secrets })
    ).rejects.toThrow("No response received from Gemini API");
  });
});

// ── TextToVideoGeminiNode ──────────────────────────────────────────────────

describe("TextToVideoGeminiNode", () => {
  it("returns video data from completed operation", async () => {
    const node = new TextToVideoGeminiNode();
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        done: true,
        response: {
          generatedVideos: [
            { video: { videoBytes: "base64videodata" } },
          ],
        },
      })
    );
    const result = await node.process({
      prompt: "a cat playing",
      ...secrets,
    });
    expect(result.output).toEqual({ data: "base64videodata" });
  });

  it("polls operation until done", async () => {
    const node = new TextToVideoGeminiNode();
    // First call: start operation
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ name: "operations/op1", done: false })
    );
    // Poll call: done
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        done: true,
        response: {
          generatedVideos: [
            { video: { videoBytes: "polledvideo" } },
          ],
        },
      })
    );
    const result = await node.process({
      prompt: "a cat",
      ...secrets,
    });
    expect(result.output).toEqual({ data: "polledvideo" });
  }, 15000);

  it("throws on empty prompt", async () => {
    const node = new TextToVideoGeminiNode();
    await expect(
      node.process({ prompt: "", ...secrets })
    ).rejects.toThrow("Video generation prompt is required");
  });

  it("throws on API error", async () => {
    const node = new TextToVideoGeminiNode();
    mockFetch.mockResolvedValueOnce(jsonResponse("err", 500));
    await expect(
      node.process({ prompt: "cat", ...secrets })
    ).rejects.toThrow("Gemini API error 500");
  });

  it("throws when no operation name and not done", async () => {
    const node = new TextToVideoGeminiNode();
    mockFetch.mockResolvedValueOnce(jsonResponse({ done: false }));
    await expect(
      node.process({ prompt: "cat", ...secrets })
    ).rejects.toThrow("No operation name in response");
  });

  it("throws when no video in response", async () => {
    const node = new TextToVideoGeminiNode();
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ done: true, response: { generatedVideos: [] } })
    );
    await expect(
      node.process({ prompt: "cat", ...secrets })
    ).rejects.toThrow("No video generated");
  });
});

// ── ImageToVideoGeminiNode ─────────────────────────────────────────────────

describe("ImageToVideoGeminiNode", () => {
  it("returns video data from image input", async () => {
    const node = new ImageToVideoGeminiNode();
    const b64 = Buffer.from("test-image").toString("base64");
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        done: true,
        response: {
          generatedVideos: [
            { video: { videoBytes: "videofromimage" } },
          ],
        },
      })
    );
    const result = await node.process({
      image: { data: b64 },
      prompt: "animate",
      ...secrets,
    });
    expect(result.output).toEqual({ data: "videofromimage" });
  });

  it("throws when image is not set", async () => {
    const node = new ImageToVideoGeminiNode();
    await expect(
      node.process({ image: {}, prompt: "animate", ...secrets })
    ).rejects.toThrow("Input image is required");
  });

  it("throws when image has no data bytes", async () => {
    const node = new ImageToVideoGeminiNode();
    await expect(
      node.process({
        image: { uri: "https://example.com/img.png", data: "" },
        prompt: "animate",
        ...secrets,
      })
    ).rejects.toThrow();
  });

  it("throws on API error", async () => {
    const node = new ImageToVideoGeminiNode();
    const b64 = Buffer.from("img").toString("base64");
    mockFetch.mockResolvedValueOnce(jsonResponse("err", 400));
    await expect(
      node.process({
        image: { data: b64 },
        prompt: "animate",
        ...secrets,
      })
    ).rejects.toThrow("Gemini API error 400");
  });
});

// ── TextToSpeechGeminiNode ─────────────────────────────────────────────────

describe("TextToSpeechGeminiNode", () => {
  it("returns WAV audio data", async () => {
    const node = new TextToSpeechGeminiNode();
    const pcmBase64 = Buffer.from(new Uint8Array(100)).toString("base64");
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        candidates: [
          {
            content: {
              parts: [{ inlineData: { data: pcmBase64 } }],
            },
          },
        ],
      })
    );
    const result = await node.process({ text: "Hello", ...secrets });
    const output = result.output as Record<string, string>;
    expect(output.data).toBeDefined();
    // Verify it's a valid base64 string containing WAV header
    const decoded = Buffer.from(output.data, "base64");
    expect(decoded[0]).toBe(0x52); // 'R' in RIFF
  });

  it("handles snake_case inline_data", async () => {
    const node = new TextToSpeechGeminiNode();
    const pcmBase64 = Buffer.from(new Uint8Array(10)).toString("base64");
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        candidates: [
          {
            content: {
              parts: [{ inline_data: { data: pcmBase64 } }],
            },
          },
        ],
      })
    );
    const result = await node.process({ text: "Hi", ...secrets });
    expect((result.output as Record<string, string>).data).toBeDefined();
  });

  it("includes style prompt", async () => {
    const node = new TextToSpeechGeminiNode();
    const pcmBase64 = Buffer.from(new Uint8Array(10)).toString("base64");
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        candidates: [
          {
            content: {
              parts: [{ inlineData: { data: pcmBase64 } }],
            },
          },
        ],
      })
    );
    await node.process({
      text: "Hello",
      style_prompt: "Speak cheerfully",
      ...secrets,
    });
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.contents[0].parts[0].text).toContain("Speak cheerfully");
  });

  it("throws on empty text", async () => {
    const node = new TextToSpeechGeminiNode();
    await expect(
      node.process({ text: "", ...secrets })
    ).rejects.toThrow("The input text cannot be empty");
  });

  it("throws on no candidates", async () => {
    const node = new TextToSpeechGeminiNode();
    mockFetch.mockResolvedValueOnce(jsonResponse({ candidates: [] }));
    await expect(
      node.process({ text: "hello", ...secrets })
    ).rejects.toThrow("No audio generated");
  });

  it("throws on no audio data in parts", async () => {
    const node = new TextToSpeechGeminiNode();
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        candidates: [
          { content: { parts: [{ text: "no audio here" }] } },
        ],
      })
    );
    await expect(
      node.process({ text: "hello", ...secrets })
    ).rejects.toThrow("No audio data found in the response");
  });

  it("throws on API error", async () => {
    const node = new TextToSpeechGeminiNode();
    mockFetch.mockResolvedValueOnce(jsonResponse("err", 500));
    await expect(
      node.process({ text: "hello", ...secrets })
    ).rejects.toThrow("Gemini API error 500");
  });
});

// ── TranscribeGeminiNode ───────────────────────────────────────────────────

describe("TranscribeGeminiNode", () => {
  it("returns transcription text", async () => {
    const node = new TranscribeGeminiNode();
    const audioB64 = Buffer.from(new Uint8Array([0xff, 0xfb, 0x90, 0x00])).toString("base64");
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        candidates: [
          {
            content: { parts: [{ text: "Hello world" }] },
          },
        ],
      })
    );
    const result = await node.process({
      audio: { data: audioB64 },
      ...secrets,
    });
    expect(result.output).toBe("Hello world");
  });

  it("concatenates multiple text parts", async () => {
    const node = new TranscribeGeminiNode();
    const audioB64 = Buffer.from(new Uint8Array([0x52, 0x49, 0x46, 0x46])).toString("base64");
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        candidates: [
          {
            content: {
              parts: [{ text: "Hello " }, { text: "world" }],
            },
          },
        ],
      })
    );
    const result = await node.process({
      audio: { data: audioB64 },
      ...secrets,
    });
    expect(result.output).toBe("Hello world");
  });

  it("detects WAV mime type", async () => {
    const node = new TranscribeGeminiNode();
    // RIFF header bytes
    const wavBytes = new Uint8Array([0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0]);
    const audioB64 = Buffer.from(wavBytes).toString("base64");
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        candidates: [{ content: { parts: [{ text: "wav audio" }] } }],
      })
    );
    await node.process({ audio: { data: audioB64 }, ...secrets });
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.contents[0].parts[1].inline_data.mime_type).toBe("audio/wav");
  });

  it("detects FLAC mime type", async () => {
    const node = new TranscribeGeminiNode();
    const flacBytes = new Uint8Array([0x66, 0x4c, 0x61, 0x43, 0, 0]);
    const audioB64 = Buffer.from(flacBytes).toString("base64");
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        candidates: [{ content: { parts: [{ text: "flac" }] } }],
      })
    );
    await node.process({ audio: { data: audioB64 }, ...secrets });
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.contents[0].parts[1].inline_data.mime_type).toBe("audio/flac");
  });

  it("detects OGG mime type", async () => {
    const node = new TranscribeGeminiNode();
    const oggBytes = new Uint8Array([0x4f, 0x67, 0x67, 0x53, 0, 0]);
    const audioB64 = Buffer.from(oggBytes).toString("base64");
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        candidates: [{ content: { parts: [{ text: "ogg" }] } }],
      })
    );
    await node.process({ audio: { data: audioB64 }, ...secrets });
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.contents[0].parts[1].inline_data.mime_type).toBe("audio/ogg");
  });

  it("detects MP3 with ID3 tag", async () => {
    const node = new TranscribeGeminiNode();
    const id3Bytes = new Uint8Array([0x49, 0x44, 0x33, 0x04, 0, 0]);
    const audioB64 = Buffer.from(id3Bytes).toString("base64");
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        candidates: [{ content: { parts: [{ text: "mp3" }] } }],
      })
    );
    await node.process({ audio: { data: audioB64 }, ...secrets });
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.contents[0].parts[1].inline_data.mime_type).toBe("audio/mpeg");
  });

  it("defaults to audio/mpeg for short bytes", async () => {
    const node = new TranscribeGeminiNode();
    const shortBytes = new Uint8Array([0x01, 0x02]);
    const audioB64 = Buffer.from(shortBytes).toString("base64");
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        candidates: [{ content: { parts: [{ text: "short" }] } }],
      })
    );
    await node.process({ audio: { data: audioB64 }, ...secrets });
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.contents[0].parts[1].inline_data.mime_type).toBe("audio/mpeg");
  });

  it("throws when audio not set", async () => {
    const node = new TranscribeGeminiNode();
    await expect(
      node.process({ audio: {}, ...secrets })
    ).rejects.toThrow("Audio file is required for transcription");
  });

  it("throws when audio has no data", async () => {
    const node = new TranscribeGeminiNode();
    await expect(
      node.process({ audio: { uri: "x" }, ...secrets })
    ).rejects.toThrow("Audio data is required");
  });

  it("handles Uint8Array audio data", async () => {
    const node = new TranscribeGeminiNode();
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        candidates: [{ content: { parts: [{ text: "from uint8" }] } }],
      })
    );
    const result = await node.process({
      audio: { data: new Uint8Array([0xff, 0xfb, 0x90, 0x00]) },
      ...secrets,
    });
    expect(result.output).toBe("from uint8");
  });

  it("throws on no candidates", async () => {
    const node = new TranscribeGeminiNode();
    const audioB64 = Buffer.from(new Uint8Array([0xff, 0xfb])).toString("base64");
    mockFetch.mockResolvedValueOnce(jsonResponse({ candidates: [] }));
    await expect(
      node.process({ audio: { data: audioB64 }, ...secrets })
    ).rejects.toThrow("No transcription generated from the audio");
  });

  it("throws on API error", async () => {
    const node = new TranscribeGeminiNode();
    const audioB64 = Buffer.from(new Uint8Array([0xff, 0xfb])).toString("base64");
    mockFetch.mockResolvedValueOnce(jsonResponse("err", 500));
    await expect(
      node.process({ audio: { data: audioB64 }, ...secrets })
    ).rejects.toThrow("Gemini API error 500");
  });
});
