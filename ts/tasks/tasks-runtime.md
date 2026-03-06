# Runtime / Providers Tasks — `packages/runtime`

Parity gaps between `src/nodetool/providers/` + `src/nodetool/chat/` (Python) and `ts/packages/runtime/src/` (TypeScript).

---

## Phase 1 — Critical missing capabilities

### T-RT-1 · Embeddings support
**Status:** 🔴 open
**Python:** `providers/openai/provider.py` — `create_embeddings()`, `OpenAIEmbeddingProvider`
**TS:** Zero embedding support in any provider.

- [ ] **TEST** — Add `createEmbeddings(texts: string[], model: string): Promise<number[][]>` to `BaseProvider` interface. Write test: calling `openaiProvider.createEmbeddings(["hello"])` returns array of float arrays.
- [ ] **IMPL** — Implement `createEmbeddings()` in `OpenAIProvider` using `client.embeddings.create()`.
- [ ] **TEST** — Write test: Ollama provider supports embeddings via `/api/embeddings` endpoint.
- [ ] **IMPL** — Implement `createEmbeddings()` in `OllamaProvider`.

---

### T-RT-2 · Vision / multimodal inputs
**Status:** 🔴 open
**Python:** All major providers accept `ImageRef` content items in messages.
**TS:** `MessageContent` type exists but image inputs are not forwarded to any provider.

- [ ] **TEST** — Write test: `AnthropicProvider.chat([{ role: "user", content: [{ type: "image_url", url: "..." }] }])` sends image in API request body.
- [ ] **IMPL** — Update `AnthropicProvider` to map `image_url` and `image_base64` content items to Anthropic's vision format.
- [ ] **TEST** — Write test: `OpenAIProvider.chat` with image content sends correct `content` array to API.
- [ ] **IMPL** — Update `OpenAIProvider` to map image content items to OpenAI vision format.
- [ ] **TEST** — Write test: `GeminiProvider.chat` with image content sends `inlineData` parts.
- [ ] **IMPL** — Update `GeminiProvider` to map image content to Gemini multimodal format.

---

### T-RT-3 · OpenAI image generation
**Status:** 🔴 open
**Python:** `providers/openai/provider.py` — `generate_image()`, DALL-E integration
**TS:** Not implemented.

- [ ] **TEST** — Write test: `openaiProvider.generateImage({ prompt, model, size })` returns URL or base64.
- [ ] **IMPL** — Add `generateImage()` to `OpenAIProvider` using `client.images.generate()`.
- [ ] **IMPL** — Add `generateImage?()` to `BaseProvider` interface as optional.

---

## Phase 2 — Provider feature completeness

### T-RT-4 · OpenAI speech / TTS
**Status:** 🔴 open
**Python:** `providers/openai/provider.py` — `text_to_speech()`, `transcribe_audio()`

- [ ] **TEST** — Write test: `openaiProvider.textToSpeech({ text, voice, model })` returns audio buffer.
- [ ] **IMPL** — Add `textToSpeech()` to `OpenAIProvider` using `client.audio.speech.create()`.
- [ ] **TEST** — Write test: `openaiProvider.transcribeAudio(buffer, model)` returns text.
- [ ] **IMPL** — Add `transcribeAudio()` using `client.audio.transcriptions.create()`.

---

### T-RT-5 · Anthropic extended thinking
**Status:** 🔴 open
**Python:** Anthropic provider supports `thinking` parameter with budget tokens.
**TS:** Not mapped in request serialization.

- [ ] **TEST** — Write test: when `thinking: { type: "enabled", budget_tokens: 5000 }` is passed in options, it appears in the Anthropic API request body.
- [ ] **IMPL** — Thread `thinking` option through `AnthropicProvider.chat()` into request body.

---

### T-RT-6 · Gemini multimodal (video, audio)
**Status:** 🔴 open
**Python:** Gemini provider handles video/audio content parts.
**TS:** Only text content mapped.

- [ ] **TEST** — Write test: video URL content item maps to Gemini `fileData` part.
- [ ] **IMPL** — Extend Gemini content mapper to handle `video_url`, `audio_url` content types.

---

### T-RT-7 · Provider model listing
**Status:** 🔴 open
**Python:** Most providers implement `get_models()` returning available models.
**TS:** `BaseProvider` has no `listModels()` method.

- [ ] **TEST** — Write test: `openaiProvider.listModels()` returns array of model descriptors with id and name.
- [ ] **IMPL** — Add `listModels?(): Promise<ModelInfo[]>` to `BaseProvider` interface.
- [ ] **IMPL** — Implement `listModels()` in `OpenAIProvider`, `AnthropicProvider`, `OllamaProvider`.

---

### T-RT-8 · Mistral full feature parity
**Status:** 🔴 open
**Python:** 534 LOC — includes streaming, function calling, safe mode, random seed.
**TS:** 96 LOC — minimal.

- [ ] **TEST** — Write test: streaming Mistral response streams correctly.
- [ ] **TEST** — Write test: tool/function calling round-trips correctly.
- [ ] **IMPL** — Extend `MistralProvider` with streaming, tool calling, safe_mode, random_seed.

---

### T-RT-9 · OpenRouter full feature parity
**Status:** 🔴 open
**Python:** 753 LOC — includes model routing, provider preferences, transforms.
**TS:** 95 LOC — minimal.

- [ ] **TEST** — Write test: OpenRouter-specific headers (`HTTP-Referer`, `X-Title`) are sent.
- [ ] **TEST** — Write test: provider preferences in request body are forwarded.
- [ ] **IMPL** — Extend `OpenRouterProvider` with OpenRouter-specific request options.

---

### T-RT-10 · Together / Cerebras / LM Studio full parity
**Status:** 🔴 open (low priority)
**Python:** Each has 400–500 LOC with streaming, function calling, model listing.
**TS:** Each has ~85–99 LOC (basic chat only).

- [ ] **TEST** — Write streaming test for each provider.
- [ ] **TEST** — Write tool-calling test for each provider.
- [ ] **IMPL** — Extend each to full feature set (streaming, tools, model list).

---

## Phase 3 — New providers

### T-RT-11 · HuggingFace provider
**Status:** 🔴 open
**Python:** 1,443 LOC — complex inference API with model loading, task routing, pipeline support.

- [ ] **TEST** — Write test: `HuggingFaceProvider.chat()` calls HF inference API with correct Authorization header.
- [ ] **IMPL** — Port core `HuggingFaceProvider` using `@huggingface/inference` npm package.

---

### T-RT-12 · Chat infrastructure — CLI and persistence
**Status:** ⚪ deferred
**Python:** `chat/` has a full CLI (1,400 LOC), Server-Sent Events transport, command system, and DB-backed message persistence.
**TS:** Only core message loop (`message-processor.ts`).

Deferred until a TS CLI is scoped. Partially covered by the existing `cli` package.

---

### T-RT-13 · ComfyUI providers (N/A for now)
**Status:** ⚪ deferred — local image gen infra, Python-specific.

### T-RT-14 · Meshy / Rodin 3D generation (N/A for now)
**Status:** ⚪ deferred — specialized, low demand.

### T-RT-15 · KIE / MiniMax / ZAI / AIME providers
**Status:** ⚪ deferred — regional/specialized, low demand in TS.

---

## Context packer

### T-RT-16 · context_packer port
**Status:** 🟢 done
**Python source:** `messaging/context_packer.py` — serializes conversation history + system prompt into a token-limited context window.

- [ ] **TEST** — Write test: `packContext(messages, systemPrompt, maxTokens)` truncates oldest messages to fit within token budget.
- [ ] **IMPL** — Port `context_packer.py` to `ts/packages/runtime/src/context-packer.ts`. Use existing `token-counter.ts`.
