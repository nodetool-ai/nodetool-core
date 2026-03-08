# Runtime / Providers Tasks — `packages/runtime`

Parity gaps between `src/nodetool/providers/` + `src/nodetool/chat/` (Python) and `ts/packages/runtime/src/` (TypeScript).

---

## Completed

| ID | Feature | Status |
|----|---------|--------|
| T-RT-16 | Context packer | 🟢 done |
| — | Anthropic retry logic for model listing | 🟢 done |
| — | BaseProvider `isRateLimitError()`, `isAuthError()` | 🟢 done |
| — | Async `hasToolSupport()` across all 12 providers | 🟢 done |
| — | Ollama `/api/show` capability detection + tool emulation | 🟢 done |
| — | OpenAI `defaultSerializer` for tool args | 🟢 done |
| — | New param types (guidanceScale, seed, scheduler, etc.) | 🟢 done |
| — | 3D model types (Model3D, TextTo3DParams, ImageTo3DParams) | 🟢 done |

---

## Phase 1 — Critical missing capabilities

### T-RT-1 · Embeddings support
**Status:** 🟢 done — `generateEmbedding()` implemented in `OpenAIProvider` and `OllamaProvider`.

---

### T-RT-2 · Vision / multimodal inputs
**Status:** 🟢 done — All major providers handle image content items.

---

### T-RT-3 · OpenAI image generation
**Status:** 🟢 done — `textToImage()` implemented in `OpenAIProvider`.

---

## Phase 2 — Provider feature completeness

### T-RT-4 · OpenAI speech / TTS
**Status:** 🟢 done — `textToSpeech()` and `automaticSpeechRecognition()` implemented in `OpenAIProvider`.

### T-RT-5 · Anthropic extended thinking
**Status:** 🟢 done — `thinking` option threaded through `AnthropicProvider.generateMessages()`.

### T-RT-6 · Gemini multimodal (video, audio)
**Status:** 🟢 done — Gemini content mapper handles video/audio. Full modality support: embeddings, text-to-image (Gemini + Imagen), image-to-image, TTS (30 voices), ASR (audio header detection), text-to-video (Veo async polling), image-to-video, plus all model listing methods.

### T-RT-7 · Provider model listing
**Status:** 🟢 done — All providers implement `getAvailableLanguageModels()`.

### T-RT-8 · Mistral full feature parity
**Status:** 🔴 open (low priority)

### T-RT-9 · OpenRouter full feature parity
**Status:** 🔴 open (low priority)

### T-RT-10 · Together / Cerebras / LM Studio full parity
**Status:** 🔴 open (low priority)

---

## Phase 3 — New providers

### T-RT-11 · HuggingFace provider
**Status:** 🔴 open (low priority)

---

## Deferred

| Task | Reason |
|------|--------|
| T-RT-12 Chat infrastructure | Partially covered by `cli` package |
| T-RT-13 ComfyUI providers | Python-specific |
| T-RT-14 Meshy / Rodin 3D | Low demand |
| T-RT-15 KIE / MiniMax / ZAI / AIME | Regional/specialized |
