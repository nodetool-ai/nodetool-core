[← Back to Docs Index](../../../docs/index.md)

# Nodetool Chat Providers

This directory contains implementations for various chat completion providers used by the `nodetool.chat` module. These
providers act as adapters, translating between Nodetool's internal chat message format and the specific API requirements
of different large language models (LLMs) like OpenAI, Anthropic Claude, Google Gemini, and locally hosted models via
Ollama.

## Core Concept: `ChatProvider`

The foundation is the abstract base class `ChatProvider` defined in `base.py`. It establishes a common interface that
all specific provider implementations must adhere to. This ensures that the chat module can interact with any supported
LLM provider in a consistent manner.

The key methods defined by `ChatProvider` are:

- `generate_message`: Generates a single, non-streaming response from the LLM.
- `generate_messages`: Generates a streaming response, yielding chunks of text (`Chunk`) or requests to call tools
  (`ToolCall`) as they become available.

Providers are responsible for:

1. **Message Conversion:** Translating Nodetool's `Message` objects (including roles like 'user', 'assistant', 'system',
   'tool', and potentially multimodal content) into the format expected by the target LLM API.
1. **API Interaction:** Handling the actual HTTP requests to the LLM API endpoint.
1. **Streaming:** Managing the asynchronous generation and yielding of response chunks.
1. **Tool/Function Calling:** Formatting tool definitions for the LLM and parsing tool call requests from the LLM's
   response.
1. **Configuration:** Managing API keys and other provider-specific settings (often via environment variables).
1. **Usage Tracking:** Optionally tracking token usage.

## Implemented Providers

The following providers are currently implemented:

- **OpenAI (`openai_provider.py`):** Connects to OpenAI's API (e.g., GPT-4, GPT-3.5).
- **Anthropic (`anthropic_provider.py`):** Connects to Anthropic's Claude API.
- **Gemini (`gemini_provider.py`):** Connects to Google's Gemini API.
- **Ollama (`ollama_provider.py`):** Connects to a locally running Ollama instance, allowing the use of various
  open-source models.
- **vLLM (`vllm_provider.py`):** Connects to an externally managed vLLM server that exposes an OpenAI-compatible API.
- **xllamacpp (`xllamacpp_provider.py`):** High-performance local LLM inference using xllamacpp's Cython bindings.
  Provides built-in GPU offloading with memory estimation, thread-safe batching, and supports multiple platforms
  (CPU, CUDA, Vulkan, Metal).

## Feature Support Summary

| Feature                      | OpenAI (`openai_provider.py`) | Anthropic (`anthropic_provider.py`) | Gemini (`gemini_provider.py`) | Ollama (`ollama_provider.py`)             | vLLM (`vllm_provider.py`)                  | xllamacpp (`xllamacpp_provider.py`)       |
| :--------------------------- | :---------------------------- | :---------------------------------- | :---------------------------- | :---------------------------------------- | :----------------------------------------- | :---------------------------------------- |
| **Streaming**                | Yes ✅                        | Yes ✅                              | Yes ✅                        | Yes ✅                                    | Yes ✅                                     | Yes ✅                                    |
| **Tool Use (Native)**        | Yes ✅                        | Yes ✅                              | Yes ✅                        | Yes ✅ (Model Dependent)                  | Yes ✅ (Model Dependent)                   | No ❌                                     |
| **Tool Use (Textual)**       | No                            | No                                  | No                            | Yes ✅ (Fallback for incompatible models) | No                                         | Yes ✅ (Emulation)                        |
| **System Prompt**            | Yes ✅                        | Yes ✅                              | Yes ✅                        | Yes ✅                                    | Yes ✅                                     | Yes ✅                                    |
| **Image Input (Multimodal)** | Yes ✅                        | Yes ✅                              | No ❌ (File input only)       | Yes ✅ (Base64)                           | Yes ✅ (OpenAI format)                     | Yes ✅ (Base64, model dependent)          |
| **File Input (Generic)**     | No ❌                         | No ❌                               | Yes ✅ (Via Blobs)            | No ❌                                     | No ❌                                      | No ❌                                     |
| **JSON Mode**                | Yes ✅                        | Yes ✅ (Via Tool)                   | Yes ✅                        | Yes ✅ (Model Dependent)                  | Yes ✅ (Model Dependent)                   | Yes ✅ (Grammar support)                  |
| **API Key Required**         | Yes ✅                        | Yes ✅                              | Yes ✅                        | Optional                                  | Optional                                   | No ❌                                     |
| **Backend Type**             | Cloud ☁️                      | Cloud ☁️                            | Cloud ☁️                      | Local/Self-Hosted 🏠                      | Local/Self-Hosted / Cloud 🏠☁️             | Local 🏠                                  |
| **Configuration**            | `OPENAI_API_KEY`              | `ANTHROPIC_API_KEY`                 | `GEMINI_API_KEY`              | `OLLAMA_API_URL`                          | `VLLM_BASE_URL`, `VLLM_API_KEY` (optional) | `XLLAMACPP_*` environment variables       |
| **GPU Offloading**           | N/A                           | N/A                                 | N/A                           | External                                  | External                                   | Yes ✅ (Auto-estimated)                   |
| **Memory Estimation**        | N/A                           | N/A                                 | N/A                           | No                                        | No                                         | Yes ✅                                    |

**Notes:**

- **Ollama Tool Use:** Native tool use depends on the specific Ollama model. A textual fallback mechanism (prompting the
  model to generate specific text for tool calls) is available.
- **vLLM Hosting:** vLLM instances are managed outside Nodetool; configure connection details via environment variables
  like `VLLM_BASE_URL` and `VLLM_API_KEY`.
- **Ollama JSON Mode:** Support depends on the specific Ollama model's ability to follow formatting instructions.
- **Anthropic JSON Mode:** Implemented by instructing the model to use a predefined "json_output" tool.
- **Gemini File Input:** Supports generic file input via `Blob` data, not just images embedded directly in message
  content like other providers. Image content within messages is not implemented.
- **Multimodal:** Refers to including images directly alongside text within a single message turn.
- **xllamacpp Provider:** High-performance local LLM inference using Cython-based llama.cpp bindings. Key features:
  - **Pythonic Management:** No external process management - models loaded directly in Python
  - **Memory Estimation:** Automatic GPU layer offloading using built-in memory estimation
  - **Platform Support:** Optimized builds for CPU, CUDA (NVIDIA), Vulkan (AMD/Intel), and Metal (Apple Silicon)
  - **Thread Safety:** Built-in continuous batching with thread-safe server
  - **Installation:** Install platform-specific builds via pip with custom indexes (see pyproject.toml)
  - **Tool Calling:** Uses text-based emulation (similar to Ollama fallback)
  - **Configuration:** Use `XLLAMACPP_*` environment variables to control context size, GPU layers, threads, etc.

## xllamacpp Installation

xllamacpp provides pre-built wheels for multiple platforms. Choose the appropriate installation command:

**CPU / Mac (default):**
```bash
pip install xllamacpp
```

**CUDA 12.8 (NVIDIA GPUs):**
```bash
pip install xllamacpp --index-url https://xorbitsai.github.io/xllamacpp/whl/cu128
```

**CUDA 12.4 (NVIDIA GPUs):**
```bash
pip install xllamacpp --index-url https://xorbitsai.github.io/xllamacpp/whl/cu124
```

**Vulkan (AMD/Intel GPUs):**
```bash
pip install xllamacpp --index-url https://xorbitsai.github.io/xllamacpp/whl/vulkan
```

**Note:** For optimal performance on specific CPU architectures, you may want to build from source. See the
[xllamacpp documentation](https://github.com/xorbitsai/xllamacpp) for details.

This provider system allows Nodetool to flexibly leverage the capabilities of different LLMs for chat-based interactions
and agentic workflows.
