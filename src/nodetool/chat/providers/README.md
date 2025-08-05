# Nodetool Chat Providers

This directory contains implementations for various chat completion providers used by the `nodetool.chat` module. These providers act as adapters, translating between Nodetool's internal chat message format and the specific API requirements of different large language models (LLMs) like OpenAI, Anthropic Claude, Google Gemini, and locally hosted models via Ollama.

## Core Concept: `ChatProvider`

The foundation is the abstract base class `ChatProvider` defined in `base.py`. It establishes a common interface that all specific provider implementations must adhere to. This ensures that the chat module can interact with any supported LLM provider in a consistent manner.

The key methods defined by `ChatProvider` are:

- `generate_message`: Generates a single, non-streaming response from the LLM.
- `generate_messages`: Generates a streaming response, yielding chunks of text (`Chunk`) or requests to call tools (`ToolCall`) as they become available.

Providers are responsible for:

1.  **Message Conversion:** Translating Nodetool's `Message` objects (including roles like 'user', 'assistant', 'system', 'tool', and potentially multimodal content) into the format expected by the target LLM API.
2.  **API Interaction:** Handling the actual HTTP requests to the LLM API endpoint.
3.  **Streaming:** Managing the asynchronous generation and yielding of response chunks.
4.  **Tool/Function Calling:** Formatting tool definitions for the LLM and parsing tool call requests from the LLM's response.
5.  **Configuration:** Managing API keys and other provider-specific settings (often via environment variables).
6.  **Usage Tracking:** Optionally tracking token usage.

## Implemented Providers

The following providers are currently implemented:

- **OpenAI (`openai_provider.py`):** Connects to OpenAI's API (e.g., GPT-4, GPT-3.5).
- **Anthropic (`anthropic_provider.py`):** Connects to Anthropic's Claude API.
- **Gemini (`gemini_provider.py`):** Connects to Google's Gemini API.
- **Ollama (`ollama_provider.py`):** Connects to a locally running Ollama instance, allowing the use of various open-source models.

## Feature Support Summary

| Feature                      | OpenAI (`openai_provider.py`) | Anthropic (`anthropic_provider.py`) | Gemini (`gemini_provider.py`) | Ollama (`ollama_provider.py`)             |
| :--------------------------- | :---------------------------- | :---------------------------------- | :---------------------------- | :---------------------------------------- |
| **Streaming**                | Yes ✅                        | Yes ✅                              | Yes ✅                        | Yes ✅                                    |
| **Tool Use (Native)**        | Yes ✅                        | Yes ✅                              | Yes ✅                        | Yes ✅ (Model Dependent)                  |
| **Tool Use (Textual)**       | No                            | No                                  | No                            | Yes ✅ (Fallback for incompatible models) |
| **System Prompt**            | Yes ✅                        | Yes ✅                              | Yes ✅                        | Yes ✅                                    |
| **Image Input (Multimodal)** | Yes ✅                        | Yes ✅                              | No ❌ (File input only)       | Yes ✅ (Base64)                           |
| **File Input (Generic)**     | No ❌                         | No ❌                               | Yes ✅ (Via Blobs)            | No ❌                                     |
| **JSON Mode**                | Yes ✅                        | Yes ✅ (Via Tool)                   | Yes ✅                        | Yes ✅ (Model Dependent)                  |
| **API Key Required**         | Yes ✅                        | Yes ✅                              | Yes ✅                        | Optional                                  |
| **Backend Type**             | Cloud ☁️                      | Cloud ☁️                            | Cloud ☁️                      | Local/Self-Hosted 🏠                      |
| **Configuration**            | `OPENAI_API_KEY`              | `ANTHROPIC_API_KEY`                 | `GEMINI_API_KEY`              | `OLLAMA_API_URL`               |

**Notes:**

- **Ollama Tool Use:** Native tool use depends on the specific Ollama model. A textual fallback mechanism (prompting the model to generate specific text for tool calls) is available.
- **Ollama JSON Mode:** Support depends on the specific Ollama model's ability to follow formatting instructions.
- **Anthropic JSON Mode:** Implemented by instructing the model to use a predefined "json_output" tool.
- **Gemini File Input:** Supports generic file input via `Blob` data, not just images embedded directly in message content like other providers. Image content within messages is not implemented.
- **Multimodal:** Refers to including images directly alongside text within a single message turn.

This provider system allows Nodetool to flexibly leverage the capabilities of different LLMs for chat-based interactions and agentic workflows.
