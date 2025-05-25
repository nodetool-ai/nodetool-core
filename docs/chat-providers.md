# Chat Providers

Chat providers translate between NodeTool messages and specific LLM APIs.
Current implementations include OpenAI, Anthropic, Gemini and Ollama.

Each provider supports streaming responses and optional tool calls.
The table below summarises key features:

| Feature | OpenAI | Anthropic | Gemini | Ollama |
| ------- | ------ | --------- | ------ | ------ |
| Streaming | ✅ | ✅ | ✅ | ✅ |
| Tool Calls | ✅ | ✅ | ✅ | ✅ (model dependent) |
| JSON Mode | ✅ | ✅ | ✅ | ✅ |
| API Key Required | ✅ | ✅ | ✅ | Optional |

See [providers README](../src/nodetool/chat/providers/README.md) for more information.
