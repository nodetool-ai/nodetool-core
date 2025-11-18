[← Back to Providers](../providers.md)

# OpenAI

**Audience:** Users calling OpenAI models through NodeTool.  
**What you will learn:** Supported capabilities, authentication, and example usage.

## Capabilities

- Chat/completions with tool calling and JSON mode
- Vision inputs
- Image generation (DALL-E)
- Text-to-speech and speech-to-text
- Streaming responses

## Authentication

Set `OPENAI_API_KEY` in your environment or secrets store.

```bash
export OPENAI_API_KEY=sk-...
```

## Example (OpenAI-compatible API)

```bash
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -X POST http://127.0.0.1:8000/v1/chat/completions \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello from NodeTool"}],
    "stream": true
  }'
```

## Notes

- Tool calling maps to OpenAI function calling; ensure tool schemas are JSON-serializable.
- When using the NodeTool proxy, keep `/v1` paths stable and prefer TLS.
