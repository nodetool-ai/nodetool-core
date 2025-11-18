[← Back to Providers](../providers.md)

# Ollama

**Audience:** Users running local/open-source models through NodeTool.  
**What you will learn:** Capabilities, configuration, and streaming behavior.

## Capabilities

- Chat completions with optional tool use (model-dependent)
- Vision via base64 images (model-dependent)
- Streaming token output

## Configuration

- Default base URL: `http://127.0.0.1:11434`
- Override with `OLLAMA_API_URL`
- No API key required by default

```bash
export OLLAMA_API_URL=http://127.0.0.1:11434
ollama pull llama3.2
```

## Example

```bash
curl -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -X POST http://127.0.0.1:8000/v1/chat/completions \
  -d '{
    "model": "llama3.2",
    "provider": "ollama",
    "messages": [{"role": "user", "content": "Explain Ollama routing"}],
    "stream": true
  }'
```

## Notes

- Tool calling and JSON mode depend on the selected model; provide graceful fallbacks.
- Keep models pulled and warmed to avoid cold-start latency in production.
