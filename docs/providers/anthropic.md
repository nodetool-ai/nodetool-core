[← Back to Providers](../providers.md)

# Anthropic

**Audience:** Users running Claude models through NodeTool.  
**What you will learn:** Capabilities, auth, and streaming behavior.

## Capabilities

- Chat with tool use and system prompts
- Vision inputs
- Streaming token output

## Authentication

Set `ANTHROPIC_API_KEY`.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Example

```bash
curl -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -X POST http://127.0.0.1:8000/v1/chat/completions \
  -d '{
    "model": "claude-3-5-sonnet-latest",
    "provider": "anthropic",
    "messages": [{"role": "user", "content": "Summarize this"}],
    "stream": true
  }'
```

## Notes

- JSON mode is available via tool calling; ensure tool outputs remain valid JSON.
- Respect message safety filters; handle `error` events on streaming connections.
