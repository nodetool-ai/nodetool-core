[← Back to Providers](../providers.md)

# Gemini

**Audience:** Users running Google Gemini through NodeTool.  
**What you will learn:** Capabilities, auth variables, and example requests.

## Capabilities

- Chat with tools and system prompts
- Vision inputs and file/blob uploads
- Video generation (Veo) where supported
- Streaming responses

## Authentication

Set `GEMINI_API_KEY`.

```bash
export GEMINI_API_KEY=your-key
```

## Example

```bash
curl -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -X POST http://127.0.0.1:8000/v1/chat/completions \
  -d '{
    "model": "gemini-1.5-pro",
    "provider": "gemini",
    "messages": [{"role": "user", "content": "List three use cases"}],
    "stream": false
  }'
```

## Notes

- Use blobs for file inputs; see DSL examples for attaching assets.
- Some parameters (e.g., safety settings) are provider-specific; keep workflows declarative with generic nodes where possible.
