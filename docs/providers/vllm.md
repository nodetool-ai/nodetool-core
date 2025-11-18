[← Back to Providers](../providers.md)

# vLLM

**Audience:** Users hosting OpenAI-compatible gateways via vLLM.  
**What you will learn:** Capabilities, auth, and configuration tips.

## Capabilities

- Chat/completions over OpenAI-compatible API
- Streaming token output
- Tool calling when enabled in model config

## Authentication

- Optional `VLLM_API_KEY` if your gateway enforces keys
- Point NodeTool to your gateway via `VLLM_BASE_URL` or override per request

```bash
export VLLM_BASE_URL=http://vllm-gateway:8000
export VLLM_API_KEY=optional-key
```

## Example

```bash
curl -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "X-NodeTool-Base-Url: http://vllm-gateway:8000" \
  -X POST http://127.0.0.1:8000/v1/chat/completions \
  -d '{
    "model": "llama3-70b-instruct",
    "provider": "vllm",
    "messages": [{"role": "user", "content": "How do I scale vLLM?"}],
    "stream": true
  }'
```

## Notes

- Keep model shards pinned and monitor throughput; vLLM excels at high concurrency.
- Align TLS and auth at the proxy in front of the vLLM gateway when exposed publicly.
