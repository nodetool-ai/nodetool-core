[← Back to Providers](../providers.md)

# HuggingFace

**Audience:** Users invoking HuggingFace-backed models through NodeTool.  
**What you will learn:** Capabilities, auth, and sub-provider notes.

## Capabilities

- Text-to-image and image-to-image via FAL, Replicate, and other hosted backends
- Text and chat models where supported
- Streaming availability depends on upstream provider

## Authentication

Set `HF_TOKEN` and any sub-provider keys (e.g., `FAL_API_KEY`, `REPLICATE_API_TOKEN`).

```bash
export HF_TOKEN=hf_...
export FAL_API_KEY=fal_...
```

## Example

```bash
curl -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -X POST http://127.0.0.1:8000/api/workflows/<workflow_id>/run?stream=true \
  -d '{"params": {"model": "fal-ai/flux"}}'
```

## Notes

- Capabilities vary widely by model; prefer generic nodes and expose only parameters supported by the chosen backend.
- Keep tokens scoped per provider to limit blast radius.
