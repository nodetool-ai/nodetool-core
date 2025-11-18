[← Back to Providers](../providers.md)

# ComfyUI

**Audience:** Users running ComfyUI workflows via NodeTool.  
**What you will learn:** Capabilities, deployment options, and caveats.

## Capabilities

- Image and video workflows executed through ComfyUI graphs
- Streaming progress where the workflow emits updates
- GPU acceleration when available

## Deployment

- **Local:** run ComfyUI alongside NodeTool and point the provider to the local URL.
- **RunPod:** use the ComfyUI-ready images and set `RUNPOD_API_KEY` and endpoint ID.
- **Docker:** build images with compatible CUDA/cuDNN; allocate GPUs per job with `--gpus` constraints.

## Configuration

Set the appropriate base URL and credentials for your deployment:

```bash
export COMFY_FOLDER=/path/to/comfy
export AIME_USER=optional_user
export AIME_API_KEY=optional_key
```

## Notes

- Keep images pinned to specific CUDA versions to avoid runtime mismatches.
- Limit resource usage with Docker CPU/memory/PIDs and avoid running as root in production.
- Integrate with generic nodes when possible so workflows stay provider-agnostic.
