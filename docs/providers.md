[← Back to Docs Index](index.md)

# Providers

**Audience:** Workflow authors and integrators.  
**What you will learn:** Available providers, required auth variables, streaming support, and where to find per-provider details.

## Overview

NodeTool providers adapt external AI services into a common interface for chat, image/video generation, and audio. Choose providers via node `model` fields or CLI flags, and swap them without changing workflow logic.

## Comparison Table

| Provider  | Capabilities                                   | Streaming | Auth variables                   | Notes |
|-----------|------------------------------------------------|-----------|---------------------------------|-------|
| OpenAI    | Chat, tools, vision, images, TTS/ASR           | yes       | `OPENAI_API_KEY`                | OpenAI-compatible endpoints and models |
| Anthropic | Chat, tools, vision                            | yes       | `ANTHROPIC_API_KEY`             | Claude models; JSON mode via tool use |
| Gemini    | Chat, tools, vision, video                     | yes       | `GEMINI_API_KEY`                | File/blobs supported |
| HuggingFace | Text/image/video depending on sub-provider   | depends   | `HF_TOKEN`                      | Works with FAL, Replicate, Together |
| Ollama    | Chat, tools (model-dependent)                  | yes       | optional `OLLAMA_API_URL`       | Local/self-hosted models, no key by default |
| vLLM      | Chat (OpenAI-compatible)                       | yes       | optional `VLLM_API_KEY`         | Self-hosted; match base URL to gateway |
| ComfyUI   | Image/video workflows                          | stream per workflow | depends on deployment | Use local or RunPod deployments |

## Provider Guides

- [OpenAI](providers/openai.md)
- [Anthropic](providers/anthropic.md)
- [Gemini](providers/gemini.md)
- [HuggingFace](providers/huggingface.md)
- [Ollama](providers/ollama.md)
- [vLLM](providers/vllm.md)
- [ComfyUI](providers/comfyui.md)

## Generic Nodes

NodeTool ships provider-agnostic nodes that route requests based on the selected model. See the [DSL guide](dsl.md#generic-nodes) for usage and parameter mapping across providers.

## Development Notes

- Register new providers under `src/nodetool/providers` and add configuration to `.env.example`.
- Include streaming support where the upstream API allows; fall back gracefully when unsupported.
- Keep provider docs aligned with the [Security Hardening](security-hardening.md) guidance for key management.
