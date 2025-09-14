"""
Chat provider module for multi-provider chat functionality.

This module provides a collection of language model providers that can be used
to interact with various LLM APIs including OpenAI, Anthropic, and Ollama.
"""

# Base provider class and testing utilities
from nodetool.chat.providers.base import ChatProvider, MockProvider
from nodetool.chat.providers.fake_provider import (
    FakeProvider,
    create_fake_tool_call,
    create_simple_fake_provider,
    create_streaming_fake_provider,
    create_tool_calling_fake_provider,
)
from nodetool.metadata.types import Provider as ProviderEnum
from nodetool.workflows.types import Chunk

# Provider instance cache
_provider_cache: dict[ProviderEnum, ChatProvider] = {}


def get_provider(provider_type: ProviderEnum, **kwargs) -> ChatProvider:
    """
    Get a chat provider instance based on the provider type.
    Providers are cached after first creation.

    Args:
        provider_type: The provider type enum

    Returns:
        A chat provider instance

    Raises:
        ValueError: If the provider type is not supported
    """
    from nodetool.chat.providers.huggingface_provider import HuggingFaceProvider
    from nodetool.chat.providers.openai_provider import OpenAIProvider
    from nodetool.chat.providers.gemini_provider import GeminiProvider
    from nodetool.chat.providers.anthropic_provider import AnthropicProvider
    from nodetool.chat.providers.ollama_provider import OllamaProvider
    from nodetool.chat.providers.llama_provider import LlamaProvider

    if provider_type in _provider_cache:
        return _provider_cache[provider_type]

    provider: ChatProvider
    # Lazy-import providers to avoid importing optional dependencies at module import time
    if provider_type == ProviderEnum.OpenAI:
        provider = OpenAIProvider(**kwargs)
    elif provider_type == ProviderEnum.Gemini:
        provider = GeminiProvider(**kwargs)
    elif provider_type == ProviderEnum.Anthropic:
        provider = AnthropicProvider(**kwargs)
    elif provider_type == ProviderEnum.Ollama:
        provider = OllamaProvider(**kwargs)
    elif provider_type == ProviderEnum.LlamaCpp:
        provider = LlamaProvider(**kwargs)
    elif provider_type == ProviderEnum.HuggingFace:
        provider = HuggingFaceProvider(**kwargs)
    elif provider_type == ProviderEnum.HuggingFaceGroq:
        provider = HuggingFaceProvider("groq", **kwargs)
    elif provider_type == ProviderEnum.HuggingFaceCerebras:
        provider = HuggingFaceProvider("cerebras", **kwargs)
    elif provider_type == ProviderEnum.HuggingFaceCohere:
        provider = HuggingFaceProvider("cohere", **kwargs)
    elif provider_type == ProviderEnum.HuggingFaceFalAI:
        provider = HuggingFaceProvider("fal-ai", **kwargs)
    elif provider_type == ProviderEnum.HuggingFaceFeatherlessAI:
        provider = HuggingFaceProvider("featherless-ai", **kwargs)
    elif provider_type == ProviderEnum.HuggingFaceFireworksAI:
        provider = HuggingFaceProvider("fireworks-ai", **kwargs)
    elif provider_type == ProviderEnum.HuggingFaceBlackForestLabs:
        provider = HuggingFaceProvider("black-forest-labs", **kwargs)
    elif provider_type == ProviderEnum.HuggingFaceHFInference:
        provider = HuggingFaceProvider("hf-inference", **kwargs)
    elif provider_type == ProviderEnum.HuggingFaceHyperbolic:
        provider = HuggingFaceProvider("hyperbolic", **kwargs)
    elif provider_type == ProviderEnum.HuggingFaceNebius:
        provider = HuggingFaceProvider("nebius", **kwargs)
    elif provider_type == ProviderEnum.HuggingFaceNovita:
        provider = HuggingFaceProvider("novita", **kwargs)
    elif provider_type == ProviderEnum.HuggingFaceNscale:
        provider = HuggingFaceProvider("nscale", **kwargs)
    elif provider_type == ProviderEnum.HuggingFaceOpenAI:
        provider = HuggingFaceProvider("openai", **kwargs)
    elif provider_type == ProviderEnum.HuggingFaceReplicate:
        provider = HuggingFaceProvider("replicate", **kwargs)
    elif provider_type == ProviderEnum.HuggingFaceSambanova:
        provider = HuggingFaceProvider("sambanova", **kwargs)
    elif provider_type == ProviderEnum.HuggingFaceTogether:
        provider = HuggingFaceProvider("together", **kwargs)
    else:
        raise ValueError(f"Provider {provider_type} not supported")

    _provider_cache[provider_type] = provider
    return provider


__all__ = [
    "ChatProvider",
    "MockProvider",
    "FakeProvider",
    "create_fake_tool_call",
    "create_simple_fake_provider",
    "create_streaming_fake_provider",
    "create_tool_calling_fake_provider",
    "Chunk",
    "get_provider",
]
