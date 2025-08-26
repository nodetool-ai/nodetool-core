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

    if provider_type in _provider_cache:
        return _provider_cache[provider_type]

    provider: ChatProvider
    # Lazy-import providers to avoid importing optional dependencies at module import time
    if provider_type == ProviderEnum.OpenAI:
        from nodetool.chat.providers.openai_provider import OpenAIProvider
        provider = OpenAIProvider(**kwargs)
    elif provider_type == ProviderEnum.Gemini:
        from nodetool.chat.providers.gemini_provider import GeminiProvider
        provider = GeminiProvider(**kwargs)
    elif provider_type == ProviderEnum.Anthropic:
        from nodetool.chat.providers.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider(**kwargs)
    elif provider_type == ProviderEnum.Ollama:
        from nodetool.chat.providers.ollama_provider import OllamaProvider
        provider = OllamaProvider(**kwargs)
    elif provider_type == ProviderEnum.HuggingFace:
        from nodetool.chat.providers.huggingface_provider import HuggingFaceProvider
        provider = HuggingFaceProvider(**kwargs)
    elif provider_type == ProviderEnum.HuggingFaceGroq:
        from nodetool.chat.providers.huggingface_provider import HuggingFaceProvider
        provider = HuggingFaceProvider("groq", **kwargs)
    elif provider_type == ProviderEnum.HuggingFaceCerebras:
        from nodetool.chat.providers.huggingface_provider import HuggingFaceProvider
        provider = HuggingFaceProvider("cerebras", **kwargs)
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
