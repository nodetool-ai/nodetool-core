"""
Chat provider module for multi-provider chat functionality.

This module provides a collection of language model providers that can be used
to interact with various LLM APIs including OpenAI, Anthropic, and Ollama.
"""

# Base provider class
from nodetool.chat.providers.base import ChatProvider

# Provider factory
from nodetool.chat.providers.gemini_provider import GeminiProvider
from nodetool.chat.providers.anthropic_provider import AnthropicProvider
from nodetool.chat.providers.ollama_provider import OllamaProvider
from nodetool.chat.providers.openai_provider import OpenAIProvider
from nodetool.chat.providers.huggingface_provider import HuggingFaceProvider
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
    if provider_type == ProviderEnum.OpenAI:
        provider = OpenAIProvider(**kwargs)
    elif provider_type == ProviderEnum.Gemini:
        provider = GeminiProvider(**kwargs)
    elif provider_type == ProviderEnum.Anthropic:
        provider = AnthropicProvider(**kwargs)
    elif provider_type == ProviderEnum.Ollama:
        provider = OllamaProvider(**kwargs)
    elif provider_type == ProviderEnum.HuggingFace:
        provider = HuggingFaceProvider(**kwargs)
    else:
        raise ValueError(f"Provider {provider_type} not supported")

    _provider_cache[provider_type] = provider
    return provider


# Export all providers
__all__ = [
    # Base class
    "ChatProvider",
    "Chunk",
    # Provider implementations
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    # Factory function
    "get_provider",
]
