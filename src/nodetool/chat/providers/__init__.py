"""
Chat provider module for multi-provider chat functionality.

This module provides a collection of language model providers that can be used
to interact with various LLM APIs including OpenAI, Anthropic, and Ollama.
"""

# Base provider class
from nodetool.chat.providers.base import ChatProvider, Chunk

# OpenAI provider
from nodetool.chat.providers.openai import OpenAIProvider

# Anthropic provider
from nodetool.chat.providers.anthropic import AnthropicProvider

# Ollama provider
from nodetool.chat.providers.ollama import OllamaProvider

# Provider factory
from nodetool.metadata.types import Provider as ProviderEnum

# Provider instance cache
_provider_cache: dict[ProviderEnum, ChatProvider] = {}


def get_provider(provider_type: ProviderEnum) -> ChatProvider:
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
        provider = OpenAIProvider()
    elif provider_type == ProviderEnum.Anthropic:
        provider = AnthropicProvider()
    elif provider_type == ProviderEnum.Ollama:
        provider = OllamaProvider()
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
