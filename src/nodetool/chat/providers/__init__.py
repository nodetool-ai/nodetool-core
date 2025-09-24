"""
Chat provider module for multi-provider chat functionality.

This module provides a collection of language model providers that can be used
to interact with various LLM APIs including OpenAI, Anthropic, and Ollama.
"""

# Base provider class and testing utilities
from nodetool.chat.providers.base import (
    ChatProvider,
    MockProvider,
    get_registered_provider,
)
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

    # import providers to ensure they are registered
    from nodetool.chat.providers import anthropic_provider
    from nodetool.chat.providers import gemini_provider
    from nodetool.chat.providers import huggingface_provider
    from nodetool.chat.providers import llama_provider
    from nodetool.chat.providers import ollama_provider
    from nodetool.chat.providers import openai_provider
    from nodetool.chat.providers import fake_provider
    from nodetool.chat.providers import gemini_provider
    from nodetool.chat.providers import huggingface_provider
    from nodetool.chat.providers import llama_provider
    from nodetool.chat.providers import ollama_provider
    from nodetool.chat.providers import openai_provider

    try:
        import nodetool.mlx.mlx_provider  # type: ignore
    except ImportError:
        pass

    provider_cls, kwargs = get_registered_provider(provider_type)
    if provider_cls is None:
        raise ValueError(
            f"Provider {provider_type.value} is not available. Install the corresponding package via nodetool's package manager."
        )

    provider: ChatProvider = provider_cls(**kwargs)

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
