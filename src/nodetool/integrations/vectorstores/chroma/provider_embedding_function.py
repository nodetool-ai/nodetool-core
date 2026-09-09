"""
Provider-based embedding function for ChromaDB.

This module provides embedding functions that use the provider interface to generate embeddings,
allowing collections to use OpenAI, Ollama, or other provider APIs for embedding generation.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Coroutine

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import Provider

log = get_logger(__name__)

# Default fallback model for SentenceTransformer
DEFAULT_SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"


def _run_coroutine_sync(make_coro: Callable[[], Coroutine[Any, Any, Any]]) -> Any:
    """Run a coroutine to completion from synchronous code.

    Chroma's `EmbeddingFunction` interface is synchronous, so it may be invoked
    from inside a running event loop. In that case we first try `nest_asyncio`
    (cheap, and preserves the current context such as ResourceScope). If it is
    unavailable or refuses to re-enter the loop, we run the coroutine on a
    dedicated worker thread with its own event loop and block on the result.

    Args:
        make_coro: Factory returning a fresh coroutine. A factory is required
            because a coroutine object cannot be awaited more than once.

    Returns:
        The coroutine's result.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop, safe to use asyncio.run()
        return asyncio.run(make_coro())

    # We're in an async context - try nest_asyncio to preserve ResourceScope
    coro = make_coro()
    try:
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    except (ImportError, RuntimeError) as e:
        coro.close()
        log.warning(
            f"Running in async context but the running loop cannot be re-entered ({e}). "
            "Embeddings will be generated on a separate thread without ResourceScope. "
            "Install nest_asyncio for proper context propagation."
        )

    # Run on a separate thread with its own event loop and block on the result.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="chroma-embed") as executor:
        return executor.submit(lambda: asyncio.run(make_coro())).result()


class ProviderEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    ChromaDB embedding function that uses the provider interface.

    This function wraps the provider's generate_embedding method to generate
    embeddings for documents when adding or querying collections.

    Args:
        provider: The provider enum value (e.g., Provider.OpenAI, Provider.Ollama)
        model: The embedding model ID (e.g., "text-embedding-3-small", "nomic-embed-text")
        **kwargs: Additional arguments passed to the provider's generate_embedding method
    """

    def __init__(
        self,
        provider: Provider | str,
        model: str,
        **kwargs,
    ):
        self._provider_enum = Provider(provider) if isinstance(provider, str) else provider
        self._model = model
        self._kwargs = kwargs
        self._provider_instance = None

    def _get_provider(self):
        """Lazily initialize the provider instance."""
        if self._provider_instance is None:
            from nodetool.providers.base import get_registered_provider

            provider_cls, provider_kwargs = get_registered_provider(self._provider_enum)

            # Build secrets from environment
            secrets = {}
            for secret_name in provider_cls.required_secrets():
                value = Environment.get(secret_name)
                if value:
                    secrets[secret_name] = value

            self._provider_instance = provider_cls(secrets=secrets, **provider_kwargs)

        return self._provider_instance

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for the given documents.

        Args:
            input: List of document strings to embed

        Returns:
            List of embedding vectors
        """
        if not input:
            return []

        provider = self._get_provider()

        # Run the async generate_embedding in a sync context
        embeddings = _run_coroutine_sync(
            lambda: provider.generate_embedding(
                text=list(input),
                model=self._model,
                **self._kwargs,
            )
        )

        return list(embeddings)  # type: ignore[return-value]


class OpenAIEmbeddingFunction(ProviderEmbeddingFunction):
    """
    Convenience class for OpenAI embeddings.

    Args:
        model: The OpenAI embedding model ID (default: "text-embedding-3-small")
        dimensions: Optional output dimensions for text-embedding-3-* models
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
    ):
        kwargs = {}
        if dimensions:
            kwargs["dimensions"] = dimensions
        super().__init__(provider=Provider.OpenAI, model=model, **kwargs)


class OllamaProviderEmbeddingFunction(ProviderEmbeddingFunction):
    """
    Convenience class for Ollama embeddings using the provider interface.

    Args:
        model: The Ollama embedding model ID (default: "nomic-embed-text")
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
    ):
        super().__init__(provider=Provider.Ollama, model=model)


def get_provider_embedding_function(
    embedding_model: str,
    provider: Provider | str | None = None,
) -> EmbeddingFunction[Documents]:
    """
    Get an embedding function for the given model and optional provider.

    This function determines the appropriate provider based on the model name
    if no provider is explicitly specified.

    Args:
        embedding_model: The embedding model ID
        provider: Optional provider to use. If None, determined from model name.

    Returns:
        An embedding function compatible with ChromaDB
    """
    # If provider is specified, use it directly
    if provider is not None:
        provider_enum = Provider(provider) if isinstance(provider, str) else provider
        return ProviderEmbeddingFunction(provider=provider_enum, model=embedding_model)

    # Try to determine provider from model name
    # OpenAI embedding models typically start with "text-embedding-"
    if embedding_model.startswith("text-embedding-"):
        return OpenAIEmbeddingFunction(model=embedding_model)

    # For most other cases, try Ollama (common local embedding models)
    # This includes: nomic-embed-text, all-minilm, mxbai-embed-large, etc.
    ollama_url = Environment.get("OLLAMA_API_URL")
    if ollama_url:
        return OllamaProviderEmbeddingFunction(model=embedding_model)

    # Fallback to SentenceTransformer for local embeddings
    from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
        SentenceTransformerEmbeddingFunction,
    )

    log.warning(
        f"Could not determine provider for embedding model '{embedding_model}'. Falling back to SentenceTransformer."
    )
    return SentenceTransformerEmbeddingFunction(model_name=DEFAULT_SENTENCE_TRANSFORMER_MODEL)
