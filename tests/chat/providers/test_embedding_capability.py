"""
Tests for the embedding functionality in providers.

This module tests the generate_embedding and get_available_embedding_models
methods for providers that support embeddings (OpenAI, Ollama).
"""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nodetool.metadata.types import EmbeddingModel, Provider
from nodetool.providers.base import ProviderCapability


class TestOpenAIEmbeddings:
    """Test suite for OpenAI embedding functionality."""

    def create_provider(self):
        """Create an OpenAI provider with test credentials."""
        from nodetool.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(secrets={"OPENAI_API_KEY": "test-api-key"})

    @pytest.mark.asyncio
    async def test_get_available_embedding_models(self):
        """Test that OpenAI returns available embedding models."""
        provider = self.create_provider()
        models = await provider.get_available_embedding_models()

        assert len(models) == 3
        assert all(isinstance(m, EmbeddingModel) for m in models)
        assert all(m.provider == Provider.OpenAI for m in models)

        model_ids = [m.id for m in models]
        assert "text-embedding-3-small" in model_ids
        assert "text-embedding-3-large" in model_ids
        assert "text-embedding-ada-002" in model_ids

    @pytest.mark.asyncio
    async def test_generate_embedding_single_text(self):
        """Test generating embedding for a single text."""
        provider = self.create_provider()

        # Mock the OpenAI embeddings response
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        class MockData:
            def __init__(self, embedding):
                self.embedding = embedding

        class MockResponse:
            def __init__(self, embeddings):
                self.data = [MockData(emb) for emb in embeddings]

        async def mock_create(**kwargs):
            return MockResponse([mock_embedding])

        with patch.object(provider, "get_client") as mock_client:
            mock_client.return_value.embeddings.create = mock_create

            result = await provider.generate_embedding(
                text="Hello, world!",
                model="text-embedding-3-small",
            )

            assert len(result) == 1
            assert result[0] == mock_embedding

    @pytest.mark.asyncio
    async def test_generate_embedding_multiple_texts(self):
        """Test generating embeddings for multiple texts."""
        provider = self.create_provider()

        mock_embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]

        class MockData:
            def __init__(self, embedding):
                self.embedding = embedding

        class MockResponse:
            def __init__(self, embeddings):
                self.data = [MockData(emb) for emb in embeddings]

        async def mock_create(**kwargs):
            return MockResponse(mock_embeddings)

        with patch.object(provider, "get_client") as mock_client:
            mock_client.return_value.embeddings.create = mock_create

            result = await provider.generate_embedding(
                text=["Text 1", "Text 2", "Text 3"],
                model="text-embedding-3-small",
            )

            assert len(result) == 3
            assert result == mock_embeddings

    @pytest.mark.asyncio
    async def test_generate_embedding_with_dimensions(self):
        """Test that dimensions parameter is passed to API."""
        provider = self.create_provider()

        mock_embedding = [0.1, 0.2, 0.3]  # Reduced dimensions

        class MockData:
            def __init__(self, embedding):
                self.embedding = embedding

        class MockResponse:
            def __init__(self, embeddings):
                self.data = [MockData(emb) for emb in embeddings]

        async def mock_create(**kwargs):
            # Verify dimensions was passed
            assert kwargs.get("dimensions") == 256
            return MockResponse([mock_embedding])

        with patch.object(provider, "get_client") as mock_client:
            mock_client.return_value.embeddings.create = mock_create

            result = await provider.generate_embedding(
                text="Hello",
                model="text-embedding-3-small",
                dimensions=256,
            )

            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_generate_embedding_empty_text_raises_error(self):
        """Test that empty text raises ValueError."""
        provider = self.create_provider()

        with pytest.raises(ValueError, match="text must not be empty"):
            await provider.generate_embedding(text="", model="text-embedding-3-small")

    def test_embedding_capability_detected(self):
        """Test that the GENERATE_EMBEDDING capability is detected."""
        provider = self.create_provider()
        capabilities = provider.get_capabilities()

        assert ProviderCapability.GENERATE_EMBEDDING in capabilities


class TestOllamaEmbeddings:
    """Test suite for Ollama embedding functionality."""

    def create_provider(self):
        """Create an Ollama provider with test configuration."""
        from nodetool.providers.ollama_provider import OllamaProvider

        return OllamaProvider(secrets={"OLLAMA_API_URL": "http://localhost:11434"})

    @pytest.mark.asyncio
    async def test_generate_embedding_single_text(self):
        """Test generating embedding for a single text with Ollama."""
        provider = self.create_provider()

        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        class MockResponse:
            def __init__(self, embeddings):
                self.embeddings = embeddings

        async def mock_embed(**kwargs):
            return MockResponse([mock_embedding])

        with patch(
            "nodetool.providers.ollama_provider.get_ollama_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.embed = mock_embed
            mock_get_client.return_value.__aenter__.return_value = mock_client

            result = await provider.generate_embedding(
                text="Hello, world!",
                model="nomic-embed-text",
            )

            assert len(result) == 1
            assert result[0] == mock_embedding

    @pytest.mark.asyncio
    async def test_generate_embedding_multiple_texts(self):
        """Test generating embeddings for multiple texts with Ollama."""
        provider = self.create_provider()

        mock_embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]

        class MockResponse:
            def __init__(self, embeddings):
                self.embeddings = embeddings

        async def mock_embed(**kwargs):
            return MockResponse(mock_embeddings)

        with patch(
            "nodetool.providers.ollama_provider.get_ollama_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.embed = mock_embed
            mock_get_client.return_value.__aenter__.return_value = mock_client

            result = await provider.generate_embedding(
                text=["Text 1", "Text 2"],
                model="nomic-embed-text",
            )

            assert len(result) == 2
            assert result == mock_embeddings

    @pytest.mark.asyncio
    async def test_generate_embedding_empty_text_raises_error(self):
        """Test that empty text raises ValueError."""
        provider = self.create_provider()

        with pytest.raises(ValueError, match="text must not be empty"):
            await provider.generate_embedding(text="", model="nomic-embed-text")

    def test_embedding_capability_detected(self):
        """Test that the GENERATE_EMBEDDING capability is detected."""
        provider = self.create_provider()
        capabilities = provider.get_capabilities()

        assert ProviderCapability.GENERATE_EMBEDDING in capabilities


class TestGeminiEmbeddings:
    """Test suite for Gemini embedding functionality."""

    def create_provider(self):
        """Create a Gemini provider with test credentials."""
        from nodetool.providers.gemini_provider import GeminiProvider

        return GeminiProvider(secrets={"GEMINI_API_KEY": "test-api-key"})

    @pytest.mark.asyncio
    async def test_get_available_embedding_models(self):
        """Test that Gemini returns available embedding models."""
        provider = self.create_provider()
        models = await provider.get_available_embedding_models()

        assert len(models) == 3
        assert all(isinstance(m, EmbeddingModel) for m in models)
        assert all(m.provider == Provider.Gemini for m in models)

        model_ids = [m.id for m in models]
        assert "text-embedding-004" in model_ids
        assert "text-embedding-005" in model_ids
        assert "gemini-embedding-exp-03-07" in model_ids

    @pytest.mark.asyncio
    async def test_get_available_embedding_models_no_api_key(self):
        """Test that Gemini returns empty list when no API key."""
        from nodetool.providers.gemini_provider import GeminiProvider

        # We need to create a provider without API key - this should raise an assertion error
        # because the provider requires the key. Let's test the behavior with an empty key instead.
        provider = self.create_provider()
        provider.api_key = ""
        models = await provider.get_available_embedding_models()
        assert len(models) == 0

    @pytest.mark.asyncio
    async def test_generate_embedding_single_text(self):
        """Test generating embedding for a single text."""
        provider = self.create_provider()

        # Mock the Gemini embeddings response
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        class MockContentEmbedding:
            def __init__(self, values):
                self.values = values

        class MockResponse:
            def __init__(self, embeddings):
                self.embeddings = [MockContentEmbedding(emb) for emb in embeddings]

        async def mock_embed_content(**kwargs):
            return MockResponse([mock_embedding])

        with patch.object(provider, "get_client") as mock_client:
            mock_client.return_value.models.embed_content = mock_embed_content

            result = await provider.generate_embedding(
                text="Hello, world!",
                model="text-embedding-004",
            )

            assert len(result) == 1
            assert result[0] == mock_embedding

    @pytest.mark.asyncio
    async def test_generate_embedding_multiple_texts(self):
        """Test generating embeddings for multiple texts."""
        provider = self.create_provider()

        mock_embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]

        class MockContentEmbedding:
            def __init__(self, values):
                self.values = values

        class MockResponse:
            def __init__(self, embeddings):
                self.embeddings = [MockContentEmbedding(emb) for emb in embeddings]

        async def mock_embed_content(**kwargs):
            return MockResponse(mock_embeddings)

        with patch.object(provider, "get_client") as mock_client:
            mock_client.return_value.models.embed_content = mock_embed_content

            result = await provider.generate_embedding(
                text=["Text 1", "Text 2", "Text 3"],
                model="text-embedding-004",
            )

            assert len(result) == 3
            assert result == mock_embeddings

    @pytest.mark.asyncio
    async def test_generate_embedding_with_dimensions(self):
        """Test that dimensions parameter is passed to API."""
        provider = self.create_provider()

        mock_embedding = [0.1, 0.2, 0.3]  # Reduced dimensions

        class MockContentEmbedding:
            def __init__(self, values):
                self.values = values

        class MockResponse:
            def __init__(self, embeddings):
                self.embeddings = [MockContentEmbedding(emb) for emb in embeddings]

        async def mock_embed_content(**kwargs):
            # Verify dimensions was passed via config
            assert kwargs.get("config") is not None
            assert kwargs["config"]["output_dimensionality"] == 256
            return MockResponse([mock_embedding])

        with patch.object(provider, "get_client") as mock_client:
            mock_client.return_value.models.embed_content = mock_embed_content

            result = await provider.generate_embedding(
                text="Hello",
                model="text-embedding-004",
                dimensions=256,
            )

            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_generate_embedding_empty_text_raises_error(self):
        """Test that empty text raises ValueError."""
        provider = self.create_provider()

        with pytest.raises(ValueError, match="text must not be empty"):
            await provider.generate_embedding(text="", model="text-embedding-004")

    def test_embedding_capability_detected(self):
        """Test that the GENERATE_EMBEDDING capability is detected."""
        provider = self.create_provider()
        capabilities = provider.get_capabilities()

        assert ProviderCapability.GENERATE_EMBEDDING in capabilities


class TestMistralEmbeddings:
    """Test suite for Mistral embedding functionality."""

    def create_provider(self):
        """Create a Mistral provider with test credentials."""
        from nodetool.providers.mistral_provider import MistralProvider

        return MistralProvider(secrets={"MISTRAL_API_KEY": "test-api-key"})

    @pytest.mark.asyncio
    async def test_get_available_embedding_models(self):
        """Test that Mistral returns available embedding models."""
        provider = self.create_provider()
        models = await provider.get_available_embedding_models()

        assert len(models) == 1
        assert all(isinstance(m, EmbeddingModel) for m in models)
        assert all(m.provider == Provider.Mistral for m in models)

        model_ids = [m.id for m in models]
        assert "mistral-embed" in model_ids

    @pytest.mark.asyncio
    async def test_get_available_embedding_models_no_api_key(self):
        """Test that Mistral returns empty list when no API key."""
        provider = self.create_provider()
        provider.api_key = ""
        models = await provider.get_available_embedding_models()
        assert len(models) == 0

    @pytest.mark.asyncio
    async def test_generate_embedding_single_text(self):
        """Test generating embedding for a single text."""
        provider = self.create_provider()

        # Mock the OpenAI embeddings response (Mistral uses OpenAI-compatible API)
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        class MockData:
            def __init__(self, embedding):
                self.embedding = embedding

        class MockResponse:
            def __init__(self, embeddings):
                self.data = [MockData(emb) for emb in embeddings]

        async def mock_create(**kwargs):
            return MockResponse([mock_embedding])

        with patch.object(provider, "get_client") as mock_client:
            mock_client.return_value.embeddings.create = mock_create

            result = await provider.generate_embedding(
                text="Hello, world!",
                model="mistral-embed",
            )

            assert len(result) == 1
            assert result[0] == mock_embedding

    @pytest.mark.asyncio
    async def test_generate_embedding_empty_text_raises_error(self):
        """Test that empty text raises ValueError."""
        provider = self.create_provider()

        with pytest.raises(ValueError, match="text must not be empty"):
            await provider.generate_embedding(text="", model="mistral-embed")

    def test_embedding_capability_detected(self):
        """Test that the GENERATE_EMBEDDING capability is detected."""
        provider = self.create_provider()
        capabilities = provider.get_capabilities()

        assert ProviderCapability.GENERATE_EMBEDDING in capabilities


class TestProviderEmbeddingFunction:
    """Test suite for the ChromaDB provider embedding function wrapper."""

    def test_openai_embedding_function_creation(self):
        """Test creating an OpenAI embedding function."""
        from nodetool.integrations.vectorstores.chroma.provider_embedding_function import (
            OpenAIEmbeddingFunction,
        )

        func = OpenAIEmbeddingFunction(model="text-embedding-3-small")
        assert func._provider_enum == Provider.OpenAI
        assert func._model == "text-embedding-3-small"

    def test_ollama_embedding_function_creation(self):
        """Test creating an Ollama embedding function."""
        from nodetool.integrations.vectorstores.chroma.provider_embedding_function import (
            OllamaProviderEmbeddingFunction,
        )

        func = OllamaProviderEmbeddingFunction(model="nomic-embed-text")
        assert func._provider_enum == Provider.Ollama
        assert func._model == "nomic-embed-text"

    def test_get_provider_embedding_function_openai(self):
        """Test auto-detection of OpenAI embedding models."""
        from nodetool.integrations.vectorstores.chroma.provider_embedding_function import (
            OpenAIEmbeddingFunction,
            get_provider_embedding_function,
        )

        func = get_provider_embedding_function("text-embedding-3-small")
        assert isinstance(func, OpenAIEmbeddingFunction)

    def test_get_provider_embedding_function_with_explicit_provider(self):
        """Test explicit provider specification."""
        from nodetool.integrations.vectorstores.chroma.provider_embedding_function import (
            ProviderEmbeddingFunction,
            get_provider_embedding_function,
        )

        func = get_provider_embedding_function(
            embedding_model="custom-model",
            provider=Provider.OpenAI,
        )
        assert isinstance(func, ProviderEmbeddingFunction)
        assert func._provider_enum == Provider.OpenAI
