"""
Unit tests for ApifyProvider SERP provider.

Tests cover:
- ApifyProvider initialization
- Search methods with mocked API responses
- Error handling
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient, Response

from nodetool.agents.serp_providers.apify_provider import ApifyProvider


@pytest.fixture
def mock_api_key():
    """Mock API key for testing."""
    return "test_apify_api_key_12345"


@pytest.fixture
def apify_provider(mock_api_key, monkeypatch):
    """Create an ApifyProvider instance with mocked API key."""
    monkeypatch.setenv("APIFY_API_KEY", mock_api_key)
    return ApifyProvider(api_key=mock_api_key)


@pytest.fixture
def mock_search_response():
    """Mock successful search response from Apify."""
    return {
        "data": {
            "status": "SUCCEEDED",
            "defaultDatasetId": "test_dataset_id",
        }
    }


@pytest.fixture
def mock_dataset_response():
    """Mock dataset response with search results."""
    return [
        {
            "organicResults": [
                {
                    "title": "Test Result 1",
                    "url": "https://example.com/1",
                    "description": "This is test result 1",
                    "rank": 1,
                },
                {
                    "title": "Test Result 2",
                    "url": "https://example.com/2",
                    "description": "This is test result 2",
                    "rank": 2,
                },
            ]
        }
    ]


@pytest.fixture
def mock_news_response():
    """Mock dataset response with news results."""
    return [
        {
            "title": "Breaking News 1",
            "url": "https://news.example.com/1",
            "source": "Test News",
            "date": "2024-01-15",
            "description": "This is breaking news 1",
        },
        {
            "title": "Breaking News 2",
            "url": "https://news.example.com/2",
            "source": "Test News",
            "date": "2024-01-14",
            "description": "This is breaking news 2",
        },
    ]


class TestApifyProviderInitialization:
    """Tests for ApifyProvider initialization."""

    def test_init_with_api_key(self, mock_api_key):
        """Test initialization with API key."""
        provider = ApifyProvider(api_key=mock_api_key)
        assert provider.api_key == mock_api_key
        assert provider.country_code == "us"
        assert provider.language_code == "en"

    def test_init_with_custom_settings(self, mock_api_key):
        """Test initialization with custom country and language codes."""
        provider = ApifyProvider(
            api_key=mock_api_key, country_code="gb", language_code="en-GB"
        )
        assert provider.country_code == "gb"
        assert provider.language_code == "en-GB"

    def test_init_without_api_key(self, monkeypatch):
        """Test initialization fails without API key."""
        monkeypatch.delenv("APIFY_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Apify API key"):
            ApifyProvider(api_key=None)


@pytest.mark.asyncio
class TestApifyProviderSearch:
    """Tests for ApifyProvider search methods."""

    async def test_search_organic(
        self, apify_provider, mock_search_response, mock_dataset_response
    ):
        """Test organic search with mocked responses."""
        with patch.object(
            apify_provider, "_get_client", return_value=AsyncMock()
        ) as mock_get_client:
            mock_client = mock_get_client.return_value

            # Mock the run response
            mock_run_response = MagicMock(spec=Response)
            mock_run_response.json.return_value = mock_search_response
            mock_run_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_run_response)

            # Mock the dataset response
            mock_data_response = MagicMock(spec=Response)
            mock_data_response.json.return_value = mock_dataset_response
            mock_data_response.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_data_response)

            # Execute search
            result = await apify_provider.search("test query", num_results=10)

            # Verify results
            assert "organic_results" in result
            assert len(result["organic_results"]) > 0
            assert result["organic_results"][0]["title"] == "Test Result 1"

    async def test_search_news(
        self, apify_provider, mock_search_response, mock_news_response
    ):
        """Test news search with mocked responses."""
        with patch.object(
            apify_provider, "_get_client", return_value=AsyncMock()
        ) as mock_get_client:
            mock_client = mock_get_client.return_value

            # Mock the run response
            mock_run_response = MagicMock(spec=Response)
            mock_run_response.json.return_value = mock_search_response
            mock_run_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_run_response)

            # Mock the dataset response
            mock_data_response = MagicMock(spec=Response)
            mock_data_response.json.return_value = mock_news_response
            mock_data_response.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_data_response)

            # Execute search
            result = await apify_provider.search_news("test news", num_results=10)

            # Verify results
            assert isinstance(result, list)
            assert len(result) > 0
            assert result[0]["title"] == "Breaking News 1"
            assert result[0]["type"] == "news"

    async def test_search_images_without_keyword(self, apify_provider):
        """Test image search fails without keyword or image_url."""
        result = await apify_provider.search_images(keyword=None, image_url=None)
        assert "error" in result
        assert "keyword" in result["error"].lower() or "image_url" in result["error"].lower()

    async def test_search_images_reverse_not_supported(self, apify_provider):
        """Test reverse image search is not supported."""
        result = await apify_provider.search_images(
            keyword=None, image_url="https://example.com/image.jpg"
        )
        assert "error" in result
        assert "not" in result["error"].lower() and "supported" in result["error"].lower()

    async def test_search_finance_not_supported(self, apify_provider):
        """Test finance search is not supported."""
        result = await apify_provider.search_finance("AAPL")
        assert "error" in result
        assert "not supported" in result["error"]

    async def test_search_jobs_not_supported(self, apify_provider):
        """Test jobs search is not supported."""
        result = await apify_provider.search_jobs("software engineer")
        assert "error" in result
        assert "not supported" in result["error"]

    async def test_search_lens_not_supported(self, apify_provider):
        """Test lens search is not supported."""
        result = await apify_provider.search_lens("https://example.com/image.jpg")
        assert "error" in result
        assert "not supported" in result["error"]

    async def test_error_handling(self, apify_provider):
        """Test error handling when actor run fails."""
        with patch.object(
            apify_provider, "_get_client", return_value=AsyncMock()
        ) as mock_get_client:
            mock_client = mock_get_client.return_value

            # Mock failed run response
            failed_response = {
                "data": {
                    "status": "FAILED",
                }
            }
            mock_run_response = MagicMock(spec=Response)
            mock_run_response.json.return_value = failed_response
            mock_run_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_run_response)

            # Execute search
            result = await apify_provider.search("test query")

            # Verify error
            assert "error" in result
            assert "FAILED" in result["error"]


@pytest.mark.asyncio
class TestApifyProviderCleanup:
    """Tests for ApifyProvider cleanup."""

    async def test_close(self, apify_provider):
        """Test close method."""
        mock_client = AsyncMock(spec=AsyncClient)
        apify_provider._client = mock_client

        await apify_provider.close()

        # Verify client was closed
        mock_client.aclose.assert_called_once()

    async def test_context_manager(self, mock_api_key):
        """Test provider as context manager."""
        async with ApifyProvider(api_key=mock_api_key) as provider:
            assert provider.api_key == mock_api_key

        # Cleanup should have been called
        # Note: We can't easily verify this without mocking, but the context manager should work
