"""
Unit tests for BraveSearchProvider.

Tests cover:
- Initialization and configuration
- Web search
- News search
- Image search
- Unsupported operations (finance, jobs, lens, maps, shopping)
- Error handling
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient, HTTPStatusError, Request, Response

# Import tools first to avoid circular import issues
# (the tools __init__ imports serp_tools which imports the providers)
import nodetool.agents.tools.serp_tools
from nodetool.agents.serp_providers.brave_search_provider import BraveSearchProvider


@pytest.fixture
def mock_api_key():
    """Provide a mock API key for testing."""
    return "test-brave-api-key"


@pytest.fixture
def brave_provider(mock_api_key):
    """Create a BraveSearchProvider instance for testing."""
    return BraveSearchProvider(api_key=mock_api_key)


class TestBraveSearchProviderInit:
    """Tests for BraveSearchProvider initialization."""

    def test_init_with_api_key(self, mock_api_key):
        """Test initialization with an API key."""
        provider = BraveSearchProvider(api_key=mock_api_key)
        assert provider.api_key == mock_api_key
        assert provider.country == "us"
        assert provider.language == "en"

    def test_init_with_custom_params(self, mock_api_key):
        """Test initialization with custom country and language."""
        provider = BraveSearchProvider(
            api_key=mock_api_key, country="gb", language="fr"
        )
        assert provider.country == "gb"
        assert provider.language == "fr"

    def test_init_without_api_key_raises(self, monkeypatch):
        """Test that initialization without an API key raises ValueError."""
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        with pytest.raises(ValueError, match=r"Brave API key.*not found"):
            BraveSearchProvider(api_key=None)


class TestBraveSearchProviderHeaders:
    """Tests for BraveSearchProvider header generation."""

    def test_get_headers(self, brave_provider, mock_api_key):
        """Test that headers are correctly generated."""
        headers = brave_provider._get_headers()
        assert headers["Accept"] == "application/json"
        assert headers["Accept-Encoding"] == "gzip"
        assert headers["X-Subscription-Token"] == mock_api_key


class TestBraveSearchProviderSearch:
    """Tests for BraveSearchProvider web search."""

    @pytest.mark.asyncio
    async def test_search_success(self, brave_provider):
        """Test successful web search."""
        mock_response = {
            "type": "web",
            "query": {"original": "test query"},
            "web": {
                "results": [
                    {
                        "title": "Test Result",
                        "url": "https://example.com",
                        "description": "A test result",
                    }
                ]
            },
        }

        with patch.object(
            brave_provider, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            result = await brave_provider.search("test query", num_results=5)

            assert result == mock_response
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == brave_provider.WEB_SEARCH_URL
            assert call_args[0][1]["q"] == "test query"
            assert call_args[0][1]["count"] == 5

    @pytest.mark.asyncio
    async def test_search_caps_results_at_20(self, brave_provider):
        """Test that num_results is capped at 20 for web search."""
        with patch.object(
            brave_provider, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = {}

            await brave_provider.search("test", num_results=50)

            call_args = mock_request.call_args
            assert call_args[0][1]["count"] == 20

    @pytest.mark.asyncio
    async def test_search_error_response(self, brave_provider):
        """Test handling of API error response."""
        with patch.object(
            brave_provider, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = {"error": "API rate limit exceeded"}

            result = await brave_provider.search("test")

            assert "error" in result


class TestBraveSearchProviderNews:
    """Tests for BraveSearchProvider news search."""

    @pytest.mark.asyncio
    async def test_search_news_success(self, brave_provider):
        """Test successful news search."""
        mock_response = {
            "type": "news",
            "results": [
                {
                    "title": "Breaking News",
                    "url": "https://news.example.com",
                    "description": "Test news article",
                    "age": "2 hours ago",
                }
            ],
        }

        with patch.object(
            brave_provider, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            result = await brave_provider.search_news("breaking news", num_results=5)

            assert result == mock_response
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == brave_provider.NEWS_SEARCH_URL


class TestBraveSearchProviderImages:
    """Tests for BraveSearchProvider image search."""

    @pytest.mark.asyncio
    async def test_search_images_success(self, brave_provider):
        """Test successful image search."""
        mock_response = {
            "type": "images",
            "results": [
                {
                    "title": "Test Image",
                    "url": "https://example.com/image.jpg",
                    "thumbnail": {"src": "https://example.com/thumb.jpg"},
                }
            ],
        }

        with patch.object(
            brave_provider, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            result = await brave_provider.search_images(keyword="cats", num_results=10)

            assert result == mock_response
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == brave_provider.IMAGES_SEARCH_URL

    @pytest.mark.asyncio
    async def test_search_images_caps_at_150(self, brave_provider):
        """Test that num_results is capped at 150 for image search."""
        with patch.object(
            brave_provider, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = {}

            await brave_provider.search_images(keyword="test", num_results=200)

            call_args = mock_request.call_args
            assert call_args[0][1]["count"] == 150

    @pytest.mark.asyncio
    async def test_search_images_no_keyword_or_url(self, brave_provider):
        """Test that image search requires keyword or image_url."""
        result = await brave_provider.search_images()
        assert "error" in result

    @pytest.mark.asyncio
    async def test_search_images_reverse_not_supported(self, brave_provider):
        """Test that reverse image search is not supported."""
        result = await brave_provider.search_images(
            image_url="https://example.com/image.jpg"
        )
        assert "error" in result
        assert "reverse image search" in result["error"].lower()


class TestBraveSearchProviderUnsupportedOperations:
    """Tests for unsupported operations."""

    @pytest.mark.asyncio
    async def test_search_finance_not_supported(self, brave_provider):
        """Test that finance search returns not supported error."""
        result = await brave_provider.search_finance("AAPL")
        assert "error" in result
        assert "not supported" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_search_jobs_not_supported(self, brave_provider):
        """Test that jobs search returns not supported error."""
        result = await brave_provider.search_jobs("software engineer")
        assert "error" in result
        assert "not supported" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_search_lens_not_supported(self, brave_provider):
        """Test that lens search returns not supported error."""
        result = await brave_provider.search_lens("https://example.com/image.jpg")
        assert "error" in result
        assert "not supported" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_search_maps_not_supported(self, brave_provider):
        """Test that maps search returns not supported error."""
        result = await brave_provider.search_maps("restaurants near me")
        assert "error" in result
        assert "not supported" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_search_shopping_not_supported(self, brave_provider):
        """Test that shopping search returns not supported error."""
        result = await brave_provider.search_shopping("laptop")
        assert "error" in result
        assert "not supported" in result["error"].lower()


class TestBraveSearchProviderMakeRequest:
    """Tests for BraveSearchProvider HTTP request handling."""

    @pytest.mark.asyncio
    async def test_make_request_success(self, brave_provider):
        """Test successful HTTP request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"web": {"results": []}}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=AsyncClient)
        mock_client.get.return_value = mock_response

        with patch.object(brave_provider, "_get_client", return_value=mock_client):
            result = await brave_provider._make_request(
                brave_provider.WEB_SEARCH_URL, {"q": "test"}
            )

            assert result == {"web": {"results": []}}

    @pytest.mark.asyncio
    async def test_make_request_http_error(self, brave_provider):
        """Test HTTP error handling."""
        mock_response = MagicMock(spec=Response)
        mock_response.status_code = 429
        mock_response.reason_phrase = "Too Many Requests"
        mock_response.text = "Rate limit exceeded"
        mock_response.raise_for_status.side_effect = HTTPStatusError(
            "Rate limit exceeded",
            request=MagicMock(spec=Request),
            response=mock_response,
        )

        mock_client = AsyncMock(spec=AsyncClient)
        mock_client.get.return_value = mock_response

        with patch.object(brave_provider, "_get_client", return_value=mock_client):
            result = await brave_provider._make_request(
                brave_provider.WEB_SEARCH_URL, {"q": "test"}
            )

            assert "error" in result
            assert "429" in result["error"]

    @pytest.mark.asyncio
    async def test_make_request_no_api_key(self, mock_api_key):
        """Test request fails when API key is missing."""
        provider = BraveSearchProvider(api_key=mock_api_key)
        provider.api_key = None  # Remove API key after construction

        result = await provider._make_request(provider.WEB_SEARCH_URL, {"q": "test"})

        assert "error" in result
        assert "API key" in result["error"]


class TestBraveSearchProviderContextManager:
    """Tests for async context manager functionality."""

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_api_key):
        """Test that the provider works as an async context manager."""
        provider = BraveSearchProvider(api_key=mock_api_key)

        async with provider as p:
            assert p is provider

    @pytest.mark.asyncio
    async def test_close(self, mock_api_key):
        """Test that close method works."""
        provider = BraveSearchProvider(api_key=mock_api_key)
        # Should not raise
        await provider.close()
