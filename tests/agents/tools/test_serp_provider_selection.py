"""
Unit tests for SERP provider selection logic.

Tests cover:
- Provider selection based on SERP_PROVIDER setting
- Auto-selection when SERP_PROVIDER is not set
- Error handling for invalid settings
"""

from unittest.mock import AsyncMock, patch

import pytest

from nodetool.agents.serp_providers.apify_provider import ApifyProvider
from nodetool.agents.serp_providers.data_for_seo_provider import DataForSEOProvider
from nodetool.agents.serp_providers.serp_api_provider import SerpApiProvider
from nodetool.agents.tools.serp_tools import _get_configured_serp_provider
from nodetool.workflows.processing_context import ProcessingContext


@pytest.fixture
def mock_context():
    """Create a mock ProcessingContext."""
    context = AsyncMock(spec=ProcessingContext)
    # By default, return None for all secrets
    context.get_secret = AsyncMock(return_value=None)
    return context


@pytest.mark.asyncio
async def test_serp_provider_explicit_serpapi(mock_context, monkeypatch):
    """Test explicit selection of SerpApi provider."""
    # Set SERP_PROVIDER to serpapi
    monkeypatch.setenv("SERP_PROVIDER", "serpapi")

    # Mock the secret retrieval to return serpapi key
    async def get_secret_mock(key):
        if key == "SERPAPI_API_KEY":
            return "test_serpapi_key"
        return None

    mock_context.get_secret = get_secret_mock

    # Get provider
    provider, error = await _get_configured_serp_provider(mock_context)

    # Verify SerpApi provider is selected
    assert provider is not None
    assert isinstance(provider, SerpApiProvider)
    assert error is None


@pytest.mark.asyncio
async def test_serp_provider_explicit_apify(mock_context, monkeypatch):
    """Test explicit selection of Apify provider."""
    # Set SERP_PROVIDER to apify
    monkeypatch.setenv("SERP_PROVIDER", "apify")

    # Mock the secret retrieval to return apify key
    async def get_secret_mock(key):
        if key == "APIFY_API_KEY":
            return "test_apify_key"
        return None

    mock_context.get_secret = get_secret_mock

    # Get provider
    provider, error = await _get_configured_serp_provider(mock_context)

    # Verify Apify provider is selected
    assert provider is not None
    assert isinstance(provider, ApifyProvider)
    assert error is None


@pytest.mark.asyncio
async def test_serp_provider_explicit_dataforseo(mock_context, monkeypatch):
    """Test explicit selection of DataForSEO provider."""
    # Set SERP_PROVIDER to dataforseo
    monkeypatch.setenv("SERP_PROVIDER", "dataforseo")

    # Mock the secret retrieval to return dataforseo credentials
    async def get_secret_mock(key):
        if key == "DATA_FOR_SEO_LOGIN":
            return "test_login"
        elif key == "DATA_FOR_SEO_PASSWORD":
            return "test_password"
        return None

    mock_context.get_secret = get_secret_mock

    # Get provider
    provider, error = await _get_configured_serp_provider(mock_context)

    # Verify DataForSEO provider is selected
    assert provider is not None
    assert isinstance(provider, DataForSEOProvider)
    assert error is None


@pytest.mark.asyncio
async def test_serp_provider_explicit_but_missing_key(mock_context, monkeypatch):
    """Test error when SERP_PROVIDER is set but key is missing."""
    # Set SERP_PROVIDER to serpapi but don't provide the key
    monkeypatch.setenv("SERP_PROVIDER", "serpapi")

    # Mock the secret retrieval to return None
    mock_context.get_secret = AsyncMock(return_value=None)

    # Get provider
    provider, error = await _get_configured_serp_provider(mock_context)

    # Verify error is returned
    assert provider is None
    assert error is not None
    assert "SERPAPI_API_KEY is not configured" in error["error"]


@pytest.mark.asyncio
async def test_serp_provider_invalid_value(mock_context, monkeypatch):
    """Test error when SERP_PROVIDER has invalid value."""
    # Set SERP_PROVIDER to an invalid value
    monkeypatch.setenv("SERP_PROVIDER", "invalid_provider")

    # Get provider
    provider, error = await _get_configured_serp_provider(mock_context)

    # Verify error is returned
    assert provider is None
    assert error is not None
    assert "Invalid SERP_PROVIDER value" in error["error"]


@pytest.mark.asyncio
async def test_serp_provider_auto_select_serpapi(mock_context, monkeypatch):
    """Test auto-selection prioritizes SerpApi when SERP_PROVIDER is not set."""
    # Don't set SERP_PROVIDER (ensure it's unset)
    monkeypatch.delenv("SERP_PROVIDER", raising=False)

    # Mock the secret retrieval to return all keys
    async def get_secret_mock(key):
        secret_map = {
            "SERPAPI_API_KEY": "test_serpapi_key",
            "APIFY_API_KEY": "test_apify_key",
            "DATA_FOR_SEO_LOGIN": "test_login",
            "DATA_FOR_SEO_PASSWORD": "test_password",
        }
        return secret_map.get(key)

    mock_context.get_secret = get_secret_mock

    # Get provider
    provider, error = await _get_configured_serp_provider(mock_context)

    # Verify SerpApi provider is selected (highest priority)
    assert provider is not None
    assert isinstance(provider, SerpApiProvider)
    assert error is None


@pytest.mark.asyncio
async def test_serp_provider_auto_select_apify(mock_context, monkeypatch):
    """Test auto-selection chooses Apify when SerpApi is not available."""
    # Don't set SERP_PROVIDER
    monkeypatch.delenv("SERP_PROVIDER", raising=False)

    # Mock the secret retrieval to return only apify and dataforseo keys
    async def get_secret_mock(key):
        secret_map = {
            "APIFY_API_KEY": "test_apify_key",
            "DATA_FOR_SEO_LOGIN": "test_login",
            "DATA_FOR_SEO_PASSWORD": "test_password",
        }
        return secret_map.get(key)

    mock_context.get_secret = get_secret_mock

    # Get provider
    provider, error = await _get_configured_serp_provider(mock_context)

    # Verify Apify provider is selected
    assert provider is not None
    assert isinstance(provider, ApifyProvider)
    assert error is None


@pytest.mark.asyncio
async def test_serp_provider_auto_select_dataforseo(mock_context, monkeypatch):
    """Test auto-selection chooses DataForSEO when others are not available."""
    # Don't set SERP_PROVIDER
    monkeypatch.delenv("SERP_PROVIDER", raising=False)

    # Mock the secret retrieval to return only dataforseo credentials
    async def get_secret_mock(key):
        if key == "DATA_FOR_SEO_LOGIN":
            return "test_login"
        elif key == "DATA_FOR_SEO_PASSWORD":
            return "test_password"
        return None

    mock_context.get_secret = get_secret_mock

    # Get provider
    provider, error = await _get_configured_serp_provider(mock_context)

    # Verify DataForSEO provider is selected
    assert provider is not None
    assert isinstance(provider, DataForSEOProvider)
    assert error is None


@pytest.mark.asyncio
async def test_serp_provider_no_credentials(mock_context, monkeypatch):
    """Test error when no provider credentials are available."""
    # Don't set SERP_PROVIDER
    monkeypatch.delenv("SERP_PROVIDER", raising=False)

    # Mock the secret retrieval to return None for all keys
    mock_context.get_secret = AsyncMock(return_value=None)

    # Get provider
    provider, error = await _get_configured_serp_provider(mock_context)

    # Verify error is returned
    assert provider is None
    assert error is not None
    assert "No SERP provider is configured" in error["error"]


@pytest.mark.asyncio
async def test_serp_provider_case_insensitive(mock_context, monkeypatch):
    """Test that SERP_PROVIDER setting is case-insensitive."""
    # Set SERP_PROVIDER to uppercase
    monkeypatch.setenv("SERP_PROVIDER", "SERPAPI")

    # Mock the secret retrieval to return serpapi key
    async def get_secret_mock(key):
        if key == "SERPAPI_API_KEY":
            return "test_serpapi_key"
        return None

    mock_context.get_secret = get_secret_mock

    # Get provider
    provider, error = await _get_configured_serp_provider(mock_context)

    # Verify SerpApi provider is selected
    assert provider is not None
    assert isinstance(provider, SerpApiProvider)
    assert error is None
