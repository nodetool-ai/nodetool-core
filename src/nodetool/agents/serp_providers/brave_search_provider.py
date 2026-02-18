import json
from contextlib import suppress
from typing import Any

from httpx import AsyncClient, HTTPStatusError, RequestError

from nodetool.agents.serp_providers.serp_providers import ErrorResponse, SerpProvider
from nodetool.agents.tools._remove_base64_images import _remove_base64_images
from nodetool.config.environment import Environment
from nodetool.runtime.resources import maybe_scope, require_scope


class BraveSearchProvider(SerpProvider):
    """
    A SERP provider that uses the Brave Search API.
    API Documentation: https://api.search.brave.com/app/documentation
    """

    WEB_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
    NEWS_SEARCH_URL = "https://api.search.brave.com/res/v1/news/search"
    IMAGES_SEARCH_URL = "https://api.search.brave.com/res/v1/images/search"
    DEFAULT_COUNTRY = "us"
    DEFAULT_LANGUAGE = "en"

    def __init__(
        self,
        api_key: str | None = None,
        country: str = DEFAULT_COUNTRY,
        language: str = DEFAULT_LANGUAGE,
    ):
        self.api_key = api_key or Environment.get("BRAVE_API_KEY")
        self.country = country
        self.language = language
        # HTTP client will be lazily initialized from ResourceScope
        self._client: AsyncClient | None = None

        if not self.api_key:
            raise ValueError("Brave API key (BRAVE_API_KEY) not found or not provided.")

    def _get_client(self) -> AsyncClient:
        """Get or create HTTP client from ResourceScope.

        Uses ResourceScope's HTTP client to ensure correct event loop binding.
        """
        if self._client is None:
            try:
                self._client = require_scope().get_http_client()
            except RuntimeError:
                # Fallback if no scope is bound (shouldn't happen in normal operation)
                self._client = AsyncClient(timeout=60.0)
        return self._client

    def _get_headers(self) -> dict[str, str]:
        """Return headers required for Brave Search API."""
        return {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key or "",
        }

    async def _make_request(
        self, url: str, params: dict[str, Any]
    ) -> dict[str, Any] | ErrorResponse:
        """Make a request to the Brave Search API."""
        if not self.api_key:
            return {"error": "Brave API key (BRAVE_API_KEY) not found or not provided."}

        try:
            response = await self._get_client().get(
                url, params=params, headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except HTTPStatusError as e:
            error_body_details = e.response.text
            with suppress(json.JSONDecodeError):
                error_body_details = e.response.json()
            return {
                "error": f"Brave Search HTTP error: {e.response.status_code} - {e.response.reason_phrase}",
                "details": error_body_details,
            }
        except RequestError as e:
            return {"error": f"Brave Search request failed: {e!s}"}
        except json.JSONDecodeError as e:
            return {"error": f"Brave Search failed to decode JSON response: {e!s}"}
        except Exception as e:
            return {"error": f"Unexpected error during Brave Search request: {e!s}"}

    async def search(
        self, keyword: str, num_results: int = 10
    ) -> dict[str, Any] | ErrorResponse:
        """
        Perform a web search using Brave Search API.
        """
        params = {
            "q": keyword,
            "count": min(num_results, 20),  # Brave API max is 20 per request
            "country": self.country,
            "search_lang": self.language,
            "text_decorations": False,
        }

        result_data = await self._make_request(self.WEB_SEARCH_URL, params)

        if "error" in result_data:
            return result_data

        # Check for API-level errors
        if result_data.get("type") == "ErrorResponse":
            return {
                "error": f"Brave Search API error: {result_data.get('message', 'Unknown error')}",
                "details": result_data,
            }

        return _remove_base64_images(result_data)

    async def search_news(
        self, keyword: str, num_results: int = 10
    ) -> dict[str, Any] | ErrorResponse:
        """
        Perform a news search using Brave Search API.
        """
        params = {
            "q": keyword,
            "count": min(num_results, 20),
            "country": self.country,
            "search_lang": self.language,
            "text_decorations": False,
        }

        result_data = await self._make_request(self.NEWS_SEARCH_URL, params)

        if "error" in result_data:
            return result_data

        if result_data.get("type") == "ErrorResponse":
            return {
                "error": f"Brave Search API error: {result_data.get('message', 'Unknown error')}",
                "details": result_data,
            }

        return _remove_base64_images(result_data)

    async def search_images(
        self,
        keyword: str | None = None,
        image_url: str | None = None,
        num_results: int = 20,
    ) -> dict[str, Any] | ErrorResponse:
        """
        Perform an image search using Brave Search API.
        Note: Brave Search does not support reverse image search (image_url).
        """
        if not keyword and not image_url:
            return {"error": "One of 'keyword' or 'image_url' is required for image search."}

        if image_url:
            return {"error": "Brave Search does not support reverse image search. Please provide a keyword instead."}

        params = {
            "q": keyword,
            "count": min(num_results, 150),  # Brave Images API allows up to 150
            "country": self.country,
            "search_lang": self.language,
        }

        result_data = await self._make_request(self.IMAGES_SEARCH_URL, params)

        if "error" in result_data:
            return result_data

        if result_data.get("type") == "ErrorResponse":
            return {
                "error": f"Brave Search API error: {result_data.get('message', 'Unknown error')}",
                "details": result_data,
            }

        return _remove_base64_images(result_data)

    async def search_finance(
        self, query: str, window: str | None = None
    ) -> dict[str, Any] | ErrorResponse:
        """
        Retrieves financial data. Not supported by Brave Search.
        """
        return {"error": "Finance search is not supported by Brave Search API."}

    async def search_jobs(
        self, query: str, location: str | None = None, num_results: int = 10
    ) -> dict[str, Any] | ErrorResponse:
        """
        Searches for jobs. Not supported by Brave Search.
        """
        return {"error": "Jobs search is not supported by Brave Search API."}

    async def search_lens(
        self, image_url: str, country: str | None = None, num_results: int = 10
    ) -> dict[str, Any] | ErrorResponse:
        """
        Searches using an image URL (lens). Not supported by Brave Search.
        """
        return {"error": "Lens/reverse image search is not supported by Brave Search API."}

    async def search_maps(
        self,
        query: str,
        ll: str | None = None,
        map_type: str = "search",
        data_id: str | None = None,
        num_results: int = 10,
    ) -> dict[str, Any] | ErrorResponse:
        """
        Searches maps. Not supported by Brave Search.
        """
        return {"error": "Maps search is not supported by Brave Search API."}

    async def search_shopping(
        self,
        query: str,
        country: str | None = None,
        domain: str | None = None,
        min_price: int | None = None,
        max_price: int | None = None,
        condition: str | None = None,
        sort_by: str | None = None,
        num_results: int = 10,
    ) -> dict[str, Any] | ErrorResponse:
        """
        Searches for shopping results. Not supported by Brave Search.
        """
        return {"error": "Shopping search is not supported by Brave Search API."}

    async def close(self) -> None:
        """Clean up resources."""
        # Only close if we created the client ourselves (not from ResourceScope)
        if self._client is not None:
            try:
                # Check if this is the scope's client by trying to get scope
                scope = maybe_scope()
                if scope and scope.get_http_client() is self._client:
                    # Don't close scope-managed client
                    return
            except Exception:
                pass
            await self._client.aclose()
