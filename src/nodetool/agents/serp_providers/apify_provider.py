import json
from contextlib import suppress
from typing import Any

from httpx import AsyncClient, HTTPStatusError, RequestError

from nodetool.agents.serp_providers._remove_base64_images import _remove_base64_images
from nodetool.agents.serp_providers.serp_providers import ErrorResponse, SerpProvider
from nodetool.config.environment import Environment
from nodetool.runtime.resources import maybe_scope, require_scope


class ApifyProvider(SerpProvider):
    """
    A SERP provider that uses the Apify API.
    Apify provides various actors (scraping tools) for different search engines.
    API Documentation: https://docs.apify.com/api/v2
    """

    BASE_URL = "https://api.apify.com/v2"
    # Apify actor IDs for various Google search types
    GOOGLE_SEARCH_ACTOR = "apify/google-search-scraper"
    GOOGLE_NEWS_ACTOR = "apify/google-news-scraper"
    GOOGLE_IMAGES_ACTOR = "apify/google-images-scraper"
    GOOGLE_MAPS_ACTOR = "apify/google-maps-scraper"
    GOOGLE_SHOPPING_ACTOR = "apify/google-shopping-scraper"

    DEFAULT_COUNTRY_CODE = "us"
    DEFAULT_LANGUAGE_CODE = "en"

    def __init__(
        self,
        api_key: str | None = None,
        country_code: str = DEFAULT_COUNTRY_CODE,
        language_code: str = DEFAULT_LANGUAGE_CODE,
    ):
        self.api_key = api_key or Environment.get("APIFY_API_KEY")
        self.country_code = country_code
        self.language_code = language_code
        # HTTP client will be lazily initialized from ResourceScope
        self._client: AsyncClient | None = None

        if not self.api_key:
            raise ValueError("Apify API key (APIFY_API_KEY) not found or not provided.")

    def _get_client(self) -> AsyncClient:
        """Get or create HTTP client from ResourceScope.

        Uses ResourceScope's HTTP client to ensure correct event loop binding.
        """
        if self._client is None:
            try:
                self._client = require_scope().get_http_client()
            except RuntimeError:
                # Fallback if no scope is bound (shouldn't happen in normal operation)
                self._client = AsyncClient(timeout=120.0)  # Apify runs can take longer
        return self._client

    async def _run_actor_and_wait(
        self, actor_id: str, run_input: dict[str, Any]
    ) -> dict[str, Any] | ErrorResponse:
        """
        Runs an Apify actor and waits for results.

        Args:
            actor_id: The ID of the Apify actor to run
            run_input: The input configuration for the actor

        Returns:
            The actor run results or an error response
        """
        if not self.api_key:
            return {"error": "Apify API key (APIFY_API_KEY) not found or not provided."}

        # Start the actor run
        run_url = f"{self.BASE_URL}/acts/{actor_id}/runs?token={self.api_key}"

        try:
            # Start the run
            response = await self._get_client().post(
                run_url,
                json=run_input,
                params={"waitForFinish": 120},  # Wait up to 120 seconds for completion
            )
            response.raise_for_status()
            run_data = response.json()

            # Check if run was successful
            status = run_data.get("data", {}).get("status")
            if status != "SUCCEEDED":
                return {
                    "error": f"Apify actor run failed with status: {status}",
                    "details": run_data,
                }

            # Get the default dataset ID
            default_dataset_id = run_data.get("data", {}).get("defaultDatasetId")
            if not default_dataset_id:
                return {"error": "No dataset ID returned from Apify actor run"}

            # Fetch the results from the dataset
            dataset_url = f"{self.BASE_URL}/datasets/{default_dataset_id}/items?token={self.api_key}"
            dataset_response = await self._get_client().get(dataset_url)
            dataset_response.raise_for_status()
            results = dataset_response.json()

            return results

        except HTTPStatusError as e:
            error_body_details = e.response.text
            with suppress(json.JSONDecodeError):
                error_body_details = e.response.json()
            return {
                "error": f"Apify HTTP error: {e.response.status_code} - {e.response.reason_phrase}",
                "details": error_body_details,
            }
        except RequestError as e:
            return {"error": f"Apify request failed: {e!s}"}
        except json.JSONDecodeError as e:
            return {"error": f"Apify failed to decode JSON response: {e!s}"}
        except Exception as e:
            return {"error": f"Unexpected error during Apify request: {e!s}"}

    async def search(self, keyword: str, num_results: int = 10) -> Any:
        """
        Perform an organic web search using Apify's Google Search Scraper.
        """
        run_input = {
            "queries": keyword,
            "resultsPerPage": num_results,
            "maxPagesPerQuery": 1,
            "languageCode": self.language_code,
            "countryCode": self.country_code.upper(),
            "mobileResults": False,
        }

        result_data = await self._run_actor_and_wait(self.GOOGLE_SEARCH_ACTOR, run_input)

        if isinstance(result_data, dict) and "error" in result_data:
            return result_data

        # Transform Apify results to common format
        if isinstance(result_data, list):
            transformed_results = []
            for item in result_data[:num_results]:
                if item.get("organicResults"):
                    for organic in item["organicResults"][:num_results]:
                        transformed_results.append(
                            {
                                "title": organic.get("title"),
                                "url": organic.get("url"),
                                "snippet": organic.get("description"),
                                "position": organic.get("rank"),
                                "type": "organic",
                            }
                        )
            return _remove_base64_images({"organic_results": transformed_results})

        return _remove_base64_images(result_data)

    async def search_news(self, keyword: str, num_results: int = 10) -> Any:
        """
        Perform a news search using Apify's Google News Scraper.
        """
        run_input = {
            "queries": keyword,
            "maxItems": num_results,
            "languageCode": self.language_code,
            "countryCode": self.country_code.upper(),
        }

        result_data = await self._run_actor_and_wait(self.GOOGLE_NEWS_ACTOR, run_input)

        if isinstance(result_data, dict) and "error" in result_data:
            return result_data

        # Transform results to common format
        if isinstance(result_data, list):
            transformed_results = []
            for item in result_data[:num_results]:
                transformed_results.append(
                    {
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "source": item.get("source"),
                        "published_at": item.get("date"),
                        "snippet": item.get("description"),
                        "type": "news",
                    }
                )
            return _remove_base64_images(transformed_results)

        return _remove_base64_images(result_data)

    async def search_images(
        self,
        keyword: str | None = None,
        image_url: str | None = None,
        num_results: int = 20,
    ) -> Any:
        """
        Perform an image search using Apify's Google Images Scraper.
        """
        if not keyword and not image_url:
            return {"error": "One of 'keyword' or 'image_url' is required for image search."}

        run_input = {
            "queries": keyword if keyword else "",
            "maxItems": num_results,
            "languageCode": self.language_code,
            "countryCode": self.country_code.upper(),
        }

        if image_url:
            # Note: Reverse image search might require a different actor or configuration
            return {"error": "Reverse image search is not currently supported by ApifyProvider."}

        result_data = await self._run_actor_and_wait(self.GOOGLE_IMAGES_ACTOR, run_input)

        if isinstance(result_data, dict) and "error" in result_data:
            return result_data

        # Transform results to common format
        if isinstance(result_data, list):
            transformed_results = []
            for item in result_data[:num_results]:
                transformed_results.append(
                    {
                        "title": item.get("title"),
                        "image_url": item.get("imageUrl"),
                        "source_url": item.get("sourceUrl"),
                        "alt_text": item.get("alt"),
                        "type": "image",
                    }
                )
            return _remove_base64_images(transformed_results)

        return _remove_base64_images(result_data)

    async def search_finance(self, query: str, window: str | None = None) -> Any:
        """
        Retrieves financial data. Not currently supported by ApifyProvider.
        """
        return {"error": "Google Finance search is not supported by ApifyProvider."}

    async def search_jobs(
        self, query: str, location: str | None = None, num_results: int = 10
    ) -> Any:
        """
        Searches for jobs. Not currently supported by ApifyProvider.
        """
        return {"error": "Google Jobs search is not supported by ApifyProvider."}

    async def search_lens(
        self, image_url: str, country: str | None = None, num_results: int = 10
    ) -> Any:
        """
        Searches using an image URL (Google Lens). Not currently supported by ApifyProvider.
        """
        return {"error": "Google Lens search is not supported by ApifyProvider."}

    async def search_maps(
        self,
        query: str,
        ll: str | None = None,
        map_type: str = "search",
        data_id: str | None = None,
        num_results: int = 10,
    ) -> Any:
        """
        Searches Google Maps using Apify's Google Maps Scraper.
        """
        run_input = {
            "searchStringsArray": [query],
            "maxCrawledPlacesPerSearch": num_results,
            "language": self.language_code,
            "countryCode": self.country_code.upper(),
        }

        if ll:
            # Parse coordinates from ll parameter (format: "@lat,lng,zoom")
            try:
                coords = ll.strip("@").split(",")
                if len(coords) >= 2:
                    run_input["lat"] = coords[0]
                    run_input["lng"] = coords[1]
            except Exception:
                pass  # Use default location if parsing fails

        result_data = await self._run_actor_and_wait(self.GOOGLE_MAPS_ACTOR, run_input)

        if isinstance(result_data, dict) and "error" in result_data:
            return result_data

        return _remove_base64_images(result_data)

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
    ) -> Any:
        """
        Searches Google Shopping using Apify's Google Shopping Scraper.
        """
        run_input = {
            "queries": query,
            "maxItems": num_results,
            "languageCode": self.language_code,
            "countryCode": (country or self.country_code).upper(),
        }

        # Add price filters if provided
        if min_price is not None:
            run_input["minPrice"] = min_price
        if max_price is not None:
            run_input["maxPrice"] = max_price

        # Note: Condition and sort_by might need different parameter names in Apify
        # Check Apify documentation for exact parameter names

        result_data = await self._run_actor_and_wait(self.GOOGLE_SHOPPING_ACTOR, run_input)

        if isinstance(result_data, dict) and "error" in result_data:
            return result_data

        return _remove_base64_images(result_data)

    async def close(self) -> None:
        """Clean up any resources (e.g., close HTTP clients)."""
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
