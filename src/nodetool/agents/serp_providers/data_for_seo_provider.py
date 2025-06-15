from httpx import AsyncClient, HTTPStatusError, RequestError
from nodetool.agents.serp_providers.serp_providers import ErrorResponse, SerpProvider
from nodetool.agents.tools._remove_base64_images import _remove_base64_images
from nodetool.common.environment import Environment


import base64
import json
from typing import Any, Dict, List, Union


class DataForSEOProvider(SerpProvider):
    """
    A SERP provider that uses the DataForSEO API.
    """

    # Default parameters for DataForSEO
    DEFAULT_LOCATION_CODE = 2840  # United States
    DEFAULT_LANGUAGE_CODE = "en"  # English
    DEFAULT_DEVICE = "desktop"
    DEFAULT_OS = "windows"
    DEFAULT_SORT_BY_NEWS = "relevance"  # "date" is also an option

    def __init__(
        self,
        api_login: str | None = None,
        api_password: str | None = None,
        location_code: int = DEFAULT_LOCATION_CODE,
        language_code: str = DEFAULT_LANGUAGE_CODE,
    ):
        self.api_login = api_login or Environment.get("DATA_FOR_SEO_LOGIN")
        self.api_password = api_password or Environment.get("DATA_FOR_SEO_PASSWORD")
        self.location_code = location_code
        self.language_code = language_code
        self._client = AsyncClient(timeout=60.0)  # Re-use client

        if not self.api_login or not self.api_password:
            # This error will be caught by _get_auth_headers and returned as ErrorResponse
            pass

    def _get_auth_headers(self) -> Union[Dict[str, str], ErrorResponse]:
        """
        Retrieves DataForSEO credentials and returns auth headers.
        Returns ErrorResponse if credentials are not found.
        """
        if not self.api_login or not self.api_password:
            return {
                "error": "DataForSEO credentials (DATA_FOR_SEO_LOGIN, DATA_FOR_SEO_PASSWORD) not found or not provided."
            }
        credentials = f"{self.api_login}:{self.api_password}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        return {
            "Authorization": f"Basic {encoded_credentials}",
            "Content-Type": "application/json",
        }

    async def _make_request(
        self, api_url: str, payload: list[dict]
    ) -> Union[Dict[str, Any], ErrorResponse]:
        """
        Makes an asynchronous POST request to the DataForSEO API.
        """
        auth_headers = self._get_auth_headers()
        if "error" in auth_headers:
            return auth_headers  # Propagate error

        try:
            response = await self._client.post(
                api_url, headers=auth_headers, json=payload
            )
            response.raise_for_status()
            return response.json()
        except HTTPStatusError as e:
            error_body_details = e.response.text  # Default to text
            try:
                error_body_details = e.response.json()  # Try to parse JSON
            except json.JSONDecodeError:
                pass  # Keep text if JSON parsing fails
            return {
                "error": f"HTTP error occurred: {e.response.status_code} - {e.response.reason_phrase}",
                "details": error_body_details,
            }
        except RequestError as e:
            return {"error": f"HTTP request failed: {str(e)}"}
        except json.JSONDecodeError as e:  # For errors during response.json() decoding
            return {"error": f"Failed to decode JSON response: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error during DataForSEO request: {str(e)}"}

    async def search(
        self, keyword: str, num_results: int = 10
    ) -> Union[List[Dict[str, Any]], ErrorResponse]:
        payload_dict = {
            "keyword": keyword,
            "location_code": self.location_code,
            "language_code": self.language_code,
            "device": self.DEFAULT_DEVICE,
            "os": self.DEFAULT_OS,
            "depth": num_results,
        }
        payload = [{k: v for k, v in payload_dict.items() if v is not None}]
        url = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"

        result_data = await self._make_request(url, payload)
        if "error" in result_data:
            return result_data

        if (
            result_data.get("status_code") != 20000
            or result_data.get("status_message") != "Ok."
        ):
            return {
                "error": f"DataForSEO API Error: {result_data.get('status_code')} - {result_data.get('status_message')}",
                "details": result_data,
            }

        organic_results = []
        try:
            task_result = result_data.get("tasks", [{}])[0].get("result")
            if task_result and isinstance(task_result, list) and len(task_result) > 0:
                items = task_result[0].get("items", [])
                if items:
                    for item in items:
                        if item.get("type") == "organic":
                            organic_results.append(
                                {
                                    "title": item.get("title"),
                                    "url": item.get("url"),
                                    "snippet": item.get(
                                        "description"
                                    ),  # DataForSEO uses "description" for snippet
                                    "position": item.get("rank_absolute"),
                                    "type": "organic",
                                }
                            )
        except Exception as e:
            return {
                "error": f"Error processing DataForSEO organic search results: {str(e)}",
                "details": result_data,
            }

        return _remove_base64_images(organic_results)

    async def search_news(
        self,
        keyword: str,
        num_results: int = 10,
    ) -> Union[List[Dict[str, Any]], ErrorResponse]:
        payload_dict = {
            "keyword": keyword,
            "location_code": self.location_code,
            "language_code": self.language_code,
            "sort_by": self.DEFAULT_SORT_BY_NEWS,
            "depth": num_results,
        }
        payload = [{k: v for k, v in payload_dict.items() if v is not None}]
        url = "https://api.dataforseo.com/v3/serp/google/news/live/advanced"

        result_data = await self._make_request(url, payload)
        if "error" in result_data:
            return result_data

        if (
            result_data.get("status_code") != 20000
            or result_data.get("status_message") != "Ok."
        ):
            return {
                "error": f"DataForSEO API Error: {result_data.get('status_code')} - {result_data.get('status_message')}",
                "details": result_data,
            }

        news_items_transformed = []
        try:
            task_result = result_data.get("tasks", [{}])[0].get("result")
            if task_result and isinstance(task_result, list) and len(task_result) > 0:
                items = task_result[0].get("items", [])
                if items:
                    for item in items:
                        if item.get("type") in [
                            "news_search",
                            "top_stories",
                        ]:  # As per previous logic
                            # Map DataForSEO fields to our common dictionary structure
                            timestamp_str = item.get(
                                "timestamp"
                            )  # e.g., "2023-10-26 14:30:00 +00:00"
                            published_at = (
                                timestamp_str.split(" ")[0] if timestamp_str else None
                            )  # Extract YYYY-MM-DD

                            news_items_transformed.append(
                                {
                                    "title": item.get("title"),
                                    "url": item.get("url"),
                                    "source": item.get("source"),
                                    "published_at": published_at,
                                    "snippet": item.get(
                                        "description"
                                    ),  # Assuming description is the snippet
                                    "type": "news",
                                }
                            )
        except Exception as e:
            return {
                "error": f"Error processing DataForSEO news search results: {str(e)}",
                "details": result_data,
            }

        return _remove_base64_images(news_items_transformed)

    async def search_images(
        self,
        keyword: str | None = None,
        image_url: str | None = None,
        num_results: int = 20,  # Default from old tool
    ) -> Union[List[Dict[str, Any]], ErrorResponse]:
        if not keyword and not image_url:
            return {
                "error": "One of 'keyword' or 'image_url' is required for image search."
            }

        payload_dict = {
            "keyword": keyword,
            "image_url": image_url,
            "location_code": self.location_code,
            "language_code": self.language_code,
            "depth": num_results,
        }
        payload = [{k: v for k, v in payload_dict.items() if v is not None}]
        api_url = "https://api.dataforseo.com/v3/serp/google/images/live/advanced"

        result_data = await self._make_request(api_url, payload)
        if "error" in result_data:
            return result_data

        if (
            result_data.get("status_code") != 20000
            or result_data.get("status_message") != "Ok."
        ):
            return {
                "error": f"DataForSEO API Error: {result_data.get('status_code')} - {result_data.get('status_message')}",
                "details": result_data,
            }

        image_items_transformed = []
        try:
            task_result = result_data.get("tasks", [{}])[0].get("result")
            if task_result and isinstance(task_result, list) and len(task_result) > 0:
                items = task_result[0].get("items", [])
                if items:
                    for item in items:
                        if item.get("type") == "images_search":
                            image_items_transformed.append(
                                {
                                    "title": item.get("title"),
                                    "image_url": item.get("image_url"),
                                    "source_url": item.get("source_url"),
                                    "alt_text": item.get("alt"),
                                    "type": "image",
                                }
                            )
                        elif item.get("type") == "carousel" and item.get("items"):
                            for carousel_item in item["items"]:
                                if carousel_item.get(
                                    "type"
                                ) == "carousel_element" and carousel_item.get(
                                    "image_url"
                                ):
                                    image_items_transformed.append(
                                        {
                                            "title": carousel_item.get("title"),
                                            "image_url": carousel_item.get("image_url"),
                                            "source_url": carousel_item.get(
                                                "url"
                                            ),  # source of the image for carousel
                                            "alt_text": carousel_item.get("title"),
                                            "type": "image_carousel_element",  # Distinguish if needed
                                        }
                                    )
        except Exception as e:
            return {
                "error": f"Error processing DataForSEO image search results: {str(e)}",
                "details": result_data,
            }

        return _remove_base64_images(image_items_transformed)

    async def search_finance(
        self, query: str, window: str | None = None
    ) -> Union[Dict[str, Any], ErrorResponse]:
        """
        Retrieves financial data. Not currently supported by DataForSEOProvider.
        """
        return {
            "error": "Google Finance search is not supported by DataForSEOProvider."
        }

    async def search_jobs(
        self, query: str, location: str | None = None, num_results: int = 10
    ) -> Union[List[Dict[str, Any]], ErrorResponse]:
        """
        Searches for jobs. Not currently supported by DataForSEOProvider.
        """
        return {"error": "Google Jobs search is not supported by DataForSEOProvider."}

    async def search_lens(
        self, image_url: str, country: str | None = None, num_results: int = 10
    ) -> Union[Dict[str, Any], ErrorResponse]:
        """
        Searches using an image URL (Google Lens). Not currently supported by DataForSEOProvider.
        """
        return {"error": "Google Lens search is not supported by DataForSEOProvider."}

    async def search_maps(
        self,
        query: str,
        ll: str | None = None,
        map_type: str = "search",
        data_id: str | None = None,
        num_results: int = 10,
    ) -> Union[List[Dict[str, Any]], ErrorResponse]:
        """
        Searches Google Maps. Not currently supported by DataForSEOProvider.
        """
        return {"error": "Google Maps search is not supported by DataForSEOProvider."}

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
    ) -> Union[List[Dict[str, Any]], ErrorResponse]:
        """
        Searches for shopping results. Not currently supported by DataForSEOProvider.
        """
        return {
            "error": "Google Shopping search is not supported by DataForSEOProvider."
        }

    async def close(self) -> None:
        """Closes the HTTP client."""
        await self._client.aclose()
