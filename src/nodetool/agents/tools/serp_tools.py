import json
import os
from typing import Any, List, Dict, Optional, Union, TypeVar
import base64
from datetime import datetime
import time

from nodetool.agents.tools.base import Tool
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.common.environment import Environment
from nodetool.agents.tools.serp_providers import SerpProvider, ErrorResponse
from httpx import AsyncClient, HTTPStatusError, RequestError


T = TypeVar("T")


def _remove_base64_images(data: T) -> T:
    """Remove image elements entirely from the API response to reduce size."""
    if isinstance(data, dict):
        keys_to_remove = ["image", "image_alt", "image_base64", "image_url"]
        for key in list(data.keys()):
            if key in keys_to_remove:
                data.pop(key, None)
            elif isinstance(data[key], str):
                if data[key].startswith("data:"):
                    data.pop(key, None)
            elif isinstance(data[key], (dict, list)):
                data[key] = _remove_base64_images(data[key])
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = _remove_base64_images(data[i])
    return data


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
        api_login: Optional[str] = None,
        api_password: Optional[str] = None,
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
        keyword: Optional[str] = None,
        image_url: Optional[str] = None,
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
        self, query: str, window: Optional[str] = None
    ) -> Union[Dict[str, Any], ErrorResponse]:
        """
        Retrieves financial data. Not currently supported by DataForSEOProvider.
        """
        return {
            "error": "Google Finance search is not supported by DataForSEOProvider."
        }

    async def search_jobs(
        self, query: str, location: Optional[str] = None, num_results: int = 10
    ) -> Union[List[Dict[str, Any]], ErrorResponse]:
        """
        Searches for jobs. Not currently supported by DataForSEOProvider.
        """
        return {"error": "Google Jobs search is not supported by DataForSEOProvider."}

    async def search_lens(
        self, image_url: str, country: Optional[str] = None, num_results: int = 10
    ) -> Union[Dict[str, Any], ErrorResponse]:
        """
        Searches using an image URL (Google Lens). Not currently supported by DataForSEOProvider.
        """
        return {"error": "Google Lens search is not supported by DataForSEOProvider."}

    async def search_maps(
        self,
        query: str,
        ll: Optional[str] = None,
        map_type: str = "search",
        data_id: Optional[str] = None,
        num_results: int = 10,
    ) -> Union[List[Dict[str, Any]], ErrorResponse]:
        """
        Searches Google Maps. Not currently supported by DataForSEOProvider.
        """
        return {"error": "Google Maps search is not supported by DataForSEOProvider."}

    async def search_shopping(
        self,
        query: str,
        country: Optional[str] = None,
        domain: Optional[str] = None,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        condition: Optional[str] = None,
        sort_by: Optional[str] = None,
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


class GoogleSearchTool(Tool):
    name = "google_search"
    description = "Search Google to retrieve organic search results. Uses available SERP provider."
    input_schema = {
        "type": "object",
        "properties": {
            "keyword": {
                "type": "string",
                "description": "The keyword to search for.",
            },
            "num_results": {
                "type": "integer",
                "description": "The number of search results to retrieve.",
                "default": 10,
            },
        },
        "required": ["keyword"],
    }
    example = """
    google_search(
        keyword="weather forecast",
        num_results=5
    )
    """

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        keyword = params.get("keyword")
        if not keyword:
            return {"error": "Keyword is required"}

        num_results = params.get("num_results", 10)

        provider_instance, error_response = _get_configured_serp_provider()
        if error_response:
            return error_response
        if (
            not provider_instance
        ):  # Should not happen if error_response is None, but as a safeguard
            return {"error": "Failed to initialize SERP provider."}

        async with provider_instance as provider:
            result_data = await provider.search(
                keyword=keyword, num_results=num_results
            )

        if "error" in result_data:
            return result_data  # This includes errors from the provider itself

        return {"success": True, "results": result_data}

    def user_message(self, params: dict) -> str:
        keyword = params.get("keyword", "something")
        msg = f"Searching Google for '{keyword}'..."
        if len(msg) > 80:
            msg = "Searching Google..."
        return msg


class GoogleNewsTool(Tool):
    name = "google_news"
    description = "Search Google News to retrieve live news articles. Uses available SERP provider."
    input_schema = {
        "type": "object",
        "properties": {
            "keyword": {
                "type": "string",
                "description": "The keyword to search for in Google News.",
            },
            "num_results": {
                "type": "integer",
                "description": "The number of news results to retrieve.",
                "default": 10,
            },
        },
        "required": ["keyword"],
    }
    example = """
    google_news(
        keyword="artificial intelligence",
        num_results=5
    )
    """

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        keyword = params.get("keyword")
        if not keyword:
            return {"error": "Keyword is required"}

        num_results = params.get("num_results", 10)

        provider_instance, error_response = _get_configured_serp_provider()
        if error_response:
            return error_response
        if not provider_instance:
            return {"error": "Failed to initialize SERP provider."}

        async with provider_instance as provider:
            result_data = await provider.search_news(
                keyword=keyword, num_results=num_results
            )

        if "error" in result_data:
            return result_data

        return {"success": True, "results": result_data}

    def user_message(self, params: dict) -> str:
        keyword = params.get("keyword", "something")
        msg = f"Searching Google News for '{keyword}'..."
        if len(msg) > 80:
            msg = "Searching Google News..."
        return msg


class GoogleImagesTool(Tool):
    name = "google_images"
    description = "Search Google Images to retrieve live image results. Uses available SERP provider."
    input_schema = {
        "type": "object",
        "properties": {
            "keyword": {
                "type": "string",
                "description": "Keyword for image search. (Optional if image_url is provided)",
            },
            "image_url": {
                "type": "string",
                "description": "URL of an image for reverse search. (Optional if keyword is provided)",
            },
            "num_results": {
                "type": "integer",
                "description": "The number of image results to retrieve.",
                "default": 20,
            },
        },
    }
    example = """
    google_images(
        keyword="cats",
        num_results=10
    )
    """

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        keyword = params.get("keyword")
        image_url = params.get("image_url")
        num_results = params.get("num_results", 20)

        if not keyword and not image_url:
            return {"error": "One of 'keyword' or 'image_url' is required."}

        provider_instance, error_response = _get_configured_serp_provider()
        if error_response:
            return error_response
        if not provider_instance:
            return {"error": "Failed to initialize SERP provider."}

        async with provider_instance as provider:
            result_data = await provider.search_images(
                keyword=keyword, image_url=image_url, num_results=num_results
            )

        if "error" in result_data:
            return result_data

        return {"success": True, "results": result_data}

    def user_message(self, params: dict) -> str:
        keyword = params.get("keyword")
        if keyword:
            search_term = f" '{keyword}'"
        elif params.get("image_url"):
            search_term = " an image URL"
        else:
            search_term = " something"
        msg = f"Searching Google Images for{search_term}..."
        if len(msg) > 80:
            msg = "Searching Google Images..."
        return msg


class GoogleFinanceTool(Tool):
    name = "google_finance"
    description = "Retrieve financial market data from Google Finance. Uses available SERP provider."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The financial query, e.g., a stock ticker like 'GOOGL:NASDAQ' or a market index like '.DJI:INDEXDJX'.",
            },
            "window": {
                "type": "string",
                "description": "The time window for historical data (e.g., '1D', '5D', '1M', '6M', '1Y', '5Y', 'MAX'). If not provided, defaults to a standard view.",
                "optional": True,
            },
        },
        "required": ["query"],
    }
    example = """
    google_finance(
        query="AAPL:NASDAQ",
        window="1M"
    )
    """

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        query = params.get("query")
        if not query:
            return {"error": "Query is required for Google Finance search."}

        window = params.get("window")  # Can be None

        provider_instance, error_response = _get_configured_serp_provider()
        if error_response:
            return error_response
        if not provider_instance:
            return {"error": "Failed to initialize SERP provider."}

        async with provider_instance as provider:
            result_data = await provider.search_finance(query=query, window=window)

        if "error" in result_data:  # This includes errors from the provider itself
            return result_data

        return {"success": True, "results": result_data}

    def user_message(self, params: dict) -> str:
        query = params.get("query", "a financial entity")
        msg = f"Retrieving Google Finance data for '{query}'..."
        if len(msg) > 80:
            msg = "Retrieving Google Finance data..."
        return msg


class GoogleJobsTool(Tool):
    name = "google_jobs"
    description = "Search Google Jobs for job listings. Uses available SERP provider."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The job search query (e.g., 'software engineer', 'barista').",
            },
            "location": {
                "type": "string",
                "description": "The location to search for jobs in (e.g., 'New York, NY', 'Remote'). Optional.",
                "optional": True,
            },
            "num_results": {
                "type": "integer",
                "description": "The number of job results to retrieve.",
                "default": 10,
            },
        },
        "required": ["query"],
    }
    example = """
    google_jobs(
        query="python developer",
        location="Austin, TX",
        num_results=5
    )
    """

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        query = params.get("query")
        if not query:
            return {"error": "Query is required for Google Jobs search."}

        location = params.get("location")
        num_results = params.get("num_results", 10)

        provider_instance, error_response = _get_configured_serp_provider()
        if error_response:
            return error_response
        if not provider_instance:
            return {"error": "Failed to initialize SERP provider."}

        async with provider_instance as provider:
            result_data = await provider.search_jobs(
                query=query, location=location, num_results=num_results
            )

        if "error" in result_data:  # This includes errors from the provider itself
            return result_data

        return {"success": True, "results": result_data}

    def user_message(self, params: dict) -> str:
        query = params.get("query", "jobs")
        location_info = params.get("location")
        location_str = f" in {location_info}" if location_info else ""
        msg = f"Searching Google Jobs for '{query}'{location_str}..."
        if len(msg) > 80:
            msg = "Searching Google Jobs..."
        return msg


class GoogleLensTool(Tool):
    name = "google_lens"
    description = "Search with an image URL using Google Lens to find visual matches and related content. Uses available SERP provider."
    input_schema = {
        "type": "object",
        "properties": {
            "image_url": {
                "type": "string",
                "description": "The URL of the image to search with.",
            },
            "num_results": {
                "type": "integer",
                "description": "The maximum number of visual matches to retrieve.",
                "default": 10,
            },
        },
        "required": ["image_url"],
    }
    example = """
    google_lens(
        image_url="https://example.com/image.jpg",
        num_results=5
    )
    """

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        image_url = params.get("image_url")
        if not image_url:
            return {"error": "Image URL is required for Google Lens search."}

        image_url = params.get("image_url")
        num_results = params.get("num_results", 10)

        assert image_url is not None, "Image URL is required for Google Lens search."

        provider_instance, error_response = _get_configured_serp_provider()
        if error_response:
            return error_response
        if not provider_instance:
            return {"error": "Failed to initialize SERP provider."}

        async with provider_instance as provider:
            result_data = await provider.search_lens(
                image_url=image_url, num_results=num_results
            )

        if "error" in result_data:
            return result_data

        return {"success": True, "results": result_data}

    def user_message(self, params: dict) -> str:
        image_url = params.get("image_url", "an image")
        msg = f"Searching Google Lens with {image_url}..."
        if len(msg) > 80:
            msg = "Searching Google Lens with an image..."
        return msg


class GoogleMapsTool(Tool):
    name = "google_maps"
    description = "Search Google Maps for places or get details about a specific place. Uses available SERP provider."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query (e.g., 'restaurants in New York', 'Eiffel Tower'). Required if 'data_id' is not provided.",
                "optional": True,
            },
            "num_results": {
                "type": "integer",
                "description": "The number of map results to retrieve for 'search' type.",
                "default": 10,
            },
        },
        # "required" depends on map_type, handled in process logic
    }
    example = """
    # General search
    google_maps(
        query="pizza near me",
        ll="@40.7455096,-74.0083012,14z",
        num_results=5
    )

    # Place details (example data_id)
    google_maps(
        map_type="place",
        data_id="0x89c2589a018531e3:0xb9df1f7387a94119"
    )
    """

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        query = params.get("query")
        num_results = params.get("num_results", 10)

        if not query:
            return {"error": "Query is required for map_type 'search'."}

        provider_instance, error_response = _get_configured_serp_provider()
        if error_response:
            return error_response
        if not provider_instance:
            return {"error": "Failed to initialize SERP provider."}

        async with provider_instance as provider:
            result_data = await provider.search_maps(
                query=query,
                num_results=num_results,
            )

        if "error" in result_data:
            return result_data

        return {"success": True, "results": result_data}

    def user_message(self, params: dict) -> str:
        query = params.get("query")
        if query:
            search_term = f"'{query}'"
        else:
            search_term = f"'{params.get('query', 'places')}'"

        msg = f"Searching Google Maps for {search_term}..."
        if len(msg) > 80:
            msg = "Searching Google Maps..."
        return msg


class GoogleShoppingTool(Tool):
    name = "google_shopping"
    description = "Search Google Shopping for products. Uses available SERP provider."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The product search query (e.g., 'running shoes', 'coffee maker').",
            },
            "country": {
                "type": "string",
                "description": "The country code to search in (e.g., 'us', 'ca', 'gb'). Corresponds to 'gl' parameter.",
                "optional": True,
            },
            "domain": {
                "type": "string",
                "description": "The Google domain to use (e.g., 'google.com', 'google.co.uk'). Corresponds to 'google_domain'.",
                "optional": True,
            },
            "min_price": {
                "type": "integer",
                "description": "Minimum product price.",
                "optional": True,
            },
            "max_price": {
                "type": "integer",
                "description": "Maximum product price.",
                "optional": True,
            },
            "condition": {
                "type": "string",
                "description": "Product condition: 'new', 'used', or 'refurbished'.",
                "enum": ["new", "used", "refurbished"],
                "optional": True,
            },
            "sort_by": {
                "type": "string",
                "description": "Sort order. E.g., 'p_price' (price ascending), 'pd_price' (price descending), 'r' (relevance/rating). Check SerpApi docs for specific values.",
                "optional": True,
            },
            "num_results": {
                "type": "integer",
                "description": "The number of shopping results to retrieve.",
                "default": 10,
            },
        },
        "required": ["query"],
    }
    example = """
    google_shopping(
        query="wireless headphones",
        num_results=15
    )
    """

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        query = params.get("query")
        if not query:
            return {"error": "Query is required for Google Shopping search."}

        provider_instance, error_response = _get_configured_serp_provider()
        if error_response:
            return error_response
        if not provider_instance:
            return {"error": "Failed to initialize SERP provider."}

        async with provider_instance as provider:
            result_data = await provider.search_shopping(
                query=query,
                country=params.get("country", "us"),
                min_price=params.get("min_price"),
                max_price=params.get("max_price"),
                condition=params.get("condition"),
                sort_by=params.get("sort_by"),
                num_results=params.get("num_results", 10),
            )

        if "error" in result_data:
            return result_data

        return {"success": True, "results": result_data}

    def user_message(self, params: dict) -> str:
        query = params.get("query", "products")
        msg = f"Searching Google Shopping for '{query}'..."
        if len(msg) > 80:
            msg = "Searching Google Shopping..."
        return msg


class SerpApiProvider(SerpProvider):
    """
    A SERP provider that uses the SerpApi.com API.
    API Documentation: https://serpapi.com/search-api
    """

    BASE_URL = "https://serpapi.com/search.json"
    DEFAULT_GL = "us"  # Country
    DEFAULT_HL = "en"  # Language

    def __init__(
        self,
        api_key: Optional[str] = None,
        gl: str = DEFAULT_GL,
        hl: str = DEFAULT_HL,
    ):
        self.api_key = api_key or Environment.get("SERPAPI_API_KEY")
        self.gl = gl
        self.hl = hl
        self._client = AsyncClient(timeout=60.0)

        if not self.api_key:
            raise ValueError(
                "SerpApi API key (SERPAPI_API_KEY) not found or not provided."
            )

    async def _make_request(
        self, params: Dict[str, Any]
    ) -> Union[Dict[str, Any], ErrorResponse]:
        if not self.api_key:
            return {
                "error": "SerpApi API key (SERPAPI_API_KEY) not found or not provided."
            }

        all_params = {**params, "api_key": self.api_key}

        try:
            response = await self._client.get(self.BASE_URL, params=all_params)
            response.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx responses
            return response.json()
        except HTTPStatusError as e:
            error_body_details = e.response.text
            try:
                error_body_details = e.response.json()
            except json.JSONDecodeError:
                pass
            return {
                "error": f"SerpApi HTTP error: {e.response.status_code} - {e.response.reason_phrase}",
                "details": error_body_details,
            }
        except RequestError as e:
            return {"error": f"SerpApi request failed: {str(e)}"}
        except json.JSONDecodeError as e:
            return {"error": f"SerpApi failed to decode JSON response: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error during SerpApi request: {str(e)}"}

    async def search(
        self, keyword: str, num_results: int = 10
    ) -> Union[List[Dict[str, Any]], ErrorResponse]:
        params = {
            "engine": "google_light",
            "q": keyword,
            "num": num_results,
            "gl": self.gl,
            "hl": self.hl,
        }
        result_data = await self._make_request(params)

        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data

        # Check for SerpApi's own error reporting
        serpapi_error_status = (
            result_data.get("search_metadata", {}).get("status") == "Error"
        )
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            return {
                "error": result_data.get("error", "SerpApi returned an error."),
                "details": result_data,
            }

        organic_results_transformed = []
        try:
            serp_organic_results = result_data.get("organic_results", [])
            for item in serp_organic_results:
                organic_results_transformed.append(
                    {
                        "title": item.get("title"),
                        "url": item.get("link"),
                        "snippet": item.get("snippet"),
                        "position": item.get("position"),
                        "type": "organic",
                    }
                )
        except Exception as e:
            return {
                "error": f"Error processing SerpApi organic search results: {str(e)}",
                "details": result_data,
            }

        return _remove_base64_images(organic_results_transformed)

    async def search_news(
        self,
        keyword: str,
        num_results: int = 10,
    ) -> Union[List[Dict[str, Any]], ErrorResponse]:
        params = {
            "engine": "google_news",
            "q": keyword,
            "num": num_results,
            "gl": self.gl,
            "hl": self.hl,
        }

        result_data = await self._make_request(params)

        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data

        serpapi_error_status = (
            result_data.get("search_metadata", {}).get("status") == "Error"
        )
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            return {
                "error": result_data.get(
                    "error", "SerpApi returned an error for news search."
                ),
                "details": result_data,
            }

        news_results_transformed = []
        try:
            serp_news_results = result_data.get("news_results", [])
            for item in serp_news_results:
                news_results_transformed.append(
                    {
                        "title": item.get("title"),
                        "url": item.get("link"),
                        "source": item.get("source"),
                        "published_at": item.get("date"),
                        "snippet": item.get("snippet"),
                        "type": "news",
                    }
                )
        except Exception as e:
            return {
                "error": f"Error processing SerpApi news results: {str(e)}",
                "details": result_data,
            }

        return _remove_base64_images(news_results_transformed)

    async def search_images(
        self,
        keyword: Optional[str] = None,
        image_url: Optional[str] = None,
        num_results: int = 20,
    ) -> Union[List[Dict[str, Any]], ErrorResponse]:
        if not keyword and not image_url:
            return {
                "error": "One of 'keyword' or 'image_url' is required for image search."
            }

        params = {
            "engine": "google_images",
            "num": num_results,
            "gl": self.gl,
            "hl": self.hl,
        }
        if keyword:
            params["q"] = keyword
        if image_url:
            params["engine"] = "google_reverse_image"
            params["image_url"] = image_url
            if "q" in params:
                del params["q"]

        result_data = await self._make_request(params)

        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data

        serpapi_error_status = (
            result_data.get("search_metadata", {}).get("status") == "Error"
        )
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            return {
                "error": result_data.get(
                    "error", "SerpApi returned an error for image search."
                ),
                "details": result_data,
            }

        image_results_transformed = []
        try:
            serp_image_results = result_data.get("image_results", [])
            if (
                not serp_image_results
                and params.get("engine") == "google_reverse_image"
            ):
                serp_image_results = result_data.get("image_sources", [])
                if not serp_image_results:
                    serp_image_results = result_data.get("visual_matches", [])

            for item in serp_image_results:
                image_results_transformed.append(
                    {
                        "title": item.get("title"),
                        "image_url": item.get("original") or item.get("image_url"),
                        "source_url": item.get("source") or item.get("link"),
                        "alt_text": item.get("title"),
                        "type": "image",
                    }
                )
        except Exception as e:
            return {
                "error": f"Error processing SerpApi image results: {str(e)}",
                "details": result_data,
            }

        return _remove_base64_images(image_results_transformed)

    async def search_finance(
        self, query: str, window: Optional[str] = None
    ) -> Union[Dict[str, Any], ErrorResponse]:
        """
        Retrieves financial data using SerpApi's Google Finance engine.
        """
        params = {
            "engine": "google_finance",
            "q": query,
            "gl": self.gl,
            "hl": self.hl,
        }
        if window:
            params["window"] = window

        result_data = await self._make_request(params)

        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            # This case handles critical errors from _make_request (e.g., API key missing, network issue)
            return result_data

        # Check for SerpApi's own error reporting within a successful HTTP response
        serpapi_error_status = (
            result_data.get("search_metadata", {}).get("status") == "Error"
        )
        # SerpApi might also return an error message directly at the top level
        serpapi_error_message = isinstance(result_data.get("error"), str)

        if serpapi_error_status or serpapi_error_message:
            return {
                "error": result_data.get(
                    "error", "SerpApi returned an error for finance search."
                ),
                "details": result_data,  # Include the full response for debugging
            }

        # At this point, we expect a successful response structure.
        # We return the whole result_data as the structure is complex and specific to finance.
        # The GoogleFinanceTool or its consumer can then parse specific parts (summary, graph, key_events).
        # No need to transform like organic/news/images as the structure is inherently different.
        return _remove_base64_images(result_data)

    async def search_jobs(
        self, query: str, location: Optional[str] = None, num_results: int = 10
    ) -> Union[List[Dict[str, Any]], ErrorResponse]:
        """
        Searches for jobs using SerpApi's Google Jobs engine.
        """
        params = {
            "engine": "google_jobs",
            "q": query,
            "hl": self.hl,
            "gl": self.gl,
            # num parameter for serpapi for jobs is not directly supported for count.
            # We take num_results and slice the results later.
        }
        if location:
            params["location"] = location

        result_data = await self._make_request(params)

        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data

        serpapi_error_status = (
            result_data.get("search_metadata", {}).get("status") == "Error"
        )
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            return {
                "error": result_data.get(
                    "error", "SerpApi returned an error for jobs search."
                ),
                "details": result_data,
            }

        jobs_results_transformed = []
        try:
            serp_jobs_results = result_data.get("jobs_results", [])
            for item in serp_jobs_results[:num_results]:  # Apply num_results limit here
                job = {
                    "title": item.get("title"),
                    "company_name": item.get("company_name"),
                    "location": item.get("location"),
                    "via": item.get("via"),
                    "description": item.get("description"),
                    "job_highlights": item.get("job_highlights", []),
                    # "related_links" was in my previous thought but not in API example, remove for now
                    "thumbnail": item.get("thumbnail"),
                    "extensions": item.get("extensions", []),
                    "detected_extensions": item.get("detected_extensions", {}),
                    "job_id": item.get("job_id"),
                    "apply_options": item.get("apply_options", []),
                }
                jobs_results_transformed.append(
                    {k: v for k, v in job.items() if v is not None}
                )

        except Exception as e:
            return {
                "error": f"Error processing SerpApi job search results: {str(e)}",
                "details": result_data,
            }
        return _remove_base64_images(jobs_results_transformed)

    async def search_lens(
        self, image_url: str, country: Optional[str] = None, num_results: int = 10
    ) -> Union[Dict[str, Any], ErrorResponse]:
        """
        Searches with an image URL using SerpApi's Google Lens engine.
        """
        params = {
            "engine": "google_lens",
            "url": image_url,
            "hl": self.hl,
            # "gl" is supported by google_lens for country, but SerpApi uses "country" for this engine
        }
        if country:
            params["country"] = country  # SerpApi specific parameter for Google Lens

        result_data = await self._make_request(params)

        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data

        serpapi_error_status = (
            result_data.get("search_metadata", {}).get("status") == "Error"
        )
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            return {
                "error": result_data.get(
                    "error", "SerpApi returned an error for Google Lens search."
                ),
                "details": result_data,
            }

        # Process visual matches and limit them
        processed_result = (
            result_data.copy()
        )  # Start with a copy of the original result
        if "visual_matches" in processed_result and isinstance(
            processed_result["visual_matches"], list
        ):
            processed_result["visual_matches"] = processed_result["visual_matches"][
                :num_results
            ]

        # Other potential fields to keep: "related_content", "image_sources", "products"
        # For now, return the processed_result which includes these if present.
        # _remove_base64_images will handle cleaning up any embedded images in the response.
        return _remove_base64_images(processed_result)

    async def search_maps(
        self,
        query: str,
        ll: Optional[str] = None,
        map_type: str = "search",
        data_id: Optional[str] = None,
        num_results: int = 10,
    ) -> Union[List[Dict[str, Any]], ErrorResponse]:
        """
        Searches Google Maps using SerpApi.
        If map_type is 'place', data_id is required and query/ll are ignored by SerpApi.
        If map_type is 'search', query is required.
        """
        params = {
            "engine": "google_maps",
            "hl": self.hl,
            "gl": self.gl,  # Standard country code for Google Maps
        }

        if map_type == "place":
            if not data_id:
                return {
                    "error": "data_id is required for map_type 'place' in SerpApiProvider."
                }
            params["type"] = "place"
            params["data_id"] = data_id
            # For 'place' type, 'q' and 'll' are not primary, data_id is key
        elif map_type == "search":
            if not query:
                return {
                    "error": "query is required for map_type 'search' in SerpApiProvider."
                }
            params["type"] = "search"
            params["q"] = query
            if ll:
                params["ll"] = ll  # Format like "@40.7455096,-74.0083012,14z"
        else:
            return {"error": f"Unsupported map_type: {map_type}"}

        result_data = await self._make_request(params)

        if (
            "error" in result_data
            and not isinstance(
                result_data.get("search_metadata"),
                dict,  # serpapi returns error this way
            )
            and not result_data.get("place_results")
        ):  # For place type, error might not have search_metadata
            return result_data

        serpapi_error_status = (
            result_data.get("search_metadata", {}).get("status") == "Error"
        )
        serpapi_error_message = isinstance(result_data.get("error"), str)

        # Place results have a different structure and error reporting sometimes
        if (
            map_type == "place"
            and not result_data.get("place_results")
            and not serpapi_error_message
        ):
            # If it's a place search and no place_results, and no explicit error, it's likely an issue.
            if (
                not serpapi_error_status and not serpapi_error_message
            ):  # If no explicit error, craft one
                return {
                    "error": "SerpApi returned no place_results for the given data_id.",
                    "details": result_data,
                }

        if serpapi_error_status or serpapi_error_message:
            return {
                "error": result_data.get(
                    "error", "SerpApi returned an error for Google Maps search."
                ),
                "details": result_data,
            }

        if map_type == "place":
            # For place details, the main result is under "place_results" which is a dict
            # It might also have "reviews", "photos" etc.
            # We return the whole "place_results" dictionary after cleaning.
            if "place_results" in result_data:
                # Type hint checker might complain, but it's a Dict for 'place'
                return _remove_base64_images(result_data["place_results"])  # type: ignore
            else:  # Should have been caught above, but as a safeguard
                return {
                    "error": "No place_results found for place search.",
                    "details": result_data,
                }

        # map_type == "search"
        maps_results_transformed = []
        try:
            # For "search" type, results are typically in "local_results"
            serp_maps_results = result_data.get("local_results", [])
            for item in serp_maps_results[:num_results]:
                # Extract common fields, structure varies greatly
                place_data = {
                    "position": item.get("position"),
                    "title": item.get("title"),
                    "place_id": item.get("place_id"),
                    "data_id": item.get("data_id"),
                    "gps_coordinates": item.get("gps_coordinates"),
                    "address": item.get("address"),
                    "phone": item.get("phone"),
                    "website": item.get("website"),
                    "rating": item.get("rating"),
                    "reviews": item.get("reviews"),
                    "price": item.get("price"),
                    "type": item.get("type")
                    or item.get("types"),  # sometimes "type", sometimes "types"
                    "description": item.get("description"),
                    "service_options": item.get("service_options"),
                    "operating_hours": item.get("operating_hours"),
                    "thumbnail": item.get("thumbnail"),
                }
                maps_results_transformed.append(
                    {k: v for k, v in place_data.items() if v is not None}
                )
        except Exception as e:
            return {
                "error": f"Error processing SerpApi map search results: {str(e)}",
                "details": result_data,
            }
        return _remove_base64_images(
            maps_results_transformed
        )  # This will be a List[Dict]

    async def search_shopping(
        self,
        query: str,
        country: Optional[str] = None,
        domain: Optional[str] = None,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        condition: Optional[str] = None,
        sort_by: Optional[str] = None,
        num_results: int = 10,
    ) -> Union[List[Dict[str, Any]], ErrorResponse]:
        """
        Searches Google Shopping using SerpApi.
        Ref: https://serpapi.com/google-shopping-api
        Ref for tbs: https://serpapi.com/google-tbs-api (though shopping has specific tbs patterns)
        """
        params = {
            "engine": "google_shopping",
            "q": query,
            "hl": self.hl,
            # num_results is handled by slicing the response later
        }
        if country:
            params["gl"] = country
        if domain:
            params["google_domain"] = domain

        tbs_parts = []
        # For price and condition, tbs often starts with "mr:1,price:1" or similar
        # However, SerpApi examples for Shopping use more direct tbs components for these.

        if min_price is not None or max_price is not None or condition or sort_by:
            # Base for many shopping filters, though individual components are often enough
            # tbs_parts.append("mr:1") # Let's try without mr:1 first, some filters apply directly
            pass  # placeholder

        if min_price is not None:
            tbs_parts.append(f"ppr_min:{min_price}")
            if "price:1" not in tbs_parts:
                tbs_parts.insert(0, "price:1")  # Ensure price:1 is included for ppr_*
            if "mr:1" not in tbs_parts:
                tbs_parts.insert(0, "mr:1")

        if max_price is not None:
            tbs_parts.append(f"ppr_max:{max_price}")
            if "price:1" not in tbs_parts:
                tbs_parts.insert(0, "price:1")
            if "mr:1" not in tbs_parts:
                tbs_parts.insert(0, "mr:1")

        if condition:
            condition_map = {
                "new": "c",
                "used": "u",
                "refurbished": "r",
            }  # Example mapping
            # SerpApi docs also mention tbs=condition:new directly
            # Let's use the direct approach first if it aligns with their tbs builder. Example: &tbs=condition:new
            # For now, I will use condition directly, as SerpApi might handle it.
            # Or, more robustly for tbs: tbs_parts.append(f"condition:{condition_map.get(condition.lower())}")
            # Based on https://serpapi.com/google-shopping-filters, &tbs=p_cond:new or &tbs=p_cond:used
            # Let's go with p_cond for now as it seems more specific for shopping products.
            if condition.lower() == "new":
                tbs_parts.append("p_cond:new")
            elif condition.lower() == "used":
                tbs_parts.append("p_cond:used")
            elif (
                condition.lower() == "refurbished"
            ):  # SerpApi might not have a direct p_cond for refurbished, often grouped with used
                tbs_parts.append(
                    "p_cond:used"
                )  # Or could try general condition:refurbished if supported

        if (
            sort_by
        ):  # Common values: "r" (relevance), "p" (price asc), "pd" (price desc)
            tbs_parts.append(f"sort:{sort_by}")

        if tbs_parts:
            params["tbs"] = ",".join(tbs_parts)

        result_data = await self._make_request(params)

        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data

        serpapi_error_status = (
            result_data.get("search_metadata", {}).get("status") == "Error"
        )
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            return {
                "error": result_data.get(
                    "error", "SerpApi returned an error for Google Shopping search."
                ),
                "details": result_data,
            }

        shopping_results_transformed = []
        try:
            # Results can be in "shopping_results", "related_shopping_results", or "featured_shopping_results"
            # We will prioritize "shopping_results"
            serp_shopping_items = result_data.get("shopping_results", [])
            if not serp_shopping_items:
                serp_shopping_items = result_data.get(
                    "inline_results", []
                )  # New layout uses this

            for item in serp_shopping_items[:num_results]:
                product = {
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "product_id": item.get("product_id"),
                    "source": item.get("source"),
                    "price": item.get("price"),
                    "extracted_price": item.get("extracted_price"),
                    "rating": item.get("rating"),
                    "reviews": item.get("reviews"),
                    "snippet": item.get("snippet"),
                    "thumbnail": item.get("thumbnail"),
                    "delivery": item.get("delivery"),
                    "tag": item.get("tag"),
                    "store_rating": item.get("store_rating"),
                    "store_reviews": item.get("store_reviews"),
                    "second_hand_condition": item.get("second_hand_condition"),
                    "extensions": item.get("extensions"),
                    # Add more fields as needed based on API doc: old_price, installment, etc.
                }
                shopping_results_transformed.append(
                    {k: v for k, v in product.items() if v is not None}
                )

        except Exception as e:
            return {
                "error": f"Error processing SerpApi shopping results: {str(e)}",
                "details": result_data,
            }
        return _remove_base64_images(shopping_results_transformed)

    async def close(self) -> None:
        await self._client.aclose()


# Helper function to get a configured SERP provider
def _get_configured_serp_provider() -> (
    tuple[Optional[SerpProvider], Optional[ErrorResponse]]
):
    """
    Selects and returns a configured SERP provider based on environment variables.
    Prioritizes SerpApi, then DataForSEO.

    Returns:
        A tuple containing an instance of a SerpProvider and None if successful,
        or (None, ErrorResponse) if no provider is configured or if a provider
        had an issue during its own basic configuration check (e.g. SerpApiProvider API key check).
    """
    d4seo_login = Environment.get("DATA_FOR_SEO_LOGIN")
    d4seo_password = Environment.get("DATA_FOR_SEO_PASSWORD")
    serpapi_key = Environment.get("SERPAPI_API_KEY")

    print(f"d4seo_login: {d4seo_login}")
    print(f"d4seo_password: {d4seo_password}")
    print(f"serpapi_key: {serpapi_key}")

    if serpapi_key:
        return SerpApiProvider(), None
    elif d4seo_login and d4seo_password:
        return DataForSEOProvider(), None
    else:
        return None, {
            "error": "No SERP provider is configured. Please set credentials for DataForSEO (DATA_FOR_SEO_LOGIN, DATA_FOR_SEO_PASSWORD) or SerpApi (SERPAPI_API_KEY)."
        }


if __name__ == "__main__":
    import asyncio

    # Ensure ProcessingContext is imported from the correct path if it's not already at the top level of the file
    # from nodetool.workflows.processing_context import ProcessingContext

    async def run_all_examples():
        context = ProcessingContext()  # Mock context

        # Example using GoogleSearchTool (auto-provider selection)
        print("\n--- Testing GoogleSearchTool (auto-provider selection) ---")
        search_tool = GoogleSearchTool()
        search_params = {
            "keyword": "latest news on space exploration",
            "num_results": 10,
        }
        print(f"Tool: {search_tool.name}, Params: {search_params}")
        start_time = time.perf_counter()
        search_result = await search_tool.process(context, search_params)
        end_time = time.perf_counter()
        print("GoogleSearchTool Result:", json.dumps(search_result, indent=2))
        print(f"GoogleSearchTool took {end_time - start_time:.4f} seconds")

        # Example using GoogleNewsTool (auto-provider selection)
        print("\n--- Testing GoogleNewsTool (auto-provider selection) ---")
        news_tool = GoogleNewsTool()
        news_params = {"keyword": "AI in healthcare", "num_results": 10}
        print(f"Tool: {news_tool.name}, Params: {news_params}")
        start_time = time.perf_counter()
        news_result = await news_tool.process(context, news_params)
        end_time = time.perf_counter()
        print("GoogleNewsTool Result:", json.dumps(news_result, indent=2))
        print(f"GoogleNewsTool took {end_time - start_time:.4f} seconds")

        # Example using GoogleImagesTool (auto-provider selection)
        print("\n--- Testing GoogleImagesTool (auto-provider selection) ---")
        images_tool = GoogleImagesTool()
        images_params_keyword = {"keyword": "aurora borealis", "num_results": 10}
        print(f"Tool: {images_tool.name}, Params: {images_params_keyword}")
        start_time = time.perf_counter()
        images_result_keyword = await images_tool.process(
            context, images_params_keyword
        )
        end_time = time.perf_counter()
        print("GoogleImagesTool Result:", json.dumps(images_result_keyword, indent=2))
        print(f"GoogleImagesTool (keyword) took {end_time - start_time:.4f} seconds")

        # Example using GoogleFinanceTool (auto-provider selection)
        print("\n--- Testing GoogleFinanceTool (auto-provider selection) ---")
        finance_tool = GoogleFinanceTool()
        finance_params = {"query": "GOOGL:NASDAQ", "window": "1M"}
        print(f"Tool: {finance_tool.name}, Params: {finance_params}")
        start_time = time.perf_counter()
        finance_result = await finance_tool.process(context, finance_params)
        end_time = time.perf_counter()
        print("GoogleFinanceTool Result:", json.dumps(finance_result, indent=2))
        print(f"GoogleFinanceTool took {end_time - start_time:.4f} seconds")

        # Example using GoogleJobsTool (auto-provider selection)
        print("\n--- Testing GoogleJobsTool (auto-provider selection) ---")
        jobs_tool = GoogleJobsTool()
        jobs_params = {"query": "barista", "location": "New York, NY", "num_results": 3}
        print(f"Tool: {jobs_tool.name}, Params: {jobs_params}")
        start_time = time.perf_counter()
        jobs_result = await jobs_tool.process(context, jobs_params)
        end_time = time.perf_counter()
        print("GoogleJobsTool Result:", json.dumps(jobs_result, indent=2))
        print(f"GoogleJobsTool took {end_time - start_time:.4f} seconds")

        # Example using GoogleLensTool (auto-provider selection)
        print("\n--- Testing GoogleLensTool (auto-provider selection) ---")
        lens_tool = GoogleLensTool()
        # Use a publicly accessible image URL for testing
        lens_params = {
            "image_url": "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png",
            "num_results": 5,
        }
        print(f"Tool: {lens_tool.name}, Params: {lens_params}")
        start_time = time.perf_counter()
        lens_result = await lens_tool.process(context, lens_params)
        end_time = time.perf_counter()
        print("GoogleLensTool Result:", json.dumps(lens_result, indent=2))
        print(f"GoogleLensTool took {end_time - start_time:.4f} seconds")

        # Example using GoogleMapsTool (auto-provider selection) - Search
        print("\n--- Testing GoogleMapsTool (search) (auto-provider selection) ---")
        maps_tool = GoogleMapsTool()
        maps_search_params = {
            "query": "restaurants in San Francisco",
            "ll": "@37.7749,-122.4194,12z",  # Example lat,lng,zoom for SF
            "num_results": 3,
        }
        print(f"Tool: {maps_tool.name}, Params: {maps_search_params}")
        start_time = time.perf_counter()
        maps_search_result = await maps_tool.process(context, maps_search_params)
        end_time = time.perf_counter()
        print(
            "GoogleMapsTool (search) Result:", json.dumps(maps_search_result, indent=2)
        )
        print(f"GoogleMapsTool (search) took {end_time - start_time:.4f} seconds")

        # Example using GoogleMapsTool (auto-provider selection) - Place Details (using a known data_id if available)
        # This requires a valid data_id. If running this test, replace with a real one.
        # For now, this part of the test might show an error if the data_id is invalid or SerpApi key is not set.
        print(
            "\n--- Testing GoogleMapsTool (place details) (auto-provider selection) ---"
        )
        # Example data_id for "Googleplex". Replace if testing with a different place.
        maps_place_params = {
            "map_type": "place",
            "data_id": "0x808fcf68c2527669:0x877cb45Ac0435C98",
        }
        print(f"Tool: {maps_tool.name}, Params: {maps_place_params}")
        start_time = time.perf_counter()
        maps_place_result = await maps_tool.process(context, maps_place_params)
        end_time = time.perf_counter()
        print(
            "GoogleMapsTool (place details) Result:",
            json.dumps(maps_place_result, indent=2),
        )
        print(
            f"GoogleMapsTool (place details) took {end_time - start_time:.4f} seconds"
        )

        # Example using GoogleShoppingTool (auto-provider selection)
        print("\n--- Testing GoogleShoppingTool (auto-provider selection) ---")
        shopping_tool = GoogleShoppingTool()
        shopping_params = {
            "query": "laptop sleeve 13 inch",
            "country": "us",
            "min_price": 10,
            "max_price": 50,
            "condition": "new",
            "sort_by": "r",  # Sort by relevance
            "num_results": 5,
        }
        print(f"Tool: {shopping_tool.name}, Params: {shopping_params}")
        start_time = time.perf_counter()
        shopping_result = await shopping_tool.process(context, shopping_params)
        end_time = time.perf_counter()
        print("GoogleShoppingTool Result:", json.dumps(shopping_result, indent=2))
        print(f"GoogleShoppingTool took {end_time - start_time:.4f} seconds")

        # Optional: Direct provider tests if needed for debugging, keeping existing conditional logic

    # Run examples if any provider is configured or to show tool's error message
    asyncio.run(run_all_examples())
