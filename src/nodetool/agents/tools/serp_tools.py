import json
import os
from typing import Any

from nodetool.agents.tools.base import Tool
from nodetool.workflows.processing_context import ProcessingContext
from httpx import AsyncClient, HTTPStatusError, RequestError
from nodetool.common.environment import Environment
import base64


def _remove_base64_images(data):
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


async def _make_dataforseo_request(api_url: str, payload: list[dict]) -> dict:
    """
    Makes an asynchronous POST request to the DataForSEO API.

    Args:
        api_url: The specific DataForSEO API endpoint URL.
        payload: The payload for the request (list containing one dictionary).

    Returns:
        A dictionary containing the parsed JSON response or an error dictionary.
    """
    try:
        headers = _get_dataforseo_auth_headers()
    except ValueError as e:
        return {"error": str(e)}

    try:
        async with AsyncClient(timeout=60.0) as client:
            response = await client.post(api_url, headers=headers, json=payload)
            response.raise_for_status()  # Raise HTTPStatusError for bad responses (4xx or 5xx)
        return response.json()
    except HTTPStatusError as e:
        try:
            error_body = e.response.json()
        except json.JSONDecodeError:
            error_body = e.response.text
        return {
            "error": f"HTTP error occurred: {e.response.status_code} - {e.response.reason_phrase}",
            "details": error_body,
        }
    except RequestError as e:
        return {"error": f"HTTP request failed: {str(e)}"}
    except json.JSONDecodeError as e:
        return {"error": f"Failed to decode JSON response: {str(e)}"}
    except Exception as e:
        # Catch any other unexpected errors during the request process
        return {"error": f"Unexpected error during DataForSEO request: {str(e)}"}


def _get_dataforseo_auth_headers():
    """
    Retrieves DataForSEO credentials from environment and returns auth headers.
    Raises ValueError if credentials are not found.
    """
    login = Environment.get("DATA_FOR_SEO_LOGIN")
    password = Environment.get("DATA_FOR_SEO_PASSWORD")

    if not login or not password:
        raise ValueError(
            "DataForSEO credentials (DATA_FOR_SEO_LOGIN, DATA_FOR_SEO_PASSWORD) not found in environment variables."
        )

    credentials = f"{login}:{password}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    return {
        "Authorization": f"Basic {encoded_credentials}",
        "Content-Type": "application/json",
    }


class GoogleSearchTool(Tool):
    """
    A tool that allows searching Google using DataForSEO's API.

    This tool enables language models to perform Google searches and get
    the search results using the DataForSEO service.
    """

    name = "google_search"
    description = "Search Google using DataForSEO's API to retrieve search results"
    input_schema = {
        "type": "object",
        "properties": {
            "keyword": {
                "type": "string",
                "description": "The keyword to search for.",
            },
            "location_code": {
                "type": "integer",
                "description": "The location code for the search, for example: 2826 for London",
            },
            "language_code": {
                "type": "string",
                "description": "The language code for the search (e.g., 'en').",
                "default": "en",
            },
            "device": {
                "type": "string",
                "description": "The device type for the search.",
                "enum": ["desktop", "mobile", "tablet"],
                "default": "desktop",
            },
            "os": {
                "type": "string",
                "description": "The operating system for the search.",
                "enum": ["windows", "macos", "ios", "android", "linux"],
                "default": "windows",
            },
            "depth": {
                "type": "integer",
                "description": "The number of search results to retrieve.",
                "default": 20,
            },
            "output_file": {
                "type": "string",
                "description": "Optional path to save the search results (relative to workspace). If not provided, results are returned directly.",
            },
        },
        "required": ["keyword"],
    }
    example = """
    dataforseo_search(
        keyword="weather forecast",
        location_code=2826,
        language_code="en",
        depth=10
    )
    """

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        """
        Execute a Google search using DataForSEO's API.

        Args:
            context: The processing context.
            params: Dictionary including keyword and optional parameters.

        Returns:
            dict: Search results or error message.
        """
        keyword = params.get("keyword")
        if not keyword:
            return {"error": "Keyword is required"}

        # Construct payload
        payload_dict = {
            "keyword": keyword,
            "location_code": params.get("location_code", 2840),
            "language_code": params.get("language_code", "en"),
            "device": params.get("device", "desktop"),
            "os": params.get("os", "windows"),
            "depth": params.get("depth", 20),
        }
        payload_dict = {k: v for k, v in payload_dict.items() if v is not None}
        payload = [payload_dict]

        url = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"

        try:
            result_data = await _make_dataforseo_request(url, payload)
            if "error" in result_data:
                return result_data  # Propagate error from request function

            # Check DataForSEO API status
            if (
                result_data.get("status_code") != 20000
                or result_data.get("status_message") != "Ok."
            ):
                return {
                    "error": f"DataForSEO API Error: {result_data.get('status_code')} - {result_data.get('status_message')}",
                    "details": result_data,
                }

            output_file = params.get("output_file")

            # Extract organic results regardless of output_file presence
            organic_results = []
            task_result = (
                result_data.get("tasks", [{}])[0].get("result")
                if result_data.get("tasks")
                else None
            )
            if task_result and isinstance(task_result, list) and len(task_result) > 0:
                items = task_result[0].get("items", [])
                if items:
                    organic_results = [
                        item for item in items if item.get("type") == "organic"
                    ]

            if output_file:
                full_path = context.resolve_workspace_path(output_file)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as f:
                    # Save only the organic results to the file
                    json.dump(organic_results, f, indent=2)
                return {"success": True, "output_file": full_path}
            else:
                # Return the extracted organic results directly
                return {"success": True, "results": organic_results}

        except Exception as e:
            return {"error": f"Error processing DataForSEO request: {str(e)}"}

    def user_message(self, params: dict) -> str:
        keyword = params.get("keyword", "something")
        msg = f"Searching Google for '{keyword}'..."
        if len(msg) > 80:
            msg = "Searching Google..."
        return msg


class GoogleNewsTool(Tool):
    """
    A tool that fetches live Google News results using DataForSEO's API.

    This tool enables language models to perform Google News searches for specific keywords,
    locations, and languages, retrieving real-time news articles.
    """

    name = "google_news"
    description = (
        "Search Google News using DataForSEO's API to retrieve live news results"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "keyword": {
                "type": "string",
                "description": "The keyword to search for in Google News.",
            },
            "location_name": {
                "type": "string",
                "description": "The location name for the search (e.g., 'London,England,United Kingdom'). Required if location_code is not set.",
            },
            "location_code": {
                "type": "integer",
                "description": "The location code for the search (e.g., 2826 for London). Required if location_name is not set.",
            },
            "language_name": {
                "type": "string",
                "description": "The full name of the language for the search (e.g., 'English'). Required if language_code is not set.",
            },
            "language_code": {
                "type": "string",
                "description": "The language code for the search (e.g., 'en'). Required if language_name is not set.",
            },
            "date_from": {
                "type": "string",
                "description": "Start date for the search in YYYY-MM-DD format.",
            },
            "date_to": {
                "type": "string",
                "description": "End date for the search in YYYY-MM-DD format.",
            },
            "sort_by": {
                "type": "string",
                "description": "Sort order for the results.",
                "enum": ["relevance", "date"],
                "default": "relevance",
            },
            "output_file": {
                "type": "string",
                "description": "Optional path to save the search results (relative to workspace). If not provided, results are returned directly.",
            },
        },
        "required": ["keyword"],
    }
    example = """
    dataforseo_google_news_live(
        keyword="artificial intelligence",
        location_name="London,England,United Kingdom",
        language_name="English",
        sort_by="date"
    )
    """

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        """
        Execute a Google News live search using DataForSEO's API.

        Args:
            context: The processing context.
            params: Dictionary including keyword and optional parameters.

        Returns:
            dict: Search results or error message.
        """
        keyword = params.get("keyword")
        if not keyword:
            return {"error": "Keyword is required"}
        if not params.get("location_name") and not params.get("location_code"):
            return {"error": "Either location_name or location_code is required"}
        if not params.get("language_name") and not params.get("language_code"):
            return {"error": "Either language_name or language_code is required"}

        # Construct payload
        payload_dict = {
            "keyword": keyword,
            "location_name": params.get("location_name"),
            "location_code": params.get("location_code"),
            "language_name": params.get("language_name"),
            "language_code": params.get("language_code"),
            "date_from": params.get("date_from"),
            "date_to": params.get("date_to"),
            "sort_by": params.get("sort_by", "relevance"),
            # Add other potential parameters from the API doc if needed later
        }
        payload_dict = {k: v for k, v in payload_dict.items() if v is not None}
        payload = [payload_dict]

        url = "https://api.dataforseo.com/v3/serp/google/news/live/advanced"

        try:
            result_data = await _make_dataforseo_request(url, payload)
            if "error" in result_data:
                return result_data  # Propagate error from request function

            # Check DataForSEO API status
            if (
                result_data.get("status_code") != 20000
                or result_data.get("status_message") != "Ok."
            ):
                return {
                    "error": f"DataForSEO API Error: {result_data.get('status_code')} - {result_data.get('status_message')}",
                    "details": result_data,
                }

            output_file = params.get("output_file")

            # Extract news items
            news_items = []
            task_result = (
                result_data.get("tasks", [{}])[0].get("result")
                if result_data.get("tasks")
                else None
            )
            if task_result and isinstance(task_result, list) and len(task_result) > 0:
                items = task_result[0].get("items", [])
                if items:
                    # Filter for 'news_search' or 'top_stories' types as per API doc
                    news_items = [
                        item
                        for item in items
                        if item.get("type") in ["news_search", "top_stories"]
                    ]

            if output_file:
                full_path = context.resolve_workspace_path(output_file)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as f:
                    # Save the extracted news items to the file
                    json.dump(news_items, f, indent=2)
                return {"success": True, "output_file": full_path}
            else:
                # Return the extracted news items directly
                return {"success": True, "results": news_items}

        except Exception as e:
            return {"error": f"Error processing DataForSEO request: {str(e)}"}

    def user_message(self, params: dict) -> str:
        keyword = params.get("keyword", "something")
        msg = f"Searching Google News for '{keyword}'..."
        if len(msg) > 80:
            msg = "Searching Google News..."
        return msg


class GoogleImagesTool(Tool):
    """
    A tool that fetches live Google Images results using DataForSEO's API.

    This tool enables language models to perform Google Images searches for specific keywords,
    locations, and languages, retrieving real-time image results. It can also optionally
    search using an image URL or base64 encoded image content instead of a keyword.
    """

    name = "google_images"
    description = (
        "Search Google Images using DataForSEO's API to retrieve live image results"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "keyword": {
                "type": "string",
                "description": "The keyword to search for in Google Images. Required if image_url or image_content is not provided.",
            },
            "image_url": {
                "type": "string",
                "description": "URL of an image to use for the search. Required if keyword or image_content is not provided.",
            },
            "image_content": {
                "type": "string",
                "description": "Base64 encoded content of an image to use for the search. Required if keyword or image_url is not provided.",
            },
            "location_name": {
                "type": "string",
                "description": "The location name for the search (e.g., 'London,England,United Kingdom').",
            },
            "language_code": {
                "type": "string",
                "description": "The language code for the search (e.g., 'en').",
            },
            "depth": {
                "type": "integer",
                "description": "The number of search results to retrieve.",
                "default": 20,
            },
            "search_param": {
                "type": "string",
                "description": "Additional Google Images search parameters (e.g., 'tbs=isz:l' for large images). Refer to Google documentation for options.",
            },
            "output_file": {
                "type": "string",
                "description": "Optional path to save the search results (relative to workspace). If not provided, results are returned directly.",
            },
        },
        # Make one of keyword, image_url, image_content required implicitly by validation logic
    }
    example = """
    dataforseo_google_images_live(
        keyword="cats",
        location_name="United States",
        language_code="en",
        depth=50
    )
    """

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        """
        Execute a Google Images live search using DataForSEO's API.

        Args:
            context: The processing context.
            params: Dictionary including keyword/image_url/image_content and optional parameters.

        Returns:
            dict: Search results or error message.
        """
        keyword = params.get("keyword")
        image_url = params.get("image_url")
        image_content = params.get("image_content")

        if not keyword and not image_url and not image_content:
            return {
                "error": "One of 'keyword', 'image_url', or 'image_content' is required"
            }

        # Construct payload
        payload_dict = {
            "keyword": keyword,
            "image_url": image_url,
            "image_content": image_content,
            "location_name": params.get("location_name"),
            "language_code": params.get("language_code"),
            "depth": params.get("depth", 20),
            "search_param": params.get("search_param"),
            # Add other potential parameters from the API doc if needed later
        }
        payload_dict = {k: v for k, v in payload_dict.items() if v is not None}
        payload = [payload_dict]

        api_url = "https://api.dataforseo.com/v3/serp/google/images/live/advanced"

        try:
            result_data = await _make_dataforseo_request(api_url, payload)
            if "error" in result_data:
                return result_data  # Propagate error from request function

            # Check DataForSEO API status
            if (
                result_data.get("status_code") != 20000
                or result_data.get("status_message") != "Ok."
            ):
                return {
                    "error": f"DataForSEO API Error: {result_data.get('status_code')} - {result_data.get('status_message')}",
                    "details": result_data,
                }

            output_file = params.get("output_file")

            # Extract image items (type 'images_search' and potentially 'carousel_element')
            image_items = []
            task_result = (
                result_data.get("tasks", [{}])[0].get("result")
                if result_data.get("tasks")
                else None
            )
            if task_result and isinstance(task_result, list) and len(task_result) > 0:
                items = task_result[0].get("items", [])
                if items:
                    for item in items:
                        if item.get("type") == "images_search":
                            image_items.append(item)
                        elif item.get("type") == "carousel" and item.get("items"):
                            # Extract images from carousels as well
                            for carousel_item in item["items"]:
                                if carousel_item.get(
                                    "type"
                                ) == "carousel_element" and carousel_item.get(
                                    "image_url"
                                ):
                                    # Adapt carousel item structure slightly to match image_search structure for consistency
                                    image_items.append(
                                        {
                                            "type": "carousel_image",  # Indicate source
                                            "title": carousel_item.get("title"),
                                            "source_url": carousel_item.get(
                                                "image_url"
                                            ),  # Use image_url as source_url
                                            "url": None,  # Carousel elements don't have a direct page URL
                                            "alt": carousel_item.get(
                                                "title"
                                            ),  # Use title as alt text
                                        }
                                    )

            if output_file:
                full_path = context.resolve_workspace_path(output_file)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as f:
                    # Save the extracted image items to the file
                    json.dump(image_items, f, indent=2)
                return {"success": True, "output_file": full_path}
            else:
                # Return the extracted image items directly
                return {"success": True, "results": image_items}

        except Exception as e:
            return {"error": f"Error processing DataForSEO request: {str(e)}"}

    def user_message(self, params: dict) -> str:
        keyword = params.get("keyword")
        if keyword:
            search_term = f" '{keyword}'"
        elif params.get("image_url"):
            search_term = " an image URL"
        elif params.get("image_content"):
            search_term = " image content"
        else:
            search_term = " something"
        msg = f"Searching Google Images for{search_term}..."
        if len(msg) > 80:
            msg = "Searching Google Images..."
        return msg


if __name__ == "__main__":
    import asyncio

    # Import the actual ProcessingContext
    from nodetool.workflows.processing_context import ProcessingContext

    async def run_google_search_example():
        # Instantiate the tool
        search_tool = GoogleSearchTool()

        # Create a mock context
        context = ProcessingContext()

        # Define search parameters (Requires DATA_FOR_SEO credentials in env)
        params = {
            "keyword": "What is AI?",
            "location_name": "US",
            "language_code": "en",
            "depth": 10,  # Limit results for example
            # "output_file": "ai_search_results.json" # Optional: save to file
        }

        print(f"Running Google Search for: '{params['keyword']}'")
        result = await search_tool.process(context, params)

    async def run_google_news_live_example():
        # Instantiate the tool
        news_tool = GoogleNewsTool()

        # Create a mock context
        context = ProcessingContext()

        # Define search parameters (Requires DATA_FOR_SEO credentials in env)
        params = {
            "keyword": "Tesla stock",
            "location_code": 2840,
            "language_code": "en",
            "sort_by": "date",
            # "output_file": "tesla_news_live.json" # Optional: save to file
        }

        print(f"Running Google News Live Search for: '{params['keyword']}'")
        result = await news_tool.process(context, params)

    async def run_google_images_example():
        # Instantiate the tool
        images_tool = GoogleImagesTool()

        # Create a mock context
        context = ProcessingContext()

        # Define search parameters (Requires DATA_FOR_SEO credentials in env)
        params = {
            "keyword": "Shiba Inu",
            "location_name": "United States",  # Use name or code
            "language_code": "en",  # Use name or code
            "depth": 10,
        }

        print(f"Running Google Images Search for: '{params['keyword']}'")
        result = await images_tool.process(context, params)

    # Run the async function
    # asyncio.run(run_google_search_example())
    # asyncio.run(run_google_news_live_example())
    asyncio.run(run_google_images_example())
