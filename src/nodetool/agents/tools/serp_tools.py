import json
import time
from typing import Any, ClassVar, TypeVar

from nodetool.agents.serp_providers.apify_provider import ApifyProvider
from nodetool.agents.serp_providers.brave_search_provider import BraveSearchProvider
from nodetool.agents.serp_providers.data_for_seo_provider import DataForSEOProvider
from nodetool.agents.serp_providers.serp_api_provider import SerpApiProvider
from nodetool.agents.serp_providers.serp_providers import ErrorResponse, SerpProvider
from nodetool.agents.tools.base import Tool
from nodetool.config.environment import Environment
from nodetool.workflows.processing_context import ProcessingContext

T = TypeVar("T")


class GoogleSearchTool(Tool):
    name = "google_search"
    description = "Search Google to retrieve organic search results. Uses available SERP provider."
    input_schema: ClassVar[dict[str, Any]] = {
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

        provider_instance, error_response = await _get_configured_serp_provider(context)
        if error_response:
            return error_response
        if not provider_instance:
            return {"error": "Failed to initialize SERP provider."}

        async with provider_instance as provider:
            result_data = await provider.search(keyword=keyword, num_results=num_results)

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
    input_schema: ClassVar[dict[str, Any]] = {
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

        provider_instance, error_response = await _get_configured_serp_provider(context)
        if error_response:
            return error_response
        if not provider_instance:
            return {"error": "Failed to initialize SERP provider."}

        async with provider_instance as provider:
            result_data = await provider.search_news(keyword=keyword, num_results=num_results)

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
    input_schema: ClassVar[dict[str, Any]] = {
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

        provider_instance, error_response = await _get_configured_serp_provider(context)
        if error_response:
            return error_response
        if not provider_instance:
            return {"error": "Failed to initialize SERP provider."}

        async with provider_instance as provider:
            result_data = await provider.search_images(keyword=keyword, image_url=image_url, num_results=num_results)

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
    input_schema: ClassVar[dict[str, Any]] = {
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

        provider_instance, error_response = await _get_configured_serp_provider(context)
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
    input_schema: ClassVar[dict[str, Any]] = {
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

        provider_instance, error_response = await _get_configured_serp_provider(context)
        if error_response:
            return error_response
        if not provider_instance:
            return {"error": "Failed to initialize SERP provider."}

        async with provider_instance as provider:
            result_data = await provider.search_jobs(query=query, location=location, num_results=num_results)

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
    input_schema: ClassVar[dict[str, Any]] = {
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

        provider_instance, error_response = await _get_configured_serp_provider(context)
        if error_response:
            return error_response
        if not provider_instance:
            return {"error": "Failed to initialize SERP provider."}

        async with provider_instance as provider:
            result_data = await provider.search_lens(image_url=image_url, num_results=num_results)

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
    input_schema: ClassVar[dict[str, Any]] = {
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

        provider_instance, error_response = await _get_configured_serp_provider(context)
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
        search_term = f"'{query}'" if query else f"'{params.get('query', 'places')}'"

        msg = f"Searching Google Maps for {search_term}..."
        if len(msg) > 80:
            msg = "Searching Google Maps..."
        return msg


class GoogleShoppingTool(Tool):
    name = "google_shopping"
    description = "Search Google Shopping for products. Uses available SERP provider."
    input_schema: ClassVar[dict[str, Any]] = {
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

        provider_instance, error_response = await _get_configured_serp_provider(context)
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


# Helper function to get a configured SERP provider
async def _get_configured_serp_provider(
    context: ProcessingContext,
) -> tuple[SerpProvider | None, ErrorResponse | None]:
    """
    Selects and returns a configured SERP provider based on environment variables.
    If SERP_PROVIDER setting is defined, uses that specific provider.
    Otherwise, auto-selects based on available API keys (SerpApi > Brave Search > DataForSEO).

    Returns:
        A tuple containing an instance of a SerpProvider and None if successful,
        or (None, ErrorResponse) if no provider is configured or if a provider
        had an issue during its own basic configuration check (e.g. SerpApiProvider API key check).
    """
    # Check if a specific provider is requested via SERP_PROVIDER setting
    serp_provider = Environment.get("SERP_PROVIDER")

    serpapi_key = await context.get_secret("SERPAPI_API_KEY")
    apify_key = await context.get_secret("APIFY_API_KEY")
    brave_api_key = await context.get_secret("BRAVE_API_KEY")
    d4seo_login = await context.get_secret("DATA_FOR_SEO_LOGIN")
    d4seo_password = await context.get_secret("DATA_FOR_SEO_PASSWORD")

    # If SERP_PROVIDER setting is defined, use the specified provider
    if serp_provider:
        serp_provider = serp_provider.lower()
        if serp_provider == "serpapi":
            if serpapi_key:
                return SerpApiProvider(api_key=serpapi_key), None
            else:
                return None, {"error": "SERP_PROVIDER is set to 'serpapi' but SERPAPI_API_KEY is not configured."}
        elif serp_provider == "apify":
            if apify_key:
                return ApifyProvider(api_key=apify_key), None
            else:
                return None, {"error": "SERP_PROVIDER is set to 'apify' but APIFY_API_KEY is not configured."}
        elif serp_provider == "brave":
            if brave_api_key:
                return BraveSearchProvider(api_key=brave_api_key), None
            else:
                return None, {"error": "SERP_PROVIDER is set to 'brave' but BRAVE_API_KEY is not configured."}
        elif serp_provider == "dataforseo":
            if d4seo_login and d4seo_password:
                return DataForSEOProvider(api_login=d4seo_login, api_password=d4seo_password), None
            else:
                return None, {
                    "error": "SERP_PROVIDER is set to 'dataforseo' but DATA_FOR_SEO_LOGIN and/or DATA_FOR_SEO_PASSWORD are not configured."
                }
        else:
            return None, {
                "error": f"Invalid SERP_PROVIDER value '{serp_provider}'. Valid options are: 'serpapi', 'apify', 'brave', 'dataforseo'."
            }

    # Auto-select based on available API keys (SerpApi > Brave Search > DataForSEO)
    if serpapi_key:
        return SerpApiProvider(api_key=serpapi_key), None
    elif brave_api_key:
        return BraveSearchProvider(api_key=brave_api_key), None
    elif d4seo_login and d4seo_password:
        return DataForSEOProvider(api_login=d4seo_login, api_password=d4seo_password), None
    else:
        return None, {
            "error": "No SERP provider is configured. Please set credentials for SerpApi, Brave Search, or DataForSEO."
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
        images_result_keyword = await images_tool.process(context, images_params_keyword)
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
        print("GoogleMapsTool (search) Result:", json.dumps(maps_search_result, indent=2))
        print(f"GoogleMapsTool (search) took {end_time - start_time:.4f} seconds")

        # Example using GoogleMapsTool (auto-provider selection) - Place Details (using a known data_id if available)
        # This requires a valid data_id. If running this test, replace with a real one.
        # For now, this part of the test might show an error if the data_id is invalid or SerpApi key is not set.
        print("\n--- Testing GoogleMapsTool (place details) (auto-provider selection) ---")
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
        print(f"GoogleMapsTool (place details) took {end_time - start_time:.4f} seconds")

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
