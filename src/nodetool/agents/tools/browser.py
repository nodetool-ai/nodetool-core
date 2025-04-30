"""
Browser interaction tools.

This module provides tools for interacting with web browsers and web pages.
"""

import os
from typing import Any
import aiohttp
import urllib.parse
import html2text
from bs4 import BeautifulSoup

from nodetool.common.environment import Environment
from nodetool.workflows.processing_context import ProcessingContext
from .base import Tool
from playwright.async_api import Page

import os
from typing import Any
import asyncio
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from browser_use import Agent, Browser, BrowserConfig

from nodetool.common.environment import Environment
from nodetool.workflows.processing_context import ProcessingContext
from .base import Tool


async def extract_metadata(page: Page):
    """
    Extract both Open Graph and standard metadata from a webpage using Playwright.

    Args:
        page: Playwright page object

    Returns:
        dict: Dictionary containing both Open Graph and standard metadata
    """
    # Create a dictionary to store the metadata
    metadata = {
        "og": {},  # For Open Graph metadata
        "standard": {},  # For standard metadata
    }

    # List of Open Graph properties to extract
    og_properties = [
        "og:locale",
        "og:type",
        "og:title",
        "og:description",
        "og:url",
        "og:site_name",
        "og:image",
        "og:image:width",
        "og:image:height",
        "og:image:type",
    ]

    # List of standard meta properties to extract
    standard_properties = [
        "description",
        "keywords",
        "author",
        "viewport",
        "robots",
        "canonical",
        "generator",
    ]

    # Extract Open Graph metadata
    for prop in og_properties:
        # Use locator to find the meta tag with the specific property
        locator = page.locator(f'meta[property="{prop}"]')

        # Check if the element exists
        if await locator.count() > 0:
            # Extract the content attribute
            content = await locator.first.get_attribute("content")
            # Store in dictionary (remove 'og:' prefix for cleaner keys)
            metadata["og"][prop.replace("og:", "")] = content

    # Extract standard metadata
    for prop in standard_properties:
        # Use locator to find the meta tag with the specific name
        locator = page.locator(f'meta[name="{prop}"]')

        # Check if the element exists
        if await locator.count() > 0:
            # Extract the content attribute
            content = await locator.first.get_attribute("content")
            # Store in dictionary
            metadata["standard"][prop] = content

    # Also get title from the title tag
    title_locator = page.locator("title")
    if await title_locator.count() > 0:
        metadata["standard"]["title"] = await title_locator.first.inner_text()

    return metadata


class BrowserTool(Tool):
    """
    A tool that allows fetching web content.

    This tool enables language models to retrieve content from web pages by
    navigating to URLs and extracting text and metadata.
    """

    name = "browser"
    description = "Fetch content from a web page"
    input_schema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to navigate to",
            },
            "selector": {
                "type": "string",
                "description": "Optional CSS selector to extract specific elements. If provided, only matching elements will be returned.",
            },
            "headless": {
                "type": "boolean",
                "description": "Run the browser in headless mode",
                "default": True,
            },
            "use_remote_browser": {
                "type": "boolean",
                "description": "Use a remote browser instead of a local one",
                "default": False,
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in milliseconds for page navigation",
                "default": 20000,
            },
            "output_file": {
                "type": "string",
                "description": "Path to save the extracted content (relative to workspace)",
            },
        },
        "required": ["url"],
    }
    example = """
    browser(
        url="https://www.google.com",
        output_file="google.txt"
    )
    """

    def __init__(self, use_readability: bool = True):
        self.use_readability = use_readability

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        """
        Fetches content from a web page using Playwright.

        Args:
            context: The processing context
            params: Dictionary including:
                url (str): URL to navigate to
                timeout (int, optional): Timeout in milliseconds for page navigation
                output_file (str, optional): Path to save the extracted content (relative to workspace)

        Returns:
            dict: Result containing page content and metadata
        """
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise Exception(
                "Playwright is not installed. Please install it with 'pip install playwright' and then run 'playwright install'"
            )

        url = params.get("url")
        timeout = params.get("timeout", 30000)  # Default 30 seconds
        if not url:
            return {"error": "URL is required"}

        selector = params.get("selector")

        # Initialize browser
        playwright_instance = await async_playwright().start()

        if params.get("use_remote_browser"):
            browser_endpoint = Environment.get("BRIGHTDATA_SCRAPING_BROWSER_ENDPOINT")
            browser = await playwright_instance.chromium.connect_over_cdp(
                browser_endpoint
            )
            # Create context with additional permissions and settings
            browser_context = await browser.new_context(
                bypass_csp=True,
            )
        else:
            # Launch browser with similar settings for local usage
            browser = await playwright_instance.chromium.launch(
                headless=params.get("headless", True)
            )
            browser_context = await browser.new_context(
                bypass_csp=True,
            )

        # Create page from the context instead of directly from browser
        page = await browser_context.new_page()

        try:
            # Navigate to the URL with the specified timeout
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout)

            # Extract metadata from the page
            metadata = await extract_metadata(page)

            result = {
                "success": True,
                "url": url,
                "metadata": metadata,
            }

            content = None
            extracted_elements = []

            if selector:
                elements = await page.query_selector_all(selector)
                if not elements:
                    return {
                        "error": f"No elements found matching selector: {selector}",
                        "url": url,
                        "metadata": metadata,
                    }

                for element in elements:
                    extracted_elements.append(await element.text_content() or "")

                content = extracted_elements

            elif self.use_readability:
                await page.add_script_tag(
                    url="https://unpkg.com/@mozilla/readability/Readability.js"
                )
                readability_result = await page.evaluate(
                    """() => {
                    try {
                        const documentClone = document.cloneNode(true);
                        const reader = new Readability(documentClone);
                        const article = reader.parse();
                        return article || { error: 'Failed to parse article with Readability' };
                    } catch (e) {
                        return { error: 'Error executing Readability: ' + e.message };
                    }
                }"""
                )
                if (
                    isinstance(readability_result, dict)
                    and "content" in readability_result
                ):
                    content = html2text.html2text(readability_result["content"])
                elif (
                    isinstance(readability_result, dict)
                    and "error" in readability_result
                ):
                    print(
                        f"Readability error: {readability_result['error']}. Falling back to full page content."
                    )
                    content = html2text.html2text(await page.content())
                else:
                    content = html2text.html2text(await page.content())

            else:
                content = html2text.html2text(await page.content())

            # Handle output file if specified
            output_file = params.get("output_file")
            if output_file:
                full_path = context.resolve_workspace_path(output_file)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as f:
                    if isinstance(content, list):
                        f.write("\n".join(content))
                    else:
                        f.write(content or "")
                result["output_file"] = full_path
            else:
                result["content"] = content

            return result
        except Exception as e:
            print(e)
            return {"error": f"Error fetching page: {str(e)}"}

        finally:
            # Always close the browser session
            await browser.close()
            await playwright_instance.stop()


class ScreenshotTool(Tool):
    """
    A tool that allows taking screenshots of web pages or specific elements.

    This tool enables language models to capture visual representations of web pages
    or specific elements for analysis or documentation.
    """

    name = "take_screenshot"
    description = (
        "Take a screenshot of the current browser window or a specific element"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "selector": {
                "type": "string",
                "description": "Optional CSS selector for capturing a specific element",
            },
            "path": {
                "type": "string",
                "description": "Workspace relative path to save the screenshot",
                "default": "screenshot.png",
            },
        },
    }
    example = """
    take_screenshot(
        selector=".title",
        path="title.png"
    )
    """

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            page = context.get("playwright_page")
            if page is None:
                return {"error": "No browser session available"}

            path = params.get("path", "screenshot.png")
            full_path = context.resolve_workspace_path(path)
            if "selector" in params:
                element = await page.query_selector(params["selector"])
                if element:
                    await element.screenshot(path=full_path)
                else:
                    return {
                        "error": f"No element found matching selector: {params['selector']}"
                    }
            else:
                await page.screenshot(path=full_path)

            return {"success": True, "path": full_path}

        except Exception as e:
            return {"error": str(e)}


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


async def make_api_request(search_url: str):
    """Make an API request and handle common error cases."""
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://api.brightdata.com/request"
            api_key = get_required_api_key(
                "BRIGHTDATA_API_KEY",
                "Brightdata API key not found. Please provide it in the secrets as 'BRIGHTDATA_API_KEY'.",
            )
            if isinstance(api_key, dict) and "error" in api_key:
                return api_key

            zone = get_required_api_key(
                "BRIGHTDATA_SERP_ZONE",
                "Brightdata SERP zone not found. Please provide it in the secrets as 'BRIGHTDATA_SERP_ZONE'.",
            )
            if isinstance(zone, dict) and "error" in zone:
                return zone
            # Brightdata-specific request preparation
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            payload = {"zone": zone, "url": search_url, "format": "json"}

            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"API request failed with status {response.status}: {error_text}"
                    )

                return await response.json()
    except Exception as e:
        return {"error": f"Error making API request: {str(e)}"}


def get_required_api_key(key_name, error_message=None):
    """Get a required API key from environment variables."""
    api_key = Environment.get(key_name)
    if not api_key:
        print(
            f"Error: {error_message or f'{key_name} not found in environment variables.'}"
        )
        raise ValueError(
            error_message or f"{key_name} not found in environment variables."
        )
    return api_key


class GoogleSearchTool(Tool):
    """
    A tool that allows searching Google using Brightdata's API.

    This tool enables language models to perform Google searches and get
    the search results without directly interacting with a browser.
    """

    name = "google_search"
    description = "Search Google using Brightdata's API to retrieve search results"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to submit to Google",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (optional)",
                "default": 20,
            },
            "site": {
                "type": "string",
                "description": "Limit search results to a specific website (e.g., 'site:example.com')",
            },
            "filetype": {
                "type": "string",
                "description": "Limit search results to specific file types (e.g., 'pdf', 'doc', 'xls')",
            },
            "time_period": {
                "type": "string",
                "description": "Limit results to a specific time period",
                "enum": ["past_24h", "past_week", "past_month", "past_year"],
            },
            "exact_phrase": {
                "type": "string",
                "description": "Search for an exact phrase (will be enclosed in quotes)",
            },
            "related": {
                "type": "string",
                "description": "Find sites related to a specific URL",
            },
            "intitle": {
                "type": "string",
                "description": "Search for pages with specific text in the title",
            },
            "inurl": {
                "type": "string",
                "description": "Search for pages with specific text in the URL",
            },
            "intext": {
                "type": "string",
                "description": "Search for pages with specific text in their content",
            },
            "country": {
                "type": "string",
                "description": "Country code to localize search results (e.g., 'us', 'uk', 'ca')",
            },
            "language": {
                "type": "string",
                "description": "Language code to filter results (e.g., 'en', 'es', 'fr')",
            },
            "start": {
                "type": "integer",
                "description": "Start index for pagination of search results",
                "default": 0,
            },
        },
        "required": ["query"],
    }
    example = """
    google_search(
        query="What is the capital of France?",
        num_results=10
    )
    """

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        """
        Execute a Google search using Brightdata's API.

        Args:
            context: The processing context which may contain API credentials
            params: Dictionary including:
                query (str): The search term to look up on Google
                num_results (int, optional): Number of results to return
                maps (bool, optional): Whether to perform a Google Maps search
                site (str, optional): Limit search to a specific website
                filetype (str, optional): Limit search to specific file types
                time_period (str, optional): Limit results to a specific time period
                exact_phrase (str, optional): Search for an exact phrase
                related (str, optional): Find sites related to a specific URL
                intitle (str, optional): Search for pages with specific text in the title
                inurl (str, optional): Search for pages with specific text in the URL
                intext (str, optional): Search for pages with specific text in their content
                country (str, optional): Country code to localize search results
                language (str, optional): Language code to filter results
                start (int, optional): Start index for pagination of search results

        Returns:
            dict: Search results or error message
        """
        # Get required parameters
        query = params.get("query")
        if not query:
            return {"error": "Search query is required"}

        # Build the search query with advanced parameters
        search_query = query

        # Add site-specific search
        if params.get("site"):
            search_query += f" site:{params.get('site')}"

        # Add filetype filter
        if params.get("filetype"):
            search_query += f" filetype:{params.get('filetype')}"

        # Add exact phrase search
        if params.get("exact_phrase"):
            search_query += f' "{params.get("exact_phrase")}"'

        # Add related search
        if params.get("related"):
            search_query += f" related:{params.get('related')}"

        # Add intitle search
        if params.get("intitle"):
            search_query += f" intitle:{params.get('intitle')}"

        # Add inurl search
        if params.get("inurl"):
            search_query += f" inurl:{params.get('inurl')}"

        # Add intext search
        if params.get("intext"):
            search_query += f" intext:{params.get('intext')}"

        # URL construction based on search type
        url_encoded_query = urllib.parse.quote(search_query)
        # Regular Google search
        search_url = f"https://www.google.com/search?q={url_encoded_query}"

        # Add number of results parameter
        if params.get("num_results"):
            search_url += f"&num={params.get('num_results')}"

        # Add time period filter
        if params.get("time_period"):
            time_param = None
            if params.get("time_period") == "past_24h":
                time_param = "qdr:d"
            elif params.get("time_period") == "past_week":
                time_param = "qdr:w"
            elif params.get("time_period") == "past_month":
                time_param = "qdr:m"
            elif params.get("time_period") == "past_year":
                time_param = "qdr:y"

            if time_param:
                search_url += f"&tbs={time_param}"

        # Add safe search parameter
        if "safe_search" in params:
            safe = "active" if params.get("safe_search") else "off"
            search_url += f"&safe={safe}"

        # Add country parameter
        if params.get("country"):
            search_url += f"&gl={params.get('country')}"

        # Add language parameter
        if params.get("language"):
            search_url += f"&hl={params.get('language')}"

        # Add start index for pagination
        if params.get("start"):
            search_url += f"&start={params.get('start')}"

        # Make the API request
        result = await make_api_request(search_url)
        if "error" in result:
            raise Exception(result["error"])

        # Google-specific response handling
        if result["status_code"] == 200:
            soup = BeautifulSoup(result["body"], "html.parser")

            # Extract a > h3 elements (commonly used in Google search results)
            search_results = []
            for a_tag in soup.select("a:has(h3)"):
                href = a_tag.get("href")
                h3_text = a_tag.h3.get_text(strip=True) if a_tag.h3 else ""

                search_results.append({"href": href, "text": h3_text})

            return {
                "success": True,
                "results": search_results,
                "num_results": len(search_results),
            }
        raise Exception(
            f"Google search failed with status {result['status_code']}: {result['body']}"
        )


class WebFetchTool(Tool):
    """
    A tool that fetches HTML content from a URL and converts it to text.

    This tool enables language models to retrieve and process web content without
    needing a full browser, using BeautifulSoup for HTML parsing and html2text for
    conversion to Markdown.
    """

    name = "web_fetch"
    description = "Fetch HTML content from a URL, convert HTML to Markdown"
    input_schema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch content from",
            },
            "output_file": {
                "type": "string",
                "description": "Path to save the output file",
            },
            "headers": {
                "type": "object",
                "description": "Optional HTTP headers for the request",
            },
            "timeout": {
                "type": "number",
                "description": "Optional timeout for the request in seconds",
                "default": 30,
            },
        },
        "required": ["url", "output_file"],
    }
    example = """
    web_fetch(
        url="https://www.google.com",
        output_file="google.txt"
    )
    """

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        """
        Fetches HTML from a URL, extracts content using BeautifulSoup, and converts to text.

        Args:
            context: The processing context
            params: Dictionary including:
                url (str): The URL to fetch content from
                headers (dict, optional): HTTP headers for the request
                timeout (int, optional): Timeout for the request in seconds

        Returns:
            dict: Result containing the extracted text content or error message
        """
        try:
            url = params.get("url")
            if not url:
                return {"error": "URL is required"}

            headers = params.get("headers", {})
            timeout = params.get("timeout", 30)
            output_file = params.get("output_file")
            if output_file:
                output_file = context.resolve_workspace_path(output_file)
            else:
                raise Exception("Output file is required")

            # Make HTTP request
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=headers, timeout=timeout
                ) as response:
                    if response.status != 200:
                        return {
                            "error": f"HTTP request failed with status {response.status}",
                            "status_code": response.status,
                        }

                    # Check content type
                    content_type = response.headers.get("Content-Type", "").lower()
                    if not (
                        "text/html" in content_type
                        or "application/xhtml+xml" in content_type
                    ):
                        return await response.text()

                    html_content = await response.text()

            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")

            # Extract content based on selector
            body = soup.body
            if body:
                extracted_html = str(body)
            else:
                return {"error": "No body element found in the HTML", "url": url}

            with open(output_file, "w") as f:
                f.write(extracted_html)

            return {
                "success": True,
                "output_file": output_file,
            }

        except aiohttp.ClientError as e:
            raise Exception(f"HTTP request error: {str(e)}")
        except Exception as e:
            raise Exception(f"Error fetching and processing content: {str(e)}")


class DownloadFileTool(Tool):
    """
    A tool that downloads files from URLs and saves them to disk.

    This tool enables language models to retrieve files of any type from the web
    and save them to the workspace directory for further processing or analysis.
    Supports downloading multiple files in parallel.
    """

    name = "download_file"
    description = "Download a text or binaryfile from a URL and save it to disk"
    input_schema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL of the file to download",
            },
            "output_file": {
                "type": "string",
                "description": "Workspace relative path where to save the file",
            },
        },
        "required": ["url", "output_file"],
    }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        """
        Downloads a file from a URL and saves it to the specified path.

        Args:
            context: The processing context
            params: Dictionary including:
                url (str): URL of the file to download
                output_file (str): Workspace relative path where to save the file
                headers (dict, optional): HTTP headers for the request
                timeout (int, optional): Timeout for the request in seconds

        Returns:
            dict: Result containing download status information
        """
        try:
            # Handle both single URL and list of URLs
            url = params.get("url")
            output_file = params.get("output_file")

            if not url:
                return {"error": "URL is required"}
            if not output_file:
                return {"error": "Output file is required"}

            headers = params.get("headers", {})
            timeout = params.get("timeout", 60)

            # Create a semaphore to limit concurrent downloads
            import asyncio

            # Ensure the directory exists
            full_path = context.resolve_workspace_path(output_file)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=headers, timeout=timeout
                ) as response:
                    if response.status != 200:
                        return {
                            "url": url,
                            "output_file": output_file,
                            "success": False,
                            "error": f"HTTP request failed with status {response.status}",
                            "status_code": response.status,
                        }

                    # Get content type and size
                    content_type = response.headers.get("Content-Type", "unknown")
                    content_length = response.headers.get("Content-Length")
                    file_size = int(content_length) if content_length else None

                    # Read the file data and write to disk
                    with open(full_path, "wb") as f:
                        f.write(await response.read())

                    return {
                        "url": url,
                        "output_file": output_file,
                        "success": True,
                        "content_type": content_type,
                        "file_size_bytes": file_size,
                    }

        except Exception as e:
            return {"error": f"Error in download process: {str(e)}"}


class BrowserNavigationTool(Tool):
    """
    A tool that enables navigation and interaction within a browser session.

    This tool allows for clicking links, navigating between pages, and performing
    basic interactions while maintaining the browser session state.
    """

    name = "browser_navigate"
    description = "Navigate, interact with, and extract content from web pages in an active browser session"
    input_schema = {
        "type": "object",
        "properties": {
            "use_remote_browser": {
                "type": "boolean",
                "description": "Use a remote browser instead of a local one",
                "default": False,
            },
            "headless": {
                "type": "boolean",
                "description": "Run the browser in headless mode",
                "default": True,
            },
            "action": {
                "type": "string",
                "description": "Navigation or extraction action to perform",
                "enum": ["click", "goto", "back", "forward", "reload", "extract"],
            },
            "selector": {
                "type": "string",
                "description": "CSS selector for the element to interact with or extract from",
            },
            "url": {
                "type": "string",
                "description": "URL to navigate to (required for 'goto' action)",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in milliseconds for the action",
                "default": 30000,
            },
            "wait_for": {
                "type": "string",
                "description": "Optional selector to wait for after performing the action",
            },
            "extract_type": {
                "type": "string",
                "description": "Type of content to extract (for 'extract' action)",
                "enum": ["text", "html", "value", "attribute"],
                "default": "text",
            },
            "attribute": {
                "type": "string",
                "description": "Attribute name to extract (when extract_type is 'attribute')",
            },
        },
        "required": ["action"],
    }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        """
        Performs navigation and interaction actions in the browser.

        Args:
            context: The processing context containing the browser page
            params: Dictionary including:
                action (str): The action to perform (click, goto, back, forward, reload)
                selector (str): CSS selector for clicking elements
                url (str): URL for navigation
                timeout (int): Timeout for actions
                wait_for (str): Selector to wait for after action

        Returns:
            dict: Result containing action status and current page information
        """
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise Exception(
                "Playwright is not installed. Please install it with 'pip install playwright' and then run 'playwright install'"
            )

        timeout = params.get("timeout", 30000)
        headless = params.get("headless", True)

        # Initialize browser
        playwright_instance = await async_playwright().start()

        try:
            if params.get("use_remote_browser"):
                browser_endpoint = Environment.get(
                    "BRIGHTDATA_SCRAPING_BROWSER_ENDPOINT"
                )
                if not browser_endpoint:
                    raise Exception("BRIGHTDATA_SCRAPING_BROWSER_ENDPOINT is not set")
                browser = await playwright_instance.chromium.connect_over_cdp(
                    browser_endpoint
                )
                # Create context with additional permissions and settings
                browser_context = await browser.new_context(
                    bypass_csp=True,
                )
            else:
                # Launch browser with similar settings for local usage
                browser = await playwright_instance.chromium.launch(headless=headless)
                browser_context = await browser.new_context(
                    bypass_csp=True,
                )

            # Create page from the context
            page = await browser_context.new_page()

            action = params.get("action")
            wait_for = params.get("wait_for")

            result = {
                "success": True,
                "action": action,
            }

            # Perform the requested action
            if action == "click":
                selector = params.get("selector")
                if not selector:
                    return {"error": "Selector is required for click action"}

                # Wait for the element to be visible and clickable
                element = await page.wait_for_selector(selector, timeout=timeout)
                if element:
                    await element.click()
                    result["clicked_selector"] = selector
                else:
                    return {"error": f"Element not found: {selector}"}

            elif action == "goto":
                url = params.get("url")
                if not url:
                    return {"error": "URL is required for goto action"}

                await page.goto(url, timeout=timeout, wait_until="domcontentloaded")
                result["navigated_to"] = url

            elif action == "back":
                await page.go_back(timeout=timeout, wait_until="domcontentloaded")

            elif action == "forward":
                await page.go_forward(timeout=timeout, wait_until="domcontentloaded")

            elif action == "reload":
                await page.reload(timeout=timeout, wait_until="domcontentloaded")

            # Add new extract action
            elif action == "extract":
                selector = params.get("selector")
                extract_type = params.get("extract_type", "text")

                if selector:
                    # Wait for the element if specified
                    element = await page.wait_for_selector(selector, timeout=timeout)
                    if not element:
                        return {"error": f"Element not found: {selector}"}

                    if extract_type == "text":
                        content = await element.text_content()
                    elif extract_type == "html":
                        content = await element.inner_html()
                    elif extract_type == "value":
                        content = await element.input_value()
                    elif extract_type == "attribute":
                        attribute = params.get("attribute")
                        if not attribute:
                            return {
                                "error": "Attribute name is required for attribute extraction"
                            }
                        content = await element.get_attribute(attribute)
                else:
                    # Extract from entire page if no selector
                    if extract_type == "text":
                        content = await page.text_content("body")
                    elif extract_type == "html":
                        content = await page.content()
                    else:
                        return {
                            "error": f"Invalid extract_type '{extract_type}' for full page extraction"
                        }

                result["content"] = content
                result["extract_type"] = extract_type
                if selector:
                    result["selector"] = selector

            # Wait for additional element if specified
            if wait_for:
                await page.wait_for_selector(wait_for, timeout=timeout)
                result["waited_for"] = wait_for

            # Add current page information to result
            result.update({"current_url": page.url, "title": await page.title()})

            return result

        except Exception as e:
            return {
                "error": f"Navigation/extraction action failed: {str(e)}",
                "action": params.get("action"),
            }

        finally:
            # Always close the browser session
            await browser.close()
            await playwright_instance.stop()


class BrowserUseTool(Tool):
    """
    A tool that uses browser_use Agent to perform browser-based tasks.

    This tool enables language models to perform complex browser interactions and automated tasks
    including but not limited to:
    - Web navigation and form filling
    - Data extraction and comparison
    - Document creation and manipulation
    - E-commerce operations (adding to cart, checkout)
    - Social media interactions
    - Job applications and professional networking
    - File downloads and uploads
    """

    name = "browser_use"
    description = "Use browser_use Agent to automate browser-based tasks with natural language instructions"
    input_schema = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Natural language description of the browser task to perform. Can include complex multi-step instructions like 'Compare prices between websites', 'Fill out forms', or 'Extract specific data'.",
                "examples": [
                    "Compare the price of gpt-4 and DeepSeek-V3",
                    "Add grocery items to cart and checkout",
                    "Write a document in Google Docs and save as PDF",
                ],
            },
            "use_remote_browser": {
                "type": "boolean",
                "description": "Whether to use a remote browser instead of a local one.",
                "default": False,
            },
            "model": {
                "type": "string",
                "description": "The model to use for the browser agent.",
                "enum": ["gpt-4o", "claude-3-5-sonnet"],
            },
            "timeout": {
                "type": "integer",
                "description": "Maximum time in seconds to allow for task completion. Complex tasks may require longer timeouts.",
                "default": 300,
                "minimum": 1,
                "maximum": 3600,
            },
        },
        "required": ["task"],
    }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        """
        Execute a browser agent task.
        """
        try:
            task = params.get("task")
            if not task:
                return {"error": "Task description is required"}

            use_remote_browser = params.get("use_remote_browser", False)

            if use_remote_browser and Environment.get("CDP_URL"):
                browser_endpoint = Environment.get("CDP_URL")
                if not browser_endpoint:
                    raise ValueError(
                        "BrightData scraping browser endpoint not found in environment variables (BRIGHTDATA_SCRAPING_BROWSER_ENDPOINT)."
                    )
                browser = Browser(
                    config=BrowserConfig(
                        headless=True,
                        cdp_url=browser_endpoint,
                    )
                )
            else:
                browser = Browser(
                    config=BrowserConfig(
                        headless=True,
                    )
                )

            if params.get("model") == "gpt-4o":
                api_key = Environment.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError(
                        "OpenAI API key not found in environment variables (OPENAI_API_KEY)."
                    )
                llm = ChatOpenAI(
                    model="gpt-4o",
                    api_key=api_key,
                    temperature=0,
                    timeout=params.get("timeout", 300),
                )
            elif params.get("model") == "claude-3-5-sonnet":
                api_key = Environment.get("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError(
                        "Anthropic API key not found in environment variables (ANTHROPIC_API_KEY)."
                    )
                llm = ChatAnthropic(
                    model_name="claude-3-5-sonnet",
                    api_key=api_key,
                    temperature=0,
                    timeout=params.get("timeout", 300),
                    stop=["\n\n"],
                )
            else:
                raise ValueError(
                    f"Invalid model: {params.get('model')}. Must be one of: gpt-4o, claude-3-5-sonnet."
                )

            agent = Agent(
                task=task,
                llm=llm,
                browser=browser,
                save_conversation_path=context.resolve_workspace_path(
                    "browser_use.log"
                ),
            )

            # Run with timeout
            timeout = params.get("timeout", 300)
            try:
                result = await asyncio.wait_for(agent.run(), timeout=timeout)
                return {
                    "success": True,
                    "task": task,
                    "result": result,
                }
            except asyncio.TimeoutError:
                return {
                    "error": f"Task timed out after {timeout} seconds",
                    "task": task,
                }

        except Exception as e:
            return {
                "error": f"Browser agent task failed: {str(e)}",
                "task": task,
            }
