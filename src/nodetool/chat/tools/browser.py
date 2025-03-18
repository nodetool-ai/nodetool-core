"""
Browser interaction tools.

This module provides tools for interacting with web browsers and web pages.
"""

import os
from typing import Any
import aiohttp
import json
import urllib.parse
import html2text
from bs4 import BeautifulSoup

from nodetool.common.environment import Environment
from nodetool.workflows.processing_context import ProcessingContext
from .base import Tool


class BrowserTool(Tool):
    """
    A tool that allows controlling a web browser for web interactions.

    This tool enables language models to interact with web pages by performing
    actions like navigating to URLs, clicking elements, typing text, and retrieving
    content from web pages using Playwright.
    """

    name = "browser_control"
    description = "Control a web browser to navigate and interact with web pages"
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Action to perform",
                "enum": [
                    "navigate",
                    "click",
                    "type",
                    "quit",
                ],
            },
            "url": {
                "type": "string",
                "description": "URL to navigate to (for 'navigate' action)",
            },
            "selector": {
                "type": "string",
                "description": "CSS selector for the target element (for 'click', 'type' actions)",
            },
            "text": {
                "type": "string",
                "description": "Text to type (for 'type' action)",
            },
        },
        "required": ["action"],
    }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        """
        Executes browser actions using Playwright.

        Supports various browser actions including navigation, clicking elements,
        typing text, retrieving text, and closing the browser.

        Args:
            context: The processing context
            params: Dictionary including:
                action (str): The action to perform ('navigate', 'click', 'type', 'get_text', 'quit')
                url (str, optional): URL to navigate to (for 'navigate' action)
                selector (str, optional): CSS selector for element (for 'click', 'type', 'get_text' actions)
                text (str, optional): Text to type (for 'type' action)

        Returns:
            dict: Result of the browser action or error message
        """
        try:
            # Import here to avoid requiring playwright for the entire module
            try:
                from playwright.async_api import async_playwright
            except ImportError:
                return {
                    "error": "Playwright is not installed. Please install it with 'pip install playwright' and then run 'playwright install'"
                }

            action = params.get("action")

            # Handle quit action separately
            if action == "quit":
                playwright_instance = context.get("playwright_instance")
                browser = context.get("playwright_browser")

                if browser:
                    await browser.close()
                    context.set("playwright_browser", None)

                if playwright_instance:
                    await playwright_instance.stop()
                    context.set("playwright_instance", None)

                context.set("playwright_page", None)
                return {"success": True, "action": "quit"}

            # Initialize browser if not already done
            browser = context.get("playwright_browser")
            page = context.get("playwright_page")

            if browser is None or page is None:
                playwright_instance = await async_playwright().start()
                browser = await playwright_instance.chromium.launch(headless=True)
                page = await browser.new_page()

                context.set("playwright_instance", playwright_instance)
                context.set("playwright_browser", browser)
                context.set("playwright_page", page)

            # At this point, page should never be None
            if page is None:
                return {"error": "Failed to initialize browser page"}

            if action == "navigate":
                url = params.get("url")
                if not url:
                    return {"error": "URL is required for navigate action"}
                await page.goto(url, wait_until="networkidle")
                page_html = await page.inner_html("body")
                page_text = html2text.html2text(page_html)
                return {"success": True, "url": url, "body": page_text}

            elif action in ["click", "type", "get_text"]:
                selector = params.get("selector")
                if not selector:
                    return {"error": "Selector is required for element actions"}

                try:
                    # Wait for element to be present
                    await page.wait_for_selector(
                        selector, state="visible", timeout=10000
                    )

                    if action == "click":
                        await page.click(selector)
                        return {"success": True, "action": "click"}
                    elif action == "type":
                        text = params.get("text")
                        if not text:
                            return {"error": "Text is required for type action"}
                        await page.fill(selector, text)
                        return {"success": True, "action": "type"}
                    elif action == "get_text":
                        elements = await page.query_selector_all(selector)
                        texts = [await element.inner_text() for element in elements]
                        return {"text": texts, "count": len(texts)}
                except Exception as e:
                    return {"error": f"Error interacting with element: {str(e)}"}

            return {"error": f"Invalid action specified: {action}"}

        except Exception as e:
            return {"error": str(e)}


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

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            page = context.get("playwright_page")
            if page is None:
                return {"error": "No browser session available"}

            path = params.get("path", "screenshot.png")
            full_path = os.path.join(self.workspace_dir, path)
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
                "BRIGHTDATA_ZONE",
                "Brightdata zone not found. Please provide it in the secrets as 'BRIGHTDATA_ZONE'.",
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
                    return {
                        "error": f"API request failed with status {response.status}: {error_text}"
                    }

                return await response.json()
    except Exception as e:
        return {"error": f"Error making API request: {str(e)}"}


def get_required_api_key(key_name, error_message=None):
    """Get a required API key from environment variables."""
    api_key = Environment.get(key_name)
    if not api_key:
        return {
            "error": error_message or f"{key_name} not found in environment variables."
        }
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
                "default": 10,
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
        try:
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
            search_url = (
                f"https://www.google.com/search?q={url_encoded_query}&brd_json=1"
            )

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
                return result

            # Google-specific response handling
            if result["status_code"] == 200:
                body = json.loads(result["body"])

                return {
                    "success": True,
                    "general": body["general"],
                    "input": body["input"],
                    "organic": [
                        {
                            "title": item["title"],
                            "link": item["link"],
                            "description": item["description"],
                        }
                        for item in body["organic"]
                    ],
                }
            return {
                "error": f"Google search failed with status {result['status_code']}: {result['body']}"
            }

        except Exception as e:
            return {"error": f"Error performing Google search: {str(e)}"}


class WebFetchTool(Tool):
    """
    A tool that fetches HTML content from a URL and converts it to text.

    This tool enables language models to retrieve and process web content without
    needing a full browser, using BeautifulSoup for HTML parsing and html2text for
    conversion to plain text.
    """

    name = "web_fetch"
    description = "Fetch HTML from a URL and extract text content using BeautifulSoup"
    input_schema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch content from",
            },
            "selector": {
                "type": "string",
                "description": "Optional CSS selector to extract specific elements (defaults to 'body')",
                "default": "body",
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
        "required": ["url"],
    }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        """
        Fetches HTML from a URL, extracts content using BeautifulSoup, and converts to text.

        Args:
            context: The processing context
            params: Dictionary including:
                url (str): The URL to fetch content from
                selector (str, optional): CSS selector for extracting specific elements (defaults to 'body')
                headers (dict, optional): HTTP headers for the request
                timeout (int, optional): Timeout for the request in seconds

        Returns:
            dict: Result containing the extracted text content or error message
        """
        try:
            url = params.get("url")
            if not url:
                return {"error": "URL is required"}

            selector = params.get("selector", "body")
            headers = params.get("headers", {})
            timeout = params.get("timeout", 30)

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

                    html_content = await response.text()

            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")

            # Extract content based on selector
            if selector:
                elements = soup.select(selector)
                if not elements:
                    return {
                        "error": f"No elements found matching selector: {selector}",
                        "url": url,
                    }

                # Get HTML content of all matching elements
                extracted_html = "".join(str(element) for element in elements)
            else:
                # Default to body if no selector provided
                body = soup.body
                if body:
                    extracted_html = str(body)
                else:
                    return {"error": "No body element found in the HTML", "url": url}

            return html2text.html2text(extracted_html)

        except aiohttp.ClientError as e:
            return {"error": f"HTTP request error: {str(e)}", "url": url}
        except Exception as e:
            return {
                "error": f"Error fetching and processing content: {str(e)}",
                "url": url,
            }


class DownloadFilesTool(Tool):
    """
    A tool that downloads files from URLs and saves them to disk.

    This tool enables language models to retrieve files of any type from the web
    and save them to the workspace directory for further processing or analysis.
    Supports downloading multiple files in parallel.
    """

    name = "download_file"
    description = "Download one or more files from URLs and save them to disk"
    input_schema = {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "description": "URL or list of URLs of the files to download",
                "items": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "UjRL of the file to download",
                        },
                        "path": {
                            "type": "string",
                            "description": "Workspace relative path where to save the file",
                        },
                    },
                },
            },
            "max_concurrent": {
                "type": "integer",
                "description": "Maximum number of concurrent downloads",
                "default": 5,
            },
        },
        "required": ["urls", "paths"],
    }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        """
        Downloads one or more files from URLs and saves them to the specified paths.
        Downloads are performed in parallel for better efficiency.

        Args:
            context: The processing context
            params: Dictionary including:
                urls (str or list): URL or list of URLs of the files to download
                paths (str or list): Path or list of paths where to save the files
                headers (dict, optional): HTTP headers for the requests
                timeout (int, optional): Timeout for the requests in seconds
                max_concurrent (int, optional): Maximum number of concurrent downloads

        Returns:
            dict: Result containing download status information for each file
        """
        try:
            # Handle both single URL and list of URLs
            urls = params.get("urls")
            paths = params.get("paths")

            if not urls:
                return {"error": "URLs are required"}
            if not paths:
                return {"error": "Save paths are required"}

            # Convert single values to lists for uniform processing
            if isinstance(urls, str):
                urls = [urls]
            if isinstance(paths, str):
                paths = [paths]

            # Validate that URLs and paths have the same length
            if len(urls) != len(paths):
                return {"error": "Number of URLs must match number of paths"}

            headers = params.get("headers", {})
            timeout = params.get("timeout", 60)
            max_concurrent = params.get("max_concurrent", 5)

            # Create a semaphore to limit concurrent downloads
            import asyncio

            semaphore = asyncio.Semaphore(max_concurrent)

            # Define the download function for a single file
            async def download_single_file(url, path):
                async with semaphore:
                    try:
                        # Ensure the directory exists
                        full_path = os.path.join(self.workspace_dir, path)
                        os.makedirs(os.path.dirname(full_path), exist_ok=True)

                        async with aiohttp.ClientSession() as session:
                            async with session.get(
                                url, headers=headers, timeout=timeout
                            ) as response:
                                if response.status != 200:
                                    return {
                                        "url": url,
                                        "path": path,
                                        "success": False,
                                        "error": f"HTTP request failed with status {response.status}",
                                        "status_code": response.status,
                                    }

                                # Get content type and size
                                content_type = response.headers.get(
                                    "Content-Type", "unknown"
                                )
                                content_length = response.headers.get("Content-Length")
                                file_size = (
                                    int(content_length) if content_length else None
                                )

                                # Read the file data and write to disk
                                with open(full_path, "wb") as f:
                                    f.write(await response.read())

                                return {
                                    "url": url,
                                    "path": full_path,
                                    "success": True,
                                    "content_type": content_type,
                                    "file_size_bytes": file_size,
                                }
                    except aiohttp.ClientError as e:
                        return {
                            "url": url,
                            "path": path,
                            "success": False,
                            "error": f"HTTP request error: {str(e)}",
                        }
                    except Exception as e:
                        return {
                            "url": url,
                            "path": path,
                            "success": False,
                            "error": f"Error downloading file: {str(e)}",
                        }

            # Run downloads in parallel
            download_tasks = [
                download_single_file(url, path) for url, path in zip(urls, paths)
            ]
            results = await asyncio.gather(*download_tasks)

            # Compile the final results
            successful = [r for r in results if r.get("success")]
            failed = [r for r in results if not r.get("success")]

            return {
                "total": len(results),
                "successful": len(successful),
                "failed": len(failed),
                "results": results,
                "message": f"Downloaded {len(successful)} of {len(results)} files successfully",
            }

        except Exception as e:
            return {"error": f"Error in download process: {str(e)}"}
