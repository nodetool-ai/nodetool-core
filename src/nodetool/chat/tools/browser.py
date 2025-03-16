"""
Browser interaction tools.

This module provides tools for interacting with web browsers and web pages.
"""

import os
from typing import Any
import aiohttp
import json
import urllib.parse

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
                "description": "Action to perform: 'navigate', 'click', 'type', 'quit'",
                "enum": ["navigate", "click", "type", "get_text", "quit"],
            },
            "url": {
                "type": "string",
                "description": "URL to navigate to (for 'navigate' action)",
            },
            "selector": {
                "type": "string",
                "description": "CSS selector for the target element (for 'click', 'type', 'get_text' actions)",
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
                await page.goto(url)
                page_text = await page.inner_text("body")
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
        },
        "required": ["query"],
    }

    def _remove_base64_images(self, data):
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
                    data[key] = self._remove_base64_images(data[key])
        elif isinstance(data, list):
            for i in range(len(data)):
                data[i] = self._remove_base64_images(data[i])
        return data

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        """
        Execute a Google search using Brightdata's API.

        Args:
            context: The processing context which may contain API credentials
            params: Dictionary including:
                query (str): The search term to look up on Google
                num_results (int, optional): Number of results to return

        Returns:
            dict: Search results or error message
        """
        try:
            # Get API key from context if not provided during initialization
            api_key = Environment.get("BRIGHTDATA_API_KEY")
            if not api_key:
                return {
                    "error": "Brightdata API key not found. Please provide it in the secrets as 'BRIGHTDATA_API_KEY'."
                }

            # Get required parameters
            query = params.get("query")
            if not query:
                return {"error": "Search query is required"}

            zone = Environment.get("BRIGHTDATA_ZONE")
            if not zone:
                return {
                    "error": "Brightdata zone not found. Please provide it in the secrets as 'BRIGHTDATA_ZONE'."
                }

            url_encoded_query = urllib.parse.quote(query)
            # Construct Google search URL
            search_url = (
                f"https://www.google.com/search?q={url_encoded_query}&brd_json=1"
            )
            if params.get("num_results"):
                search_url += f"&num={params.get('num_results')}"

            # Prepare request to Brightdata API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            payload = {"zone": zone, "url": search_url, "format": "json"}

            # Make the API request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.brightdata.com/request", headers=headers, json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return {
                            "error": f"Brightdata API request failed with status {response.status}: {error_text}"
                        }

                    result = await response.json()
                    if result["status_code"] == 200:
                        body = json.loads(result["body"])
                        body = self._remove_base64_images(body)
                        # Create the raw search result
                        search_result = {
                            "success": True,
                            "query": query,
                            "result": body,
                        }
                        # Apply the extraction filter to organize links and titles
                        extracted_data = extract_links_and_titles(search_result)
                        # Return both the raw data and the extracted links/titles
                        return {
                            "success": True,
                            "query": query,
                            "result": extracted_data,
                        }

                    return {
                        "error": f"Google search failed with status {result['status_code']}: {result['body']}"
                    }

        except Exception as e:
            return {"error": f"Error performing Google search: {str(e)}"}


def extract_links_and_titles(search_result):
    """Extract links and titles from Google search results."""
    extracted_data = {}

    # Extract from organic search results
    if "organic" in search_result["result"]:
        organic = []
        for item in search_result["result"]["organic"]:
            organic.append({"title": item["title"], "link": item["link"]})
        extracted_data["organic"] = organic

    # Extract from top stories
    # if (
    #     "top_stories" in search_result["result"]
    #     and "items" in search_result["result"]["top_stories"]
    # ):
    #     top_stories = []
    #     for item in search_result["result"]["top_stories"]["items"]:
    #         top_stories.append({"title": item["title"], "link": item["link"]})
    #     extracted_data["top_stories"] = top_stories

    # # Extract from videos section
    # if "videos" in search_result["result"]:
    #     videos = []
    #     for item in search_result["result"]["videos"]:
    #         videos.append(
    #             {
    #                 "title": item.get(
    #                     "title", f"Video by {item.get('author', 'Unknown')}"
    #                 ),
    #                 "link": item["link"],
    #             }
    #         )
    #     extracted_data["videos"] = videos

    return extracted_data
