"""
Browser interaction tools.

This module provides tools for interacting with web browsers and web pages.
"""

import os
from typing import Any, Tuple
import html2text

from nodetool.common.environment import Environment
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.tools.base import Tool
from playwright.async_api import Page

import os
from typing import Any
import asyncio


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
        },
        "required": ["url"],
    }
    example = """
    browser(
        url="https://www.google.com",
        output_file="google.json"
    )
    """

    def __init__(self):
        pass

    def user_message(self, params: dict) -> str:
        url = params.get("url", "a specific URL")
        msg = f"Browsing {url}..."
        if len(msg) > 80:
            msg = "Browsing a specified URL..."
        return msg

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
        # from playwright.async_api import async_playwright # Removed unused import

        url = params.get("url")
        timeout = (
            30000  # This is page navigation timeout, distinct from connection timeout
        )
        if not url:
            return {"error": "URL is required"}

        # headless = True # This is now controlled by _initialize_browser's internal launch_args_dict
        playwright_instance = None
        browser_context = None

        try:
            # Initialize browser using the helper function
            browser = await context.get_browser()
            browser_context = await browser.new_context(
                bypass_csp=True,
            )

            page = await browser_context.new_page()

            # Navigate to the URL with the specified timeout
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout)

            # Extract metadata from the page
            metadata = await extract_metadata(page)
            content = None

            # Directly use html2text on the full page content
            h = html2text.HTML2Text(baseurl=url, bodywidth=1000)
            h.ignore_images = True
            h.ignore_mailto_links = True
            content = h.handle(await page.content())

            return {
                "success": True,
                "url": url,
                "content": content,
                "metadata": metadata,
            }
        except Exception as e:
            print(e)
            return {"error": f"Error fetching page: {str(e)}"}

        finally:
            # Always close the browser session
            if browser_context:
                try:
                    await browser_context.close()
                except Exception as e:
                    pass


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
            "url": {
                "type": "string",
                "description": "URL to navigate to before taking screenshot",
            },
            "output_file": {
                "type": "string",
                "description": "Workspace relative path to save the screenshot",
                "default": "screenshot.png",
            },
        },
        "required": ["url", "output_file"],
    }
    example = """
    take_screenshot(
        url="https://example.com",
        selector=".title",
        path="title.png"
    )
    """

    def user_message(self, params: dict) -> str:
        url = params.get("url", "a page")
        output = params.get("output_file", "screenshot.png")
        msg = f"Taking screenshot of {url} and saving to {output}."
        if len(msg) > 80:
            msg = f"Taking screenshot of a page and saving to {output}."
        if len(msg) > 80:
            msg = "Taking screenshot of a page."
        return msg

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        url = params.get("url")
        if not url:
            return {"error": "URL is required for taking a screenshot"}

        timeout = 30000  # Page navigation timeout
        # headless = True # Controlled by _initialize_browser
        output_file = params.get("output_file", "screenshot.png")
        full_path = context.resolve_workspace_path(output_file)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        playwright_instance = None
        browser_context = None

        try:
            # Initialize browser
            browser = await context.get_browser()
            browser_context = await browser.new_context(
                bypass_csp=True,
            )
            page = await browser_context.new_page()

            # Navigate to the URL
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            await page.screenshot(path=full_path)

            return {"success": True, "output_file": full_path}

        except Exception as e:
            return {"error": f"Error taking screenshot: {str(e)}"}
        finally:
            # Always close the browser session
            if browser_context:
                try:
                    await browser_context.close()
                except Exception as e:
                    pass


if __name__ == "__main__":
    import asyncio

    # Import the actual ProcessingContext
    from nodetool.workflows.processing_context import ProcessingContext

    context = ProcessingContext()

    async def browser_tool_example():
        browser_tool = BrowserTool()
        result = await browser_tool.process(
            context, {"url": "https://news.ycombinator.com"}
        )
        print(result)

    async def reddit_example():
        browser_tool = BrowserTool()
        result = await browser_tool.process(
            context,
            {
                "url": "https://www.reddit.com/r/LocalLLaMA/comments/1ka8ban/qwen_3_unimpressive_coding_performance_so_far/",
            },
        )
        print(result)

    async def screenshot_tool_example():
        screenshot_tool = ScreenshotTool()
        result = await screenshot_tool.process(
            context,
            {
                "url": "https://news.ycombinator.com",
                "output_file": "example_screenshot.png",
            },
        )
        print(result)

    # asyncio.run(browser_tool_example())
    asyncio.run(reddit_example())
    # asyncio.run(screenshot_tool_example())
