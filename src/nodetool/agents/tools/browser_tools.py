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
from playwright.async_api import Page, Playwright, BrowserContext, async_playwright

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


async def _initialize_browser(
    headless: bool = True, use_remote_browser: bool = False
) -> Tuple[Playwright, BrowserContext]:
    """
    Initializes a Playwright browser instance (local or remote).

    Args:
        headless: Run the browser in headless mode (local only).
        use_remote_browser: Use a remote browser endpoint.

    Returns:
        A tuple containing the Playwright instance and the BrowserContext.
    """
    playwright_instance = await async_playwright().start()

    if use_remote_browser:
        browser_endpoint = Environment.get("BRIGHTDATA_SCRAPING_BROWSER_ENDPOINT")
        if not browser_endpoint:
            raise ValueError(
                "BRIGHTDATA_SCRAPING_BROWSER_ENDPOINT environment variable not set."
            )
        browser = await playwright_instance.chromium.connect_over_cdp(browser_endpoint)
        browser_context = await browser.new_context(bypass_csp=True)
    else:
        browser = await playwright_instance.chromium.launch(headless=headless)
        browser_context = await browser.new_context(bypass_csp=True)

    return playwright_instance, browser_context


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
        from playwright.async_api import async_playwright

        url = params.get("url")
        output_file = params.get("output_file")
        timeout = 30000
        if not url:
            return {"error": "URL is required"}

        headless = True
        use_remote_browser = (
            Environment.get("BRIGHTDATA_SCRAPING_BROWSER_ENDPOINT") is not None
        )

        playwright_instance = None
        browser_context = None

        try:
            # Initialize browser using the helper function
            playwright_instance, browser_context = await _initialize_browser(
                headless=headless, use_remote_browser=use_remote_browser
            )
            page = await browser_context.new_page()

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
            if playwright_instance:
                try:
                    await playwright_instance.stop()
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

        timeout = 30000
        headless = True
        output_file = params.get("output_file", "screenshot.png")
        full_path = context.resolve_workspace_path(output_file)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        playwright_instance = None
        browser_context = None

        try:
            # Initialize browser
            playwright_instance, browser_context = await _initialize_browser(
                headless=headless
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
            if playwright_instance:
                try:
                    await playwright_instance.stop()
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

    async def remote_browser_example():
        browser_tool = BrowserTool()
        result = await browser_tool.process(
            context,
            {
                "url": "https://www.reddit.com/r/LocalLLaMA/comments/1ka8ban/qwen_3_unimpressive_coding_performance_so_far/",
                "use_remote_browser": True,
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
    # asyncio.run(remote_browser_example())
    asyncio.run(screenshot_tool_example())
