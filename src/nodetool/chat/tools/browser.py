"""
Browser interaction tools.

This module provides tools for interacting with web browsers and web pages.
"""

from typing import Any

from nodetool.workflows.processing_context import ProcessingContext
from .base import Tool


class BrowserTool(Tool):
    """
    A tool that allows controlling a web browser for web interactions.

    This tool enables language models to interact with web pages by performing
    actions like navigating to URLs, clicking elements, typing text, and retrieving
    content from web pages using Playwright.
    """

    def __init__(self):
        """
        Initialize the BrowserTool with its name, description, and parameter schema.

        Sets up the tool with a detailed parameter schema that defines different
        actions that can be performed (navigate, click, type, get_text, quit) and
        their respective required parameters.
        """
        super().__init__(
            name="browser_control",
            description="Control a web browser to navigate and interact with web pages",
        )
        self.input_schema = {
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
        self.browser = None
        self.page = None

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
    def __init__(self):
        super().__init__(
            name="take_screenshot",
            description="Take a screenshot of the current browser window or a specific element",
        )
        self.input_schema = {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "Optional CSS selector for capturing a specific element",
                },
                "path": {
                    "type": "string",
                    "description": "Path where to save the screenshot",
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

            if "selector" in params:
                element = await page.query_selector(params["selector"])
                if element:
                    await element.screenshot(path=path)
                else:
                    return {
                        "error": f"No element found matching selector: {params['selector']}"
                    }
            else:
                await page.screenshot(path=path)

            return {"success": True, "path": path}

        except Exception as e:
            return {"error": str(e)}
