"""
Browser interaction tools.

This module provides tools for interacting with web browsers and web pages.
"""

import asyncio
import json
import os
from contextlib import suppress
from typing import Any, ClassVar, Dict, Optional
from urllib.parse import urlparse

import html2text
from huggingface_hub import AsyncInferenceClient, InferenceClient
from playwright.async_api import ElementHandle, Page

from nodetool.agents.tools.base import Tool
from nodetool.metadata.types import Message, ToolCall
from nodetool.workflows.processing_context import ProcessingContext


class ReaderTool:
    """
    Tool for extracting text from a HTML document.
    """

    name = "reader_lm"
    description = (
        "Send a chat completion request to jinaai/ReaderLM-v2:featherless-ai on HuggingFace Hub."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The input message for the chatbot.",
            }
        },
        "required": ["message"],
    }
    example = """
    reader_lm(
        message="What is the capital of France?"
    )
    """

    async def get_client(self, context: ProcessingContext) -> AsyncInferenceClient | None:
        if not hasattr(self, "client") or self.client is None:
            hf_token = await context.get_secret("HF_TOKEN")
            if not hf_token:
                return None
            self.client = AsyncInferenceClient(api_key=hf_token, provider="featherless-ai")
        return self.client

    async def process(self, context: ProcessingContext, params: dict) -> str:
        """
        params: dict with 'message' key
        """
        client = await self.get_client(context)
        user_message = params.get("message")
        if not user_message:
            raise ValueError("Missing required parameter: message")
        if client is None:
            return user_message

        # The .chat.completions.create API is synchronous, so run in executor
        completion = await client.chat.completions.create(
            model="jinaai/ReaderLM-v2",
            messages=[{
                "role": "user",
                "content": user_message
            }]
        )
        return completion.choices[0].message.content or ""

    def user_message(self, params):
        return f"Calling ReaderLM-v2 for input: {params.get('message', '')[:60]}..."


def generate_css_path(element_info: Dict[str, Any], parent_path: str = "") -> str:
    """
    Generate a CSS selector path for an element based on its properties.

    Args:
        element_info: Dictionary containing element information
        parent_path: CSS path of the parent element

    Returns:
        str: CSS selector path that can be used to find this element
    """
    tag = element_info.get("tagName", "")
    id_attr = element_info.get("id")
    class_name = element_info.get("className")

    # If element has an ID, use it (most specific)
    if id_attr:
        return f"#{id_attr}"

    # Build selector with tag and classes
    selector = tag
    if class_name:
        # Split classes and join with dots
        classes = class_name.strip().split()
        if classes:
            selector += "." + ".".join(
                classes[:2]
            )  # Limit to first 2 classes for readability

    # If we have a parent path, combine them
    if parent_path:
        return f"{parent_path} > {selector}"

    return selector


async def get_element_info(element: ElementHandle) -> Dict[str, Any]:
    """
    Extract comprehensive information about a DOM element.

    Args:
        element: Playwright ElementHandle

    Returns:
        dict: Element information including tag, attributes, text, etc.
    """
    try:
        info = await element.evaluate(
            """(element) => {
            const rect = element.getBoundingClientRect();
            const computedStyle = window.getComputedStyle(element);

            // Function to get nth-child position
            function getNthChild(el) {
                let nth = 1;
                let sibling = el.previousElementSibling;
                while (sibling) {
                    if (sibling.tagName === el.tagName) {
                        nth++;
                    }
                    sibling = sibling.previousElementSibling;
                }
                return nth;
            }

            return {
                tagName: element.tagName.toLowerCase(),
                id: element.id || null,
                className: element.className || null,
                attributes: Array.from(element.attributes).reduce((acc, attr) => {
                    acc[attr.name] = attr.value;
                    return acc;
                }, {}),
                textContent: element.textContent?.trim() || null,
                innerHTML: element.innerHTML,
                outerHTML: element.outerHTML.substring(0, 500), // Truncate for readability
                boundingBox: {
                    x: rect.x,
                    y: rect.y,
                    width: rect.width,
                    height: rect.height
                },
                isVisible: rect.width > 0 && rect.height > 0,
                computedStyle: {
                    display: computedStyle.display,
                    visibility: computedStyle.visibility,
                    opacity: computedStyle.opacity,
                    fontSize: computedStyle.fontSize,
                    fontWeight: computedStyle.fontWeight,
                    color: computedStyle.color,
                    backgroundColor: computedStyle.backgroundColor
                },
                childrenCount: element.children.length,
                parentTag: element.parentElement ? element.parentElement.tagName.toLowerCase() : null,
                nthChild: getNthChild(element),
                hasUniqueClass: element.className ? document.querySelectorAll('.' + element.className.split(' ')[0]).length === 1 : false
            };
        }"""
        )

        # Generate CSS path for the element
        info["cssPath"] = generate_css_path(info)

        return info
    except Exception as e:
        return {"error": f"Failed to get element info: {str(e)}"}


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
    input_schema: ClassVar[dict[str, Any]] = {
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

    SEARCH_ENGINE_HOSTS = (
        "google.",
        "bing.",
        "search.yahoo",
        "duckduckgo",
        "yandex",
        "baidu",
        "ask.",
        "jina.ai",
    )

    def __init__(self):
        self._reader_tool: ReaderTool | None = None

    def user_message(self, params: dict) -> str:
        url = params.get("url", "a specific URL")
        msg = f"Browsing {url}..."
        if len(msg) > 160:
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

        parsed_host = urlparse(url).netloc.lower()
        if any(host in parsed_host for host in self.SEARCH_ENGINE_HOSTS):
            return {
                "error": "Direct browsing of search engine result pages is disabled. Use a SERP tool (e.g., google_search) instead.",
                "url": url,
            }

        browser_context = None

        try:
            # Initialize browser using the helper function
            browser = await context.get_browser()
            browser_context = await browser.new_context(
                locale="en-US",
                timezone_id="America/New_York",
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36",
            )

            page = await browser_context.new_page()
            await page.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )

            # Navigate to the URL with the specified timeout
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout)

            # Extract metadata from the page
            metadata = await extract_metadata(page)
            content = None

            # Directly use html2text on the full page content
            h = html2text.HTML2Text(baseurl=url, bodywidth=1000)
            h.ignore_images = True
            h.ignore_mailto_links = True
            html = await page.content()
            content = h.handle(html)

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
                with suppress(Exception):
                    await browser_context.close()


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
    input_schema: ClassVar[dict[str, Any]] = {
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
        if len(msg) > 160:
            msg = f"Taking screenshot of a page and saving to {output}."
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
                with suppress(Exception):
                    await browser_context.close()


class DOMExamineTool(Tool):
    """
    A tool that examines DOM structure and provides detailed information about elements.

    This tool allows inspection of DOM elements, their properties, styles, and hierarchy.
    """

    name = "dom_examine"
    description = "Examine DOM structure and get detailed information about elements"
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to navigate to",
            },
            "selector": {
                "type": "string",
                "description": "CSS selector to examine specific elements (optional - examines full DOM structure if not provided)",
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum depth to traverse in DOM tree",
                "default": 3,
            },
        },
        "required": ["url"],
    }
    example = """
    dom_examine(
        url="https://example.com",
        selector=".main-content",
        max_depth=2
    )
    """

    def user_message(self, params: dict) -> str:
        url = params.get("url", "a page")
        selector = params.get("selector")
        if selector:
            return f"Examining DOM elements matching '{selector}' on {url}"
        return f"Examining DOM structure of {url}"

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        url = params.get("url")
        if not url:
            return {"error": "URL is required"}

        selector = params.get("selector")
        max_depth = params.get("max_depth", 3)

        try:
            page = await context.get_browser_page(url)

            if selector:
                # Examine specific elements
                elements = await page.locator(selector).all()
                results = []

                for i, element in enumerate(elements[:10]):  # Limit to first 10 matches
                    info = await get_element_info(element)  # type: ignore
                    info["index"] = i

                    # Try to generate a more specific CSS path
                    if info.get("nthChild") and info.get("parentTag"):
                        info["specificCssPath"] = (
                            f"{info['parentTag']} > {info['tagName']}:nth-child({info['nthChild']})"
                        )

                    results.append(info)

                return {
                    "success": True,
                    "url": url,
                    "selector": selector,
                    "matchCount": len(elements),
                    "elements": results,
                }
            else:
                # Examine overall DOM structure
                dom_info = await page.evaluate(
                    f"""() => {{
                    function analyzeDOM(element, depth = 0, maxDepth = {max_depth}) {{
                        if (depth > maxDepth) return null;

                        const children = Array.from(element.children)
                            .filter(child => !['script', 'style', 'noscript'].includes(child.tagName.toLowerCase()))
                            .slice(0, 10)  // Limit children per level
                            .map(child => analyzeDOM(child, depth + 1, maxDepth))
                            .filter(Boolean);

                        return {{
                            tag: element.tagName.toLowerCase(),
                            id: element.id || null,
                            classes: element.className ? element.className.split(' ').filter(Boolean) : [],
                            textLength: element.textContent?.trim().length || 0,
                            childCount: element.children.length,
                            children: children
                        }};
                    }}

                    return {{
                        title: document.title,
                        url: window.location.href,
                        bodyStructure: analyzeDOM(document.body),
                        statistics: {{
                            totalElements: document.getElementsByTagName('*').length,
                            images: document.images.length,
                            links: document.links.length,
                            forms: document.forms.length,
                        }}
                    }};
                }}"""
                )

                return {
                    "success": True,
                    "domInfo": dom_info,
                }

        except Exception as e:
            return {"error": f"Error examining DOM: {str(e)}"}


class DOMSearchTool(Tool):
    """
    A tool that searches for DOM elements using various criteria.

    This tool finds elements by text content, attributes, styles, or complex queries.
    """

    name = "dom_search"
    description = "Search for DOM elements using various criteria"
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to navigate to",
            },
            "search_type": {
                "type": "string",
                "description": "Type of search: 'text', 'attribute', 'style', 'xpath', 'css'",
                "enum": ["text", "attribute", "style", "xpath", "css"],
            },
            "query": {
                "type": "string",
                "description": "Search query based on search_type",
            },
            "exact_match": {
                "type": "boolean",
                "description": "Whether to use exact matching for text searches",
                "default": False,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 10,
            },
        },
        "required": ["url", "search_type", "query"],
    }
    example = """
    dom_search(
        url="https://example.com",
        search_type="text",
        query="Click here",
        exact_match=True
    )
    """

    def user_message(self, params: dict) -> str:
        url = params.get("url", "a page")
        search_type = params.get("search_type", "elements")
        query = params.get("query", "")
        return f"Searching for {search_type} matching '{query}' on {url}"

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        url = params.get("url")
        if not url:
            return {"error": "URL is required"}

        search_type = params.get("search_type")
        query = params.get("query")
        exact_match = params.get("exact_match", False)
        limit = params.get("limit", 10)

        if not search_type or not query:
            return {"error": "search_type and query are required"}

        try:
            page = await context.get_browser_page(url)

            elements = []

            if search_type == "text":
                if exact_match:
                    elements = await page.locator(f"*:has-text('{query}')").all()
                else:
                    elements = await page.locator(
                        f"*:text-matches('{query}', 'i')"
                    ).all()

            elif search_type == "attribute":
                # Parse query as "attribute=value" or just "attribute"
                if "=" in query:
                    attr, value = query.split("=", 1)
                    elements = await page.locator(f"[{attr}='{value}']").all()
                else:
                    elements = await page.locator(f"[{query}]").all()

            elif search_type == "style":
                # Search by computed style property
                js_query = f"""
                    Array.from(document.querySelectorAll('*')).filter(el => {{
                        const style = window.getComputedStyle(el);
                        return {query};
                    }})
                """
                element_handles = await page.evaluate_handle(js_query)
                elements = await element_handles.evaluate("els => els")

            elif search_type == "xpath":
                elements = await page.locator(f"xpath={query}").all()

            elif search_type == "css":
                elements = await page.locator(query).all()

            # Process found elements
            results = []
            for i, element in enumerate(elements[:limit]):
                try:
                    info = await get_element_info(element)  # type: ignore
                    info["index"] = i

                    # Try to generate a more specific CSS path
                    if info.get("nthChild") and info.get("parentTag"):
                        info["specificCssPath"] = (
                            f"{info['parentTag']} > {info['tagName']}:nth-child({info['nthChild']})"
                        )

                    # Also generate an XPath for the element
                    xpath = await element.evaluate(
                        """(el) => {
                        function getXPath(element) {
                            if (element.id) {
                                return '//*[@id="' + element.id + '"]';
                            }
                            if (element === document.body) {
                                return '/html/body';
                            }
                            let ix = 0;
                            let siblings = element.parentNode.childNodes;
                            for (let i = 0; i < siblings.length; i++) {
                                let sibling = siblings[i];
                                if (sibling === element) {
                                    return getXPath(element.parentNode) + '/' + element.tagName.toLowerCase() + '[' + (ix + 1) + ']';
                                }
                                if (sibling.nodeType === 1 && sibling.tagName === element.tagName) {
                                    ix++;
                                }
                            }
                        }
                        return getXPath(el);
                    }"""
                    )
                    info["xpath"] = xpath

                    results.append(info)
                except Exception:
                    continue

            return {
                "success": True,
                "url": url,
                "search_type": search_type,
                "query": query,
                "totalMatches": len(elements),
                "results": results,
            }

        except Exception as e:
            return {"error": f"Error searching DOM: {str(e)}"}


class DOMExtractTool(Tool):
    """
    A tool that extracts specific content from DOM elements.

    This tool can extract text, attributes, HTML, or structured data from elements.
    """

    name = "dom_extract"
    description = "Extract content from specific DOM elements"
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to navigate to",
            },
            "selector": {
                "type": "string",
                "description": "CSS selector for elements to extract from",
            },
            "extract_type": {
                "type": "string",
                "description": "Type of content to extract: 'text', 'html', 'attributes', 'table', 'links', 'structured'",
                "enum": ["text", "html", "attributes", "table", "links", "structured"],
            },
            "output_file": {
                "type": "string",
                "description": "Optional file to save extracted content (workspace relative)",
            },
        },
        "required": ["url", "selector", "extract_type"],
    }
    example = """
    dom_extract(
        url="https://example.com",
        selector="table.data-table",
        extract_type="table",
        output_file="extracted_data.json"
    )
    """

    def user_message(self, params: dict) -> str:
        url = params.get("url", "a page")
        extract_type = params.get("extract_type", "content")
        selector = params.get("selector", "elements")
        return f"Extracting {extract_type} from {selector} on {url}"

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        url = params.get("url")
        if not url:
            return {"error": "URL is required"}

        selector = params.get("selector")
        extract_type = params.get("extract_type")
        output_file = params.get("output_file")

        if not selector or not extract_type:
            return {"error": "selector and extract_type are required"}

        try:
            page = await context.get_browser_page(url)

            extracted_data = None

            if extract_type == "text":
                elements = await page.locator(selector).all()
                extracted_data = []
                for element in elements:
                    text = await element.text_content()
                    if text:
                        extracted_data.append(text.strip())

            elif extract_type == "html":
                elements = await page.locator(selector).all()
                extracted_data = []
                for element in elements:
                    html = await element.inner_html()
                    extracted_data.append(html)

            elif extract_type == "attributes":
                elements = await page.locator(selector).all()
                extracted_data = []
                for element in elements:
                    attrs = await element.evaluate(
                        """(el) => {
                        return Array.from(el.attributes).reduce((acc, attr) => {
                            acc[attr.name] = attr.value;
                            return acc;
                        }, {});
                    }"""
                    )
                    extracted_data.append(attrs)

            elif extract_type == "table":
                # Extract table data as structured JSON
                extracted_data = await page.evaluate(
                    """(selector) => {
                    const tables = document.querySelectorAll(selector);
                    return Array.from(tables).map(table => {
                        const headers = Array.from(table.querySelectorAll('th')).map(th => th.textContent.trim());
                        const rows = Array.from(table.querySelectorAll('tr')).slice(1).map(row => {
                            const cells = Array.from(row.querySelectorAll('td'));
                            if (headers.length > 0) {
                                return cells.reduce((acc, cell, index) => {
                                    acc[headers[index] || `col_${index}`] = cell.textContent.trim();
                                    return acc;
                                }, {});
                            } else {
                                return cells.map(cell => cell.textContent.trim());
                            }
                        });
                        return { headers, rows };
                    });
                }""",
                    selector,
                )

            elif extract_type == "links":
                # Extract all links within selected elements
                extracted_data = await page.evaluate(
                    """(selector) => {
                    const containers = document.querySelectorAll(selector);
                    const links = [];
                    containers.forEach(container => {
                        container.querySelectorAll('a[href]').forEach(link => {
                            links.push({
                                text: link.textContent.trim(),
                                href: link.href,
                                title: link.title || null,
                                target: link.target || null,
                            });
                        });
                    });
                    return links;
                }""",
                    selector,
                )

            elif extract_type == "structured":
                # Extract structured data based on common patterns
                js_code = """(selector) => {
                    const elements = document.querySelectorAll(selector);
                    return Array.from(elements).map(el => {
                        // Try to extract structured data
                        const data = {};

                        // Look for common patterns
                        const title = el.querySelector('h1, h2, h3, h4, .title, [class*="title"]');
                        if (title) data.title = title.textContent.trim();

                        const description = el.querySelector('p, .description, [class*="description"], .summary');
                        if (description) data.description = description.textContent.trim();

                        const price = el.querySelector('.price, [class*="price"], [data-price]');
                        if (price) data.price = price.textContent.trim();

                        const image = el.querySelector('img');
                        if (image) data.image = { src: image.src, alt: image.alt };

                        const link = el.querySelector('a');
                        if (link) data.link = { href: link.href, text: link.textContent.trim() };

                        // Get all text if no structured data found
                        if (Object.keys(data).length === 0) {
                            data.text = el.textContent.trim();
                        }

                        return data;
                    });
                }"""
                extracted_data = await page.evaluate(js_code, selector)

            # Save to file if requested
            if output_file and extracted_data:
                full_path = context.resolve_workspace_path(output_file)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)

                with open(full_path, "w", encoding="utf-8") as f:
                    if output_file.endswith(".json"):
                        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
                    else:
                        if isinstance(extracted_data, list):
                            f.write("\n".join(str(item) for item in extracted_data))
                        else:
                            f.write(str(extracted_data))

            return {
                "success": True,
                "url": url,
                "selector": selector,
                "extract_type": extract_type,
                "data": extracted_data,
                "count": len(extracted_data) if isinstance(extracted_data, list) else 1,
                "output_file": output_file if output_file else None,
            }

        except Exception as e:
            return {"error": f"Error extracting content: {str(e)}"}


class WebContentExtractor:
    """
    An agentic web content extractor that intelligently uses DOM tools to find and extract specific content.

    This class implements a simple agent loop that:
    1. Examines the DOM structure
    2. Searches for relevant content
    3. Extracts the found content
    4. Returns structured results
    """

    def __init__(self, processing_context: ProcessingContext):
        from nodetool.providers.openai_provider import OpenAIProvider

        self.processing_context = processing_context
        self.provider = OpenAIProvider()
        self.model = "gpt-4o-mini"
        self.tools = [DOMExamineTool(), DOMSearchTool(), DOMExtractTool()]
        self.max_iterations = 5

    def _create_system_prompt(self, objective: str) -> str:
        """Create the system prompt for the agent."""
        return f"""You are a web content extraction agent. Your goal is to extract specific content from web pages using DOM tools.

Objective: {objective}

Available tools:
1. dom_examine - Examine DOM structure and elements
2. dom_search - Search for elements using text, CSS, attributes, etc.
3. dom_extract - Extract content from found elements

Strategy:
1. First, examine the page structure to understand the layout
2. Search for elements containing the desired content
3. Extract the content in the most appropriate format

IMPORTANT: When you find elements:
- Note their CSS paths, XPath, IDs, and classes from the tool results
- Use the most specific selector available (prefer IDs, then unique classes, then CSS paths)
- For extraction, use the exact selectors provided in the search/examine results
- If a search shows elements with cssPath or specificCssPath, use those for extraction

Always be systematic and thorough. Return structured data when possible."""

    async def extract_content(
        self, url: str, objective: str, selector_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract content from a web page based on the given objective.

        Args:
            url: The URL to extract content from
            objective: What content to extract (e.g., "main article text", "product prices", etc.)
            selector_hint: Optional CSS selector hint to guide the extraction

        Returns:
            Extracted content and metadata
        """
        # Initialize message history
        messages = [
            Message(role="system", content=self._create_system_prompt(objective)),
            Message(
                role="user",
                content=f"Extract the following from {url}: {objective}"
                + (
                    f"\nHint: Look for elements matching '{selector_hint}'"
                    if selector_hint
                    else ""
                ),
            ),
        ]

        result = {
            "url": url,
            "objective": objective,
            "iterations": 0,
            "tool_calls": [],
            "extracted_content": None,
            "error": None,
        }

        # Run the agent loop
        for iteration in range(self.max_iterations):
            result["iterations"] = iteration + 1

            try:
                # Get LLM response with tool calls
                response = await self.provider.generate_message(
                    messages=messages,
                    model=self.model,
                    tools=self.tools,
                    max_tokens=4096,
                )
                # Add assistant message to history
                messages.append(response)

                # If no tool calls, we're done
                if not response.tool_calls:
                    if response.content:
                        # The agent is providing a final answer
                        result["extracted_content"] = response.content
                    break

                # Process tool calls
                for tool_call in response.tool_calls:
                    result["tool_calls"].append(
                        {"name": tool_call.name, "args": tool_call.args}
                    )

                    # Execute the tool
                    tool_result = await self._execute_tool(tool_call)

                    # Add tool result to history
                    messages.append(
                        Message(
                            role="tool",
                            tool_call_id=tool_call.id,
                            name=tool_call.name,
                            content=json.dumps(tool_result, ensure_ascii=False),
                        )
                    )

                    # Check if we found what we're looking for
                    if tool_call.name == "dom_extract" and tool_result.get("success"):
                        result["extracted_content"] = tool_result.get("data")

            except Exception as e:
                result["error"] = str(e)
                break

        return result

    async def _execute_tool(self, tool_call: ToolCall) -> Any:
        """Execute a tool call and return the result."""
        for tool in self.tools:
            if tool.name == tool_call.name:
                return await tool.process(self.processing_context, tool_call.args)
        raise ValueError(f"Tool '{tool_call.name}' not found")


class AgenticBrowserTool(Tool):
    """
    A high-level browser tool that uses an agent to intelligently extract content.

    This tool combines the low-level DOM tools with an LLM agent to provide
    intelligent content extraction capabilities.
    """

    name = "agentic_browser"
    description = (
        "Intelligently extract specific content from web pages using an AI agent"
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to extract content from",
            },
            "objective": {
                "type": "string",
                "description": "What content to extract (e.g., 'main article text', 'product information', 'contact details')",
            },
            "selector_hint": {
                "type": "string",
                "description": "Optional CSS selector hint to guide extraction",
            },
            "output_file": {
                "type": "string",
                "description": "Optional file to save extracted content (workspace relative)",
            },
        },
        "required": ["url", "objective"],
    }
    example = """
    agentic_browser(
        url="https://example.com/article",
        objective="Extract the main article content including title and body text",
        output_file="article_content.json"
    )
    """

    def user_message(self, params: dict) -> str:
        url = params.get("url", "a page")
        objective = params.get("objective", "content")
        return f"Extracting {objective} from {url} using AI agent"

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        url = params.get("url")
        if not url:
            return {"error": "URL is required"}

        objective = params.get("objective")
        if not objective:
            return {"error": "objective is required"}

        selector_hint = params.get("selector_hint")
        output_file = params.get("output_file")

        # Create the extractor and run it
        extractor = WebContentExtractor(context)
        result = await extractor.extract_content(url, objective, selector_hint)

        # Save to file if requested
        if output_file and result.get("extracted_content"):
            full_path = context.resolve_workspace_path(output_file)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            with open(full_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "url": url,
                        "objective": objective,
                        "extracted_content": result["extracted_content"],
                        "metadata": {
                            "iterations": result["iterations"],
                            "tool_calls": result["tool_calls"],
                        },
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            result["output_file"] = output_file

        return result


if __name__ == "__main__":
    import asyncio

    # Import the actual ProcessingContext
    from nodetool.workflows.processing_context import ProcessingContext

    context = ProcessingContext()

    async def agentic_browser_example():
        print("\n=== Testing Agentic Browser Tool ===\n")

        url = "https://www.reddit.com/r/AI_Agents/comments/1lbvs2c/what_simple_ai_workflowsagentsautomations_use.json"

        browser_tool = BrowserTool()
        result = await browser_tool.process(
            context,
            {
                "url": url,
            },
        )
        print(f"Result: {json.dumps(result, indent=2)}")

        # # Test 1: Extract article content from Hacker News
        # print("Test 1: Extracting top stories from Hacker News...")
        # agentic_tool = AgenticBrowserTool()
        # result = await agentic_tool.process(
        #     context,
        #     {
        #         "url": "https://news.ycombinator.com",
        #         "objective": "Extract the titles and links of the top 5 stories on the front page",
        #     },
        # )
        # print(f"Result: {json.dumps(result, indent=2)}")

        # # Test 2: Extract specific content with selector hint
        # print("\n\nTest 2: Extracting article content with selector hint...")
        # result = await agentic_tool.process(
        #     context,
        #     {
        #         "url": "https://example.com",
        #         "objective": "Extract the main page heading and any example domain information",
        #         "selector_hint": "h1, p",
        #     },
        # )
        # print(f"Result: {json.dumps(result, indent=2)}")

        # # Test 3: Direct WebContentExtractor usage
        # print("\n\nTest 3: Using WebContentExtractor directly...")
        # extractor = WebContentExtractor(context)
        # result = await extractor.extract_content(
        #     url="https://www.wikipedia.org",
        #     objective="Find and extract the search box and main navigation links",
        # )
        # print(f"Extractor Result: {json.dumps(result, indent=2)}")

    asyncio.run(agentic_browser_example())
