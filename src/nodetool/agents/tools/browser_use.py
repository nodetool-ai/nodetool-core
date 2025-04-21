"""
Browser agent tool that uses browser_use under the hood.

This module provides a tool for running browser-based agents using the browser_use library.
The agent can perform complex web automation tasks like form filling, navigation, data extraction,
and multi-step workflows using natural language instructions.
"""

import os
from typing import Any
import asyncio
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from browser_use import Agent, Browser, BrowserConfig
from dotenv import load_dotenv

from nodetool.common.environment import Environment
from nodetool.workflows.processing_context import ProcessingContext
from .base import Tool

# Load environment variables
load_dotenv()


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

            if use_remote_browser:
                browser_endpoint = Environment.get(
                    "BRIGHTDATA_SCRAPING_BROWSER_ENDPOINT"
                )
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
