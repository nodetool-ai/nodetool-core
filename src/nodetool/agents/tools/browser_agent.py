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
from langchain_core.language_models import BaseChatModel
from browser_use import Agent, Browser, BrowserConfig
from dotenv import load_dotenv

from nodetool.common.environment import Environment
from nodetool.workflows.processing_context import ProcessingContext
from .base import Tool

# Load environment variables
load_dotenv()


class BrowserAgentTool(Tool):
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

    name = "browser_agent"
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

    def __init__(self, workspace_dir: str, model: BaseChatModel):
        """Initialize the BrowserAgentTool."""
        super().__init__(workspace_dir)
        self.llm = model

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        """
        Execute a browser agent task.
        """
        try:
            task = params.get("task")
            if not task:
                return {"error": "Task description is required"}

            browser_endpoint = Environment.get("BRIGHTDATA_SCRAPING_BROWSER_ENDPOINT")

            if browser_endpoint:
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
            # Create and run the agent
            agent = Agent(
                task=task,
                llm=self.llm,
                browser=browser,
                save_conversation_path=os.path.join(
                    self.workspace_dir, "browser_agent.log"
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
