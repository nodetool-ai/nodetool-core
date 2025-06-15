#!/usr/bin/env python3
"""
Instagram Trends Analysis Task

This script demonstrates using the SubTaskContext to analyze Instagram trends,
viral content, and emerging patterns. It uses search and browser tools to gather
comprehensive data about current Instagram trends, including hashtags, content themes,
and engagement patterns.

The task outputs structured JSON data that can be used for:
- Content strategy planning
- Social media marketing insights
- Trend analysis and reporting
- Competitive intelligence

Usage:
    python instagram_scraper_task.py
"""

import asyncio
from nodetool.agents.agent import Agent
from nodetool.agents.tools.browser_tools import AgenticBrowserTool
from nodetool.chat.providers import get_provider
from nodetool.agents.tools import BrowserTool, GoogleSearchTool
from nodetool.chat.providers.base import ChatProvider
from nodetool.metadata.types import Provider, Task, SubTask
from nodetool.agents.sub_task_context import SubTaskContext
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk, TaskUpdate
import json


async def test_instagram_scraper_task(
    provider: ChatProvider,
    model: str,
):
    # 1. Set up workspace directory
    context = ProcessingContext()

    # 3. Set up browser and search tools
    tools = [
        BrowserTool(),
        GoogleSearchTool(),
    ]

    # 5. Create a sample subtask
    agent = Agent(
        name="Instagram Trends Collection",
        objective="""
        Use google to identify top 5 trending hashtags on Instagram.    
        For each hashtag, use google to find one example post.
        For each post, use the browser tool to get the post details.
        Create a summary of the trends, hashtags, and viral content.
        Return all trends you can find.
        """,
        enable_data_contracts_phase=False,
        provider=provider,
        model=model,
        output_schema={
            "type": "object",
            "properties": {
                "trends": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "hashtag": {"type": "string"},
                            "description": {"type": "string"},
                            "example_post": {
                                "type": "object",
                                "properties": {
                                    "post_url": {"type": "string"},
                                    "caption": {"type": "string"},
                                    "likes": {"type": "integer"},
                                    "comments": {"type": "integer"},
                                },
                                "required": ["post_url", "caption", "likes", "comments"],
                            },
                        },
                        "required": ["hashtag", "description", "example_post"],
                    },
                },
            },
        },
    )

    async for event in agent.execute(context):
        if isinstance(event, Chunk):
            print(event.content, end="")

    print("\nSubTask Execution Successful!")
    print(
        json.dumps(
            agent.get_results(),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    asyncio.run(
        test_instagram_scraper_task(
            provider=get_provider(Provider.OpenAI), model="gpt-4.1-mini"
        )
    )

    # asyncio.run(
    #     test_instagram_scraper_task(
    #         provider=get_provider(Provider.Anthropic),
    #         model="claude-3-7-sonnet-20250219",
    #     )
    # )
