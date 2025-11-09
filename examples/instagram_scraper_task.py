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
from nodetool.providers import get_provider
from nodetool.agents.tools import BrowserTool, GoogleSearchTool
from nodetool.providers.base import BaseProvider
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk
from nodetool.runtime.resources import ResourceScope
import json


async def test_instagram_scraper_task(
    provider: BaseProvider,
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
        enable_analysis_phase=True,
        enable_data_contracts_phase=True,
        provider=provider,
        model=model,
        tools=tools,
        output_schema={
            "type": "object",
            "properties": {
                "trends": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "hashtag": {"type": "string"},
                            "description": {"type": "string"}
                        },
                        "required": ["hashtag", "description"],
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


async def main():
    async with ResourceScope():
        await test_instagram_scraper_task(
            provider=await get_provider(Provider.HuggingFaceCerebras),
            model="openai/gpt-oss-120b",
        )


if __name__ == "__main__":
    asyncio.run(main())
