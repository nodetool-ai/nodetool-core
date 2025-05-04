#!/usr/bin/env python3
"""
Reddit Scraping Agent with Task Planner

This script creates a Reddit agent that:
1. Uses TaskPlanner to generate a dynamic plan for scraping Reddit
2. Uses GoogleSearchTool to find Reddit posts matching a search query
3. Uses BrowserTool to visit each URL and extract the content
4. Organizes and saves the results
"""

import asyncio

from nodetool.agents.agent import Agent
from nodetool.chat.dataframes import json_schema_for_dataframe
from nodetool.chat.providers import get_provider
from nodetool.agents.tools.browser_tools import BrowserTool, GoogleSearchTool
from nodetool.chat.providers.base import ChatProvider
from nodetool.metadata.types import ColumnDef, Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk


async def test_reddit_scraper_agent(
    provider: ChatProvider,
    model: str,
    reasoning_model: str,
    planning_model: str,
):
    context = ProcessingContext()
    search_agent = Agent(
        name="Reddit Search Agent",
        objective="""
        Search for Reddit posts in the r/n8n subreddit using Google Search
        Identify the customer painpoints.
        Group painpoints into themes.
        """,
        provider=provider,
        model=model,
        reasoning_model=reasoning_model,
        planning_model=planning_model,
        tools=[
            GoogleSearchTool(),
            BrowserTool(),
        ],
        output_type="markdown",
        output_schema={
            "type": "string",
        },
    )

    # 7. Execute each task in the plan
    async for item in search_agent.execute(context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    report = search_agent.get_results()
    print(report)


if __name__ == "__main__":

    asyncio.run(
        test_reddit_scraper_agent(
            provider=get_provider(Provider.OpenAI),
            model="gpt-4o-mini",
            planning_model="gpt-4o-mini",
            reasoning_model="gpt-4o-mini",
        )
    )
    # asyncio.run(
    #     test_reddit_scraper_agent(
    #         provider=get_provider(Provider.Gemini), model="gemini-2.0-flash"
    #     )
    # )

    # asyncio.run(
    #     test_reddit_scraper_agent(
    #         provider=get_provider(Provider.Anthropic),
    #         model="claude-3-5-sonnet-20241022",
    #     )
    # )
