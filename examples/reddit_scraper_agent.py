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
from nodetool.chat.providers import get_provider
from nodetool.agents.tools import BrowserTool, GoogleSearchTool
from nodetool.chat.providers.base import ChatProvider
from nodetool.metadata.types import Provider
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
        You are a Reddit expert.
        You are given a task to search for Reddit posts in the r/n8n subreddit using Google Search.
        Compile a list of posts that are related to problems that customers are facing.
        Browse the posts and extract post content and comments as a json list.
        Analyze the posts and comments to find painpoints.
        Generate a report of the findings.
        """,
        provider=provider,
        model=model,
        reasoning_model=reasoning_model,
        planning_model=planning_model,
        enable_analysis_phase=True,
        enable_data_contracts_phase=True,
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
    print(context.workspace_dir)


if __name__ == "__main__":

    asyncio.run(
        test_reddit_scraper_agent(
            provider=get_provider(Provider.OpenAI),
            model="gpt-4o",
            planning_model="gpt-4o",
            reasoning_model="gpt-4o",
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
