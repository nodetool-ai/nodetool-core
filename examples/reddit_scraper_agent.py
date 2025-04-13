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
from nodetool.agents.tools.browser import BrowserTool, GoogleSearchTool
from nodetool.chat.providers.base import ChatProvider
from nodetool.metadata.types import ColumnDef, Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk


async def test_reddit_scraper_agent(provider: ChatProvider, model: str):
    context = ProcessingContext()
    search_agent = Agent(
        name="Reddit Search Agent",
        objective="""
        Search for Reddit posts in the r/StableDiffusion subreddit using Google Search
        Return a list of URLs to the Reddit posts
        """,
        provider=provider,
        model=model,
        enable_analysis_phase=False,
        enable_data_contracts_phase=False,
        tools=[
            GoogleSearchTool(workspace_dir=str(context.workspace_dir)),
        ],
        output_schema={
            "type": "array",
            "items": {
                "type": "string",
            },
        },
    )
    # 7. Execute each task in the plan
    async for item in search_agent.execute(context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    image_urls = search_agent.get_results()
    print("Image URLs:")
    print(image_urls)

    image_scraper_agent = Agent(
        name="Image Scraper Agent",
        objective=f"""
        Visit each URL and extract the image tags using CSS img selector.
        Return a list of image URLs.
        Use the remote browser to visit the URLs.
        The URLs are:
        {image_urls}
        """,
        provider=provider,
        model=model,
        enable_analysis_phase=False,
        enable_data_contracts_phase=True,
        tools=[
            BrowserTool(workspace_dir=str(context.workspace_dir)),
        ],
        output_schema={
            "type": "array",
            "items": {
                "type": "string",
            },
        },
    )

    # 7. Execute each task in the plan
    async for item in image_scraper_agent.execute(context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    image_urls = image_scraper_agent.get_results()
    print("Image URLs:")
    print(image_urls)


if __name__ == "__main__":

    asyncio.run(
        test_reddit_scraper_agent(
            provider=get_provider(Provider.OpenAI), model="gpt-4o-mini"
        )
    )
    asyncio.run(
        test_reddit_scraper_agent(
            provider=get_provider(Provider.Gemini), model="gemini-2.0-flash"
        )
    )

    asyncio.run(
        test_reddit_scraper_agent(
            provider=get_provider(Provider.Anthropic),
            model="claude-3-5-sonnet-20241022",
        )
    )
