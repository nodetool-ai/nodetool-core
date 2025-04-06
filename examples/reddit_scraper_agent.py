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
import json
import os
from pathlib import Path

from nodetool.chat.agent import Agent
from nodetool.chat.dataframes import json_schema_for_dataframe
from nodetool.chat.providers import get_provider
from nodetool.chat.tools.browser import BrowserTool, GoogleSearchTool
from nodetool.metadata.types import ColumnDef, Provider, Task
from nodetool.chat.workspace_manager import WorkspaceManager
from nodetool.chat.task_planner import TaskPlanner
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk

# Reddit-specific system prompt to guide the agent
INSTRUCTIONS = """
Extract awesome images from the r/StableDiffusion subreddit

Your capabilities:
1. Search for Reddit posts in the r/StableDiffusion subreddit using Google Search
2. Visit Reddit URLs to extract images, post content, and metadata
3. Organize and collect the most impressive AI-generated images

When performing searches:
- Format queries to target r/StableDiffusion content (include "site:reddit.com/r/StableDiffusion" when needed)
- Look for posts showcasing high-quality AI-generated artwork and images
- Extract all Reddit post URLs from search results that contain images

When scraping Reddit posts:
- Extract the post title and direct image URLs
- Focus on posts with high upvotes and positive community reception
- Identify images with interesting prompts or techniques mentioned
- Look for images that demonstrate artistic quality or technical achievement

Present your findings as a collection of:
- Image URLs with their corresponding post titles
- Direct links to the original posts
- Brief descriptions of what makes each image impressive or unique

Always respect Reddit's structure when parsing content and prioritize finding the most visually impressive stable diffusion images.
"""


async def main():
    context = ProcessingContext()

    provider = get_provider(Provider.OpenAI)
    model = "gpt-4o"
    # Alternative model options:
    # provider = get_provider(Provider.Anthropic)
    # model = "claude-3-7-sonnet-20250219"
    # provider = get_provider(Provider.Ollama)
    # model = "qwen2.5:14b"

    tools = [
        GoogleSearchTool(workspace_dir=str(context.workspace_dir)),
        BrowserTool(workspace_dir=str(context.workspace_dir)),
    ]

    agent = Agent(
        name="Reddit Scraper",
        objective=INSTRUCTIONS,
        provider=provider,
        model=model,
        tools=tools,
        output_schema=json_schema_for_dataframe(
            columns=[
                ColumnDef(
                    name="title",
                    data_type="string",
                ),
                ColumnDef(
                    name="upvotes",
                    data_type="int",
                ),
                ColumnDef(
                    name="post_url",
                    data_type="string",
                ),
                ColumnDef(
                    name="description",
                    data_type="string",
                ),
                ColumnDef(
                    name="image_url",
                    data_type="string",
                ),
            ]
        ),
    )

    processing_context = ProcessingContext()

    # 7. Execute each task in the plan
    async for item in agent.execute(processing_context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    # 8. Print result summary
    print("\n\nTask execution completed.")
    for file in agent.get_results():
        print(file)


if __name__ == "__main__":
    asyncio.run(main())
