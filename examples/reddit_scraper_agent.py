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
from nodetool.chat.providers import get_provider, Chunk
from nodetool.chat.tools.browser import BrowserTool, GoogleSearchTool
from nodetool.metadata.types import Provider, Task
from nodetool.chat.workspace_manager import WorkspaceManager
from nodetool.chat.task_planner import TaskPlanner
from nodetool.workflows.processing_context import ProcessingContext

# Reddit-specific system prompt to guide the agent
REDDIT_SCRAPER_SYSTEM_PROMPT = """
You are a Reddit Scraping Agent designed to extract valuable content from Reddit posts.

Your capabilities:
1. Search for Reddit posts using Google Search with specific queries
2. Visit Reddit URLs to extract post content, comments, and metadata
3. Organize and summarize the collected information

When performing searches:
- Format queries to target Reddit content (include "site:reddit.com" when needed)
- Look for posts relevant to the user's objective
- Extract all Reddit post URLs from search results

When scraping Reddit posts:
- Extract the post title, content, author, timestamp, and vote information
- Identify and capture high-value comments and discussions
- Extract links to related posts or external resources mentioned in discussions
- Organize information in a structured format for analysis

Present your findings clearly with:
- Summary of key insights from each post
- Common themes or perspectives across posts
- Links to the most valuable content discovered

Always respect Reddit's structure when parsing content and prioritize extracting meaningful information.
"""


async def main():
    # 1. Set up workspace directory
    workspace_manager = WorkspaceManager()
    workspace_dir = workspace_manager.get_current_directory()
    print(f"Created workspace at: {workspace_dir}")

    # 2. Initialize provider and model
    # provider = get_provider(Provider.OpenAI)
    # model = "gpt-4o"
    # Alternative model options:
    # provider = get_provider(Provider.Anthropic)
    # model = "claude-3-7-sonnet-20250219"
    provider = get_provider(Provider.Ollama)
    model = "qwen2.5:14b"

    # 3. Set up the required tools
    tools = [
        GoogleSearchTool(workspace_dir=str(workspace_dir)),
        BrowserTool(workspace_dir=str(workspace_dir)),
    ]

    # 4. Create the Reddit scraping agent
    agent = Agent(
        name="Reddit Scraper",
        objective="Extract valuable content from Reddit posts on specific topics",
        description="An agent that discovers and scrapes Reddit posts to extract and organize content.",
        provider=provider,
        model=model,
        workspace_dir=str(workspace_dir),
        tools=tools,
        system_prompt=REDDIT_SCRAPER_SYSTEM_PROMPT,
        max_steps=15,
        max_subtask_iterations=3,
    )

    # 5. Get user input for the search topic
    search_topic = "Marketing automation tools"

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
