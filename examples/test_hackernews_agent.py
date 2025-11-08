#!/usr/bin/env python3
"""
Test script for Multi-Agent Coordination focused on Hacker News content.
"""

import asyncio
import json
from nodetool.agents.agent import Agent
from nodetool.providers import get_provider
from nodetool.agents.tools.browser_tools import BrowserTool
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk
from nodetool.runtime.resources import ResourceScope
import dotenv

dotenv.load_dotenv()


async def run_hn_agent():
    context = ProcessingContext()

    # 2. Initialize provider and model
    # provider = await get_provider(Provider.Ollama)
    # model = "qwen2.5:14b"
    # provider = await get_provider(Provider.Anthropic)
    # model = "claude-3-5-sonnet-20241022"
    provider = await get_provider(Provider.OpenAI)
    model = "gpt-4o-mini"
    # provider = await get_provider(Provider.Ollama)
    # model = "qwen3:14b"

    # 3. Set up browser tool for accessing websites
    browser_tool = BrowserTool()

    # 5. Create a Hacker News agent for collecting posts
    agent = Agent(
        name="Hacker News Agent",
        objective="""
        Scrape top 5 posts from news.ycombinator.com with their top 3 comments each.
        Focus on post titles, URLs, and comment content with authors.
        """,
        provider=provider,
        model=model,
        tools=[browser_tool],
        output_schema={
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "posts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "url": {"type": "string"},
                            "top_comments": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "text": {"type": "string"},
                                        "author": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    )

    async for item in agent.execute(context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    print(f"\nResults: {json.dumps(agent.results, indent=2)}")
    print(f"\nWorkspace: {context.workspace_dir}")


async def main():
    async with ResourceScope():
        await run_hn_agent()


if __name__ == "__main__":
    asyncio.run(main())
