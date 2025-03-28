#!/usr/bin/env python3
"""
Test script for Multi-Agent Coordination focused on Hacker News content.
"""

import asyncio
from nodetool.chat.agent import Agent
from nodetool.chat.providers import get_provider, Chunk
from nodetool.chat.tools.browser import BrowserTool
from nodetool.chat.workspace_manager import WorkspaceManager
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext


async def main():
    context = ProcessingContext()

    # 2. Initialize provider and model
    # provider = get_provider(Provider.Ollama)
    # model = "qwen2.5:14b"
    # provider = get_provider(Provider.Anthropic)
    # model = "claude-3-5-sonnet-20241022"
    provider = get_provider(Provider.OpenAI)
    model = "gpt-4o"

    # 3. Set up browser tool for accessing websites
    browser_tool = BrowserTool(context.workspace_dir)

    # 5. Create a Hacker News agent for collecting posts
    agent = Agent(
        name="Hacker News Agent",
        objective="""
        Browse http://news.ycombinator.com/ and fetch the top posts.
        Fetch the comments of each post.
        Return a summary of the top posts and the comments in the given schema.
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

    print(f"\nResults: {agent.results}")
    print(f"\nWorkspace: {context.workspace_dir}")


if __name__ == "__main__":
    asyncio.run(main())
