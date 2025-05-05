#!/usr/bin/env python3
"""
Wikipedia-style Research and Documentation Agent

This script demonstrates how to create a single agent that:
1. Researches a specific topic using web search and browsing
2. Creates a well-structured markdown documentation in Wikipedia style
3. Organizes information with proper sections and references
"""

import asyncio
from nodetool.agents.agent import Agent
from nodetool.chat.providers import get_provider
from nodetool.agents.tools import GoogleSearchTool, BrowserTool
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk


async def main():
    context = ProcessingContext()

    provider = get_provider(Provider.OpenAI)
    model = "gpt-4o-mini"

    # Set up tools for retrieval
    retrieval_tools = [
        GoogleSearchTool(),
        BrowserTool(),
    ]

    # Create the Wiki Creator agent
    agent = Agent(
        name="Wiki Creator Agent",
        objective="""
        Crawl https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)
        and create a comprehensive Wikipedia-style article about LLM fine-tuning.
        1. Research what linked pages are relevant to LLM fine-tuning
        2. Crawl the linked pages and extract the relevant information
        3. Create a well-structured markdown document
        """,
        provider=provider,
        model=model,
        tools=retrieval_tools,
        enable_analysis_phase=True,
    )

    async for item in agent.execute(context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    print(f"\nWorkspace: {context.workspace_dir}")
    print("Browse llm_fine_tuning.md to see the Wikipedia-style article.")


if __name__ == "__main__":
    asyncio.run(main())
