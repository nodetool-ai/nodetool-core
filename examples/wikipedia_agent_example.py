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
from nodetool.agents.tools.browser import GoogleSearchTool, BrowserTool
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk


async def main():
    context = ProcessingContext()

    provider = get_provider(Provider.Anthropic)
    model = "claude-3-5-sonnet-20241022"

    # Set up tools for retrieval
    retrieval_tools = [
        GoogleSearchTool(context.workspace_dir),
        BrowserTool(context.workspace_dir),
    ]

    # Create the Wiki Creator agent
    agent = Agent(
        name="Wiki Creator Agent",
        objective="""
        Create a comprehensive Wikipedia-style article about LLM fine-tuning.
        1. Research information about LLM fine-tuning using search and browsing
        2. Create a well-structured markdown document covering:
            - Introduction to LLM fine-tuning
            - Methodologies and techniques
            - Best practices and challenges
            - Applications and use cases
            - References and further reading
        3. Save the final document as 'llm_fine_tuning.md'
        """,
        provider=provider,
        model=model,
        tools=retrieval_tools,
    )

    async for item in agent.execute(context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    print(f"\nWorkspace: {context.workspace_dir}")
    print("Browse llm_fine_tuning.md to see the Wikipedia-style article.")


if __name__ == "__main__":
    asyncio.run(main())
