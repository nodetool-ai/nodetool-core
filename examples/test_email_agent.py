#!/usr/bin/env python3
"""
Test script for an Email Retrieval agent using email tools with TaskPlanner.

This script creates an Email Retrieval agent with email tools and uses a TaskPlanner
to automatically generate and execute a task plan based on the objective.
It demonstrates how to set up an agent for email processing and summarization.
"""

import asyncio
from nodetool.agents.agent import Agent
from nodetool.providers import get_provider
from nodetool.metadata.types import Provider
from nodetool.ui.console import AgentConsole
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.tools.email_tools import SearchEmailTool


async def main():
    context = ProcessingContext()

    provider = get_provider(Provider.HuggingFaceCerebras)
    model = "openai/gpt-oss-120b"
    email_tools = [
        SearchEmailTool(),
    ]

    retrieval_agent = Agent(
        name="Email Retriever",
        objective="""
        Search for emails with AI in subject from last 2 days.
        Summarize the content of the emails in a markdown format.
        """,
        provider=provider,
        model=model,
        tools=email_tools,
        display_manager=AgentConsole(),
        enable_data_contracts_phase=False,
    )

    async for item in retrieval_agent.execute(context):
        pass

    print(f"\nResults: {retrieval_agent.results}")
    print(f"\nWorkspace: {context.workspace_dir}")


if __name__ == "__main__":
    asyncio.run(main())
