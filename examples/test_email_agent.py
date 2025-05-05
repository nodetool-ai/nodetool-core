#!/usr/bin/env python3
"""
Test script for an Email Retrieval agent using email tools with TaskPlanner.

This script creates an Email Retrieval agent with email tools and uses a TaskPlanner
to automatically generate and execute a task plan based on the objective.
It demonstrates how to set up an agent for email processing and summarization.
"""

import asyncio
from nodetool.agents.agent import Agent
from nodetool.chat.dataframes import json_schema_for_dataframe
from nodetool.chat.providers import get_provider
from nodetool.metadata.types import Provider, ColumnDef
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.tools.email_tools import SearchEmailTool
from nodetool.workflows.types import Chunk


async def main():
    context = ProcessingContext()

    provider = get_provider(Provider.OpenAI)
    model = "gpt-4o-mini"
    email_tools = [
        SearchEmailTool(),
    ]

    retrieval_agent = Agent(
        name="Email Retriever",
        objective="""
        Search for emails with AI in subject from last 7 days.
        Summarize the content of the emails in a markdown format.
        """,
        provider=provider,
        model=model,
        tools=email_tools,
        enable_data_contracts_phase=False,
        output_type="markdown",
    )

    async for item in retrieval_agent.execute(context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    print(f"\nResults: {retrieval_agent.results}")
    print(f"\nWorkspace: {context.workspace_dir}")


if __name__ == "__main__":
    asyncio.run(main())
