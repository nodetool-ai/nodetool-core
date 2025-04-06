#!/usr/bin/env python3
"""
Test script for an Email Retrieval agent using email tools with TaskPlanner.

This script creates an Email Retrieval agent with email tools and uses a TaskPlanner
to automatically generate and execute a task plan based on the objective.
It demonstrates how to set up an agent for email processing and summarization.
"""

import asyncio
import os
from pathlib import Path
from nodetool.chat.agent import Agent
from nodetool.chat.dataframes import json_schema_for_dataframe
from nodetool.chat.providers import get_provider
from nodetool.metadata.types import Provider, ColumnDef
from nodetool.chat.workspace_manager import WorkspaceManager
from nodetool.chat.task_planner import TaskPlanner
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.chat.tools.email import SearchEmailTool
from nodetool.workflows.types import Chunk


async def main():
    context = ProcessingContext()

    provider = get_provider(Provider.OpenAI)
    model = "gpt-4o"
    email_tools = [
        SearchEmailTool(context.workspace_dir),
    ]

    retrieval_agent = Agent(
        name="Email Retriever",
        objective="""
        Search for emails from AINews in subject from last 24 hours.
        Extract a list of news in the specified output format.
        """,
        provider=provider,
        model=model,
        tools=email_tools,
        output_schema=json_schema_for_dataframe(
            columns=[
                ColumnDef(
                    name="title",
                    data_type="string",
                ),
                ColumnDef(
                    name="summary",
                    data_type="string",
                ),
                ColumnDef(
                    name="url",
                    data_type="string",
                ),
            ]
        ),
    )

    async for item in retrieval_agent.execute(context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    print(f"\nResults: {retrieval_agent.results}")
    print(f"\nWorkspace: {context.workspace_dir}")


if __name__ == "__main__":
    asyncio.run(main())
