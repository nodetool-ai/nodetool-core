#!/usr/bin/env python3
"""
Script for a LinkedIn job market research agent using BrowserTool with TaskPlanner.

This script creates a LinkedIn agent with the BrowserTool and GoogleSearchTool
and uses a TaskPlanner to automatically generate and execute a task plan based on the objective.
It demonstrates how to set up an agent to collect information about job trends, in-demand skills,
and industry growth areas on LinkedIn.
"""

import asyncio

from langchain_openai import ChatOpenAI

from nodetool.agents.agent import Agent
from nodetool.chat.providers import get_provider
from nodetool.agents.tools.browser_agent import BrowserAgentTool
from nodetool.common.environment import Environment
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk


async def main():
    context = ProcessingContext()

    provider = get_provider(Provider.OpenAI)
    model = "gpt-4o"
    # Alternatively, you can use Anthropic:
    # provider = get_provider(Provider.Anthropic)
    # model = "claude-3-7-sonnet-20250219"
    api_key = Environment.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    browser_agent_model = ChatOpenAI(model="gpt-4o", api_key=api_key)

    # 3. Set up browser and search tools
    tools = [
        BrowserAgentTool(
            workspace_dir=context.workspace_dir,
            model=browser_agent_model,
        ),
    ]

    # 4. Create LinkedIn job market research agent
    agent = Agent(
        name="LinkedIn Job Market Researcher",
        objective="""
        1. Go to the following URL:
        https://www.linkedin.com/jobs/search/?currentJobId=4185926087&f_C=11130470&geoId=92000000&origin=COMPANY_PAGE_JOBS_CLUSTER_EXPANSION
        2. Extract all the job postings on the page.
        """,
        enable_analysis_phase=False,
        enable_data_contracts_phase=False,
        provider=provider,
        model=model,
        tools=tools,
    )
    processing_context = ProcessingContext()

    async for item in agent.execute(processing_context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    print(f"\nWorkspace: {context.workspace_dir}")


if __name__ == "__main__":
    asyncio.run(main())
