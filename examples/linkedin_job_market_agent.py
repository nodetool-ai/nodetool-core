#!/usr/bin/env python3
"""
Script for a LinkedIn job market research agent using BrowserTool with TaskPlanner.

This script creates a LinkedIn agent with the BrowserTool and GoogleSearchTool
and uses a TaskPlanner to automatically generate and execute a task plan based on the objective.
It demonstrates how to set up an agent to collect information about job trends, in-demand skills,
and industry growth areas on LinkedIn.
"""

import asyncio
import os
import json
from pathlib import Path

from nodetool.chat.agent import Agent, RETRIEVAL_SYSTEM_PROMPT
from nodetool.chat.multi_agent import MultiAgentCoordinator
from nodetool.chat.providers import get_provider, Chunk
from nodetool.chat.tools.browser import BrowserTool, GoogleSearchTool
from nodetool.metadata.types import Provider, Task
from nodetool.chat.workspace_manager import WorkspaceManager
from nodetool.chat.task_planner import TaskPlanner
from nodetool.workflows.processing_context import ProcessingContext


async def main():
    # 1. Set up workspace directory
    workspace_manager = WorkspaceManager()
    workspace_dir = workspace_manager.get_current_directory()
    print(f"Created workspace at: {workspace_dir}")

    # 2. Initialize provider and model
    provider = get_provider(Provider.OpenAI)
    model = "gpt-4o"
    # Alternatively, you can use Anthropic:
    # provider = get_provider(Provider.Anthropic)
    # model = "claude-3-7-sonnet-20250219"

    # 3. Set up browser and search tools
    tools = [
        BrowserTool(workspace_dir=str(workspace_dir)),
        GoogleSearchTool(workspace_dir=str(workspace_dir)),
    ]

    # 4. Create LinkedIn job market research agent
    research_agent = Agent(
        name="LinkedIn Job Market Researcher",
        objective="Research current job market trends, in-demand skills, and industry growth on LinkedIn.",
        description="""
        You are a world-class researcher that can use the internet to find information about job market trends, in-demand skills, and industry growth on LinkedIn.
        You are also able to use the browser to search the web for information.
        You rely only on verified authors and sources.
        """,
        provider=provider,
        model=model,
        workspace_dir=str(workspace_dir),
        tools=tools,
        system_prompt=RETRIEVAL_SYSTEM_PROMPT,
        max_steps=20,
    )

    analyst_agent = Agent(
        name="Job Market Trend Analyzer",
        objective="Analyze job market data collected from LinkedIn and create comprehensive reports.",
        description="""
        You are a job market and industry trend analyst specializing in LinkedIn data.
        Your expertise is in identifying patterns in hiring practices, skill requirements, and industry growth areas.
        """,
        provider=provider,
        model=model,
        workspace_dir=str(workspace_dir),
        tools=[],
        max_steps=20,
    )

    # 5. Create and use the TaskPlanner
    objective = """
        Research and identify current job market trends on LinkedIn to provide insights for career development and hiring strategies.
        Focus on identifying:
        1. Fast-growing job categories and roles across different industries (tech, healthcare, finance, etc.)
        2. In-demand technical and soft skills for various positions
        3. Industry-specific hiring patterns and company growth indicators
        4. Emerging job titles and roles that didn't exist 3-5 years ago
        5. Salary trends and compensation packages for different experience levels
        Save all retrieved information as organized files in the /workspace directory.
    """

    task_planner = TaskPlanner(
        provider=provider,
        model=model,
        objective=objective,
        workspace_dir=str(workspace_dir),
        tools=tools,
        agents=[research_agent, analyst_agent],
        max_research_iterations=3,
    )

    # 7. Create a multi-agent coordinator
    coordinator = MultiAgentCoordinator(
        provider=provider,
        planner=task_planner,
        workspace_dir=str(workspace_dir),
        agents=[research_agent, analyst_agent],
        max_steps=30,
    )

    # 8. Solve the problem using the multi-agent coordinator
    processing_context = ProcessingContext()
    async for item in coordinator.solve_problem(processing_context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)
    print(f"\nWorkspace: {workspace_dir}")


if __name__ == "__main__":
    asyncio.run(main())
