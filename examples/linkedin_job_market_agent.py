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
    # provider = get_provider(Provider.OpenAI)
    # model = "gpt-4o"
    # Alternatively, you can use Anthropic:
    provider = get_provider(Provider.Anthropic)
    model = "claude-3-7-sonnet-20250219"

    # 3. Set up browser and search tools
    tools = [
        BrowserTool(workspace_dir=str(workspace_dir)),
        GoogleSearchTool(workspace_dir=str(workspace_dir)),
    ]

    # 4. Create LinkedIn job market research agent
    research_agent = Agent(
        name="LinkedIn Job Market Researcher",
        objective="Research current job market trends, in-demand skills, and industry growth on LinkedIn.",
        description="An agent that identifies and collects information about job trends and skill requirements across industries on LinkedIn.",
        provider=provider,
        model=model,
        workspace_dir=str(workspace_dir),
        tools=tools,
        system_prompt=RETRIEVAL_SYSTEM_PROMPT,
        max_steps=15,
        max_subtask_iterations=4,
    )

    analyst_agent = Agent(
        name="Job Market Trend Analyzer",
        objective="Analyze job market data collected from LinkedIn and create comprehensive reports.",
        description="""
        An agent that analyzes LinkedIn job market data and produces insightful summaries and forecasts.
        Contains one subtask to analyze the data and produce a report.
        """,
        provider=provider,
        model=model,
        workspace_dir=str(workspace_dir),
        tools=[],
        system_prompt="""
        You are a job market and industry trend analyst specializing in LinkedIn data.
        Your expertise is in identifying patterns in hiring practices, skill requirements, and industry growth areas.
        Create well-structured reports with clear categorization of trends by industry, job function, and required skills.
        Include actionable insights that could be valuable for job seekers, career changers, and workforce development.
        Pay special attention to emerging roles, in-demand certifications, and industry-specific hiring patterns.
        Identify geographic trends in hiring and remote work adoption across different sectors.
        """,
        max_steps=12,
        max_subtask_iterations=3,
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
        6. Geographic hiring hotspots and remote work adoption by industry
        7. Educational requirements and alternative credentials gaining acceptance
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

    print(f"\nCreating task plan for objective: {objective}")

    # Generate the task plan
    async for item in task_planner.create_plan():
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    # Get the generated task plan
    task_plan = task_planner.task_plan
    if not task_plan:
        print("\nFailed to create task plan")
        return

    print(f"\n\nGenerated task plan: {task_plan.title}")
    processing_context = ProcessingContext()

    # 6. Execute each task in the plan
    for i, task in enumerate(task_plan.tasks):
        print(f"\nExecuting task {i+1}/{len(task_plan.tasks)}: {task.title}")

        # Select the appropriate agent based on the task
        agent = (
            analyst_agent
            if any(
                keyword in task.title.lower()
                for keyword in ["analyze", "report", "summarize", "interpret"]
            )
            else research_agent
        )
        print(f"Using agent: {agent.name}")

        # Execute the task
        async for item in agent.execute_task(task, processing_context):
            if isinstance(item, Chunk):
                print(item.content, end="", flush=True)

    print(f"\nWorkspace: {workspace_dir}")


if __name__ == "__main__":
    asyncio.run(main())
