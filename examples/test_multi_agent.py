#!/usr/bin/env python3
"""
Test script for Multi-Agent Coordination using specialized agents.

This script demonstrates the use of MultiAgentCoordinator to orchestrate two specialized agents:
1. A Research Agent: Responsible for retrieving information from the web
2. A Summarization Agent: Processes and summarizes the retrieved information

This example shows how to:
1. Set up multiple specialized agents with different capabilities
2. Define their roles and coordination in a task plan
3. Have the MultiAgentCoordinator manage task dependencies and execution flow
4. Generate comprehensive research with information retrieval and summarization
"""

import asyncio
import os
from pathlib import Path

from nodetool.chat.agent import Agent, RETRIEVAL_SYSTEM_PROMPT
from nodetool.chat.multi_agent import MultiAgentCoordinator
from nodetool.chat.providers import get_provider, Chunk
from nodetool.chat.task_planner import DEFAULT_PLANNING_SYSTEM_PROMPT, TaskPlanner
from nodetool.chat.tools.browser import GoogleSearchTool, BrowserTool
from nodetool.chat.tools.workspace import ReadWorkspaceFileTool
from nodetool.chat.workspace_manager import WorkspaceManager
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext

SUMMARIZER_SYSTEM_PROMPT = """You are a specialized Summarization Agent. Your role is to:
1. Read research materials collected by other agents using the read_workspace_file tool
2. Analyze and identify key insights, trends, and important information
3. Create well-structured summaries that distill complex information into clear, concise content
4. Organize information in a logical hierarchy with appropriate headings and subheadings
5. Highlight the most significant findings and conclusions

Follow these guidelines:
- Focus on accuracy and objectivity in your summaries
- Preserve the key information while eliminating redundancy
- Maintain appropriate context when condensing information
- Use clear language and proper citation when referencing sources
- Structure your summaries with clear organization (headings, bullet points when appropriate)
"""


async def main():
    # 1. Set up workspace directory
    workspace_manager = WorkspaceManager()
    workspace_dir = workspace_manager.get_current_directory()
    print(f"Working in workspace at: {workspace_dir}")

    # 2. Initialize provider and model
    # provider = get_provider(Provider.Anthropic)
    # model = "claude-3-7-sonnet-20250219"
    # Uncomment to use OpenAI instead
    provider = get_provider(Provider.OpenAI)
    model = "gpt-4o"

    # 3. Set up tools for retrieval
    retrieval_tools = [
        GoogleSearchTool(str(workspace_dir)),
        BrowserTool(str(workspace_dir)),
    ]

    # 4. Create the research objective
    research_objective = "Research and create a fascinating guide to the world's most unusual sports and competitions, including their origins, rules, and cultural significance."

    # 5. Create a retrieval agent for gathering information
    retrieval_agent = Agent(
        name="Research Agent",
        objective="Research information about LLM architectures and their capabilities.",
        description="""
        You are a world-class researcher that can use the internet to find information about LLM architectures and their capabilities.
        You are also able to use the browser to search the web for information.
        """,
        provider=provider,
        model=model,
        workspace_dir=str(workspace_dir),
        tools=retrieval_tools,
        system_prompt=RETRIEVAL_SYSTEM_PROMPT,
    )

    # 6. Create a summarization agent for processing collected information
    summary_agent = Agent(
        name="Summary Agent",
        objective="Create well-structured summaries of the collected research on unusual sports and competitions.",
        description="""
        You are a world-class writer that can analyze collected information and create comprehensive summaries.
        """,
        provider=provider,
        model=model,
        workspace_dir=str(workspace_dir),
        tools=[ReadWorkspaceFileTool(str(workspace_dir))],
        system_prompt=SUMMARIZER_SYSTEM_PROMPT,
    )
    # Create planner with retrieval tools
    planner = TaskPlanner(
        provider=provider,
        model=model,
        objective=research_objective,
        workspace_dir=str(workspace_dir),
        tools=retrieval_tools,
        agents=[retrieval_agent, summary_agent],
        max_research_iterations=1,
    )

    # 7. Create a multi-agent coordinator
    coordinator = MultiAgentCoordinator(
        provider=provider,
        planner=planner,
        workspace_dir=str(workspace_dir),
        agents=[retrieval_agent, summary_agent],
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
