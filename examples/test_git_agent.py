#!/usr/bin/env python3
"""
Test script for Git Agent.

This script demonstrates an agent that can analyze a Git repository,
propose commits for changes, and perform a dry-run of the commit.
The Git Agent can:
1. Inspect the Git status to find changed files.
2. Generate a commit message for a set of changes.
3. Use Git tools to stage files and simulate a commit.

This example shows how to:
1. Set up an agent with Git tools.
2. Define a Git-related task and objective.
3. Process and interpret the agent's actions and results.
"""

import asyncio
from nodetool.agents.agent import Agent
from nodetool.agents.tools.git_tools import GitCommitTool, GitCheckoutTool
from nodetool.providers.base import BaseProvider
from nodetool.providers.huggingface_provider import HuggingFaceProvider
from nodetool.providers.openai_provider import OpenAIProvider
from nodetool.ui.console import AgentConsole
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk


async def test_git_agent(
    provider: BaseProvider,
    model: str,
):
    context = ProcessingContext()

    git_tools = [
        GitCommitTool(repo_path=context.workspace_dir),
        GitCheckoutTool(repo_path=context.workspace_dir),
    ]

    agent = Agent(
        name="Git Agent",
        objective="""
        Checkout the repo https://github.com/nodetool-ai/nodetool
        Improve the CLAUDE.md file.
        Commit the changes.
        """,
        enable_analysis_phase=False,
        enable_data_contracts_phase=False,
        provider=provider,
        model=model,
        tools=git_tools,
        display_manager=AgentConsole(),
    )

    print(f"Starting Git Agent execution in workspace: {context.workspace_dir}")
    print("Objective: Analyze git status and propose a dry-run commit for all changes.")
    print("-" * 50)

    async for item in agent.execute(context):
        pass

    print("\n" + "=" * 50)
    print("Git Agent Final Results/Summary:")
    if agent.results:
        print(agent.results)
    else:
        print(
            "No explicit results returned by the agent. Review the execution log above for details."
        )

    print(f"\nWorkspace used by the agent: {context.workspace_dir}")


if __name__ == "__main__":
    asyncio.run(
        test_git_agent(
            provider=HuggingFaceProvider("cerebras"),  # pyright: ignore[reportCallIssue]
            model="openai/gpt-oss-120b",
        )
    )
