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
import os
from nodetool.agents.agent import Agent
from nodetool.agents.tools.git_tools import GitStatusTool, GitCommitTool, GitDiffTool
from nodetool.chat.providers.base import ChatProvider
from nodetool.chat.providers.openai_provider import OpenAIProvider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk


async def test_git_agent(
    provider: ChatProvider,
    model: str,
):
    context = ProcessingContext()

    git_tools = [
        GitDiffTool(repo_path="."),
        GitCommitTool(repo_path="."),
    ]

    agent = Agent(
        name="Git Commit Proposer Agent",
        objective="""
        You are an expert Git assistant. Your task is to analyze the current Git repository's status, group coherent changes, and propose a separate commit for each group.

        Specifically, you need to:
        1. Use the 'git_diff' tool to get a detailed list of all changes.
        2. Group them into coherent changes.
        3. For each group of coherent changes:
            a. Identify all file paths belonging to this group.
            b. Generate a concise and descriptive commit message that accurately summarizes ONLY the changes within this specific group.
            c. Use the 'git_commit' tool. The 'files' parameter should be the list of file paths for THIS group. The 'message' parameter should be the commit message you generated for THIS group.
            d. CRUCIALLY, ensure the 'dry_run' parameter is set to true for each commit, so no actual commits are made.
        4. You may need to call the 'git_commit' tool multiple times, once for each distinct group of changes you identify.
        """,
        enable_analysis_phase=False,
        enable_data_contracts_phase=False,
        provider=provider,
        model=model,
        tools=git_tools,
        output_type="markdown",  # Or text, depending on desired output format
    )

    print(f"Starting Git Agent execution in workspace: {context.workspace_dir}")
    print("Objective: Analyze git status and propose a dry-run commit for all changes.")
    print("-" * 50)

    async for item in agent.execute(context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    print("\n" + "=" * 50)
    print("Git Agent Final Results/Summary:")
    # The agent's results might contain a summary or the final proposed command/message.
    # Depending on the agent's internal logic and output handling.
    # For now, we'll print the raw results if any.
    if agent.results:
        print(agent.results)
    else:
        print(
            "No explicit results returned by the agent. Review the execution log above for details."
        )

    print(f"\nWorkspace used by the agent: {context.workspace_dir}")
    print(
        "Note: The agent was instructed to perform a DRY-RUN commit. No actual changes were committed to your Git repository."
    )


if __name__ == "__main__":
    asyncio.run(
        test_git_agent(
            provider=OpenAIProvider(),
            model="gpt-4o-mini",
        )
    )
