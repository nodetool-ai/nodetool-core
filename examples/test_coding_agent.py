#!/usr/bin/env python3
"""
Test script for Code Interpreter Agent.

This script demonstrates the use of an agent that can analyze, explain, and modify code.
The Code Interpreter Agent can:
1. Understand and explain existing code
2. Suggest improvements or fixes
3. Generate new code based on requirements
4. Execute and test code to validate solutions

This example shows how to:
1. Set up a code interpreter agent with appropriate tools
2. Define code-related tasks and objectives
3. Process and interpret the agent's responses
4. Save generated code to the workspace
"""

import asyncio
import os
from nodetool.agents.agent import Agent
from nodetool.agents.tools.code_tools import ExecutePythonTool
from nodetool.agents.tools.http_tools import DownloadFileTool
from nodetool.chat.providers.base import ChatProvider
from nodetool.chat.providers.openai_provider import OpenAIProvider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk


async def test_coding_agent(
    provider: ChatProvider, model: str, planning_model: str, reasoning_model: str
):
    context = ProcessingContext()

    code_tools = [
        ExecutePythonTool(),
        DownloadFileTool(),
    ]

    agent = Agent(
        name="Code Agent",
        objective="""
        Perform an exploratory analysis of the Iris dataset (petal flowers).

        Specifically:
        1. Download the Iris dataset CSV file from the URL: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
        2. Save the downloaded file as 'iris.csv' in the workspace.
        3. Load the 'iris.csv' file. Note: This file does not have headers. The columns are: sepal_length, sepal_width, petal_length, petal_width, class.
        4. Plot the data distributions using seaborn and create a markdown report.
        5. Generate a markdown report with the results, referencing the plots with relative paths.
        """,
        enable_analysis_phase=True,
        enable_data_contracts_phase=True,
        provider=provider,
        model=model,
        planning_model=planning_model,
        reasoning_model=reasoning_model,
        tools=code_tools,
        output_type="markdown",
    )

    async for item in agent.execute(context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    print("\n" + "=" * 50)
    print("Final Results:")
    print(agent.results)
    print(f"\nWorkspace: {context.workspace_dir}")


if __name__ == "__main__":
    # asyncio.run(
    #     test_coding_agent(
    #         provider=AnthropicProvider(), model="claude-3-5-sonnet-20241022"
    #     )
    # )
    asyncio.run(
        test_coding_agent(
            provider=OpenAIProvider(),
            model="gpt-4o",
            planning_model="o3",
            reasoning_model="o3",
        )
    )
    # asyncio.run(test_coding_agent(provider=GeminiProvider(), model="gemini-2.0-flash"))
