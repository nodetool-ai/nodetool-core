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
from nodetool.chat.agent import Agent
from nodetool.chat.providers import get_provider
from nodetool.chat.tools.browser import DownloadFileTool, GoogleSearchTool
from nodetool.chat.tools.system import CodeInterpreterTool
from nodetool.chat.workspace_manager import WorkspaceManager
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk


async def main():
    context = ProcessingContext()

    # 2. Initialize provider and model
    provider = get_provider(Provider.Gemini)
    model = "gemini-2.0-flash"

    # Alternatives:
    # provider = get_provider(Provider.Anthropic)
    # model = "claude-3-5-sonnet-20241022"

    # 3. Set up tools for code interpretation
    code_tools = [
        GoogleSearchTool(context.workspace_dir),
        DownloadFileTool(context.workspace_dir),
    ]

    # 4. Create a code interpreter agent
    agent = Agent(
        name="Code Interpreter Agent",
        objective="""
        Analyze the GDP data of the United States and China.
        
        Specifically:
        1. Search for and identify a dataset for GDP data of the United States and China
        2. Download the CSV file to the workspace
        3. Perform analysis to discover meaningful patterns or insights
        4. Create at least one visualization to illustrate your findings
        5. Summarize the key insights in a clear, concise manner
        
        The solution should demonstrate:
        - Effective data retrieval techniques
        - Basic data cleaning and preprocessing
        - Meaningful statistical analysis
        - Clear visualization of results
        - Thoughtful interpretation of the findings
        
        Be sure to handle potential issues like missing data or formatting problems in the CSV file.
        """,
        provider=provider,
        model=model,
        tools=code_tools,
        output_type="markdown",
    )

    # 5. Execute the agent
    processing_context = ProcessingContext()
    async for item in agent.execute(processing_context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    print("\n" + "=" * 50)
    print("Final Results:")
    print(agent.results)
    print(f"\nWorkspace: {context.workspace_dir}")


if __name__ == "__main__":
    asyncio.run(main())
