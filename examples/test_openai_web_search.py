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
from nodetool.chat.agent import Agent
from nodetool.chat.providers import get_provider, Chunk
from nodetool.chat.tools.openai_web_search import OpenAIWebSearchTool
from nodetool.chat.workspace_manager import WorkspaceManager
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext


async def main():
    # 1. Set up workspace directory
    workspace_manager = WorkspaceManager()
    workspace_dir = workspace_manager.get_current_directory()

    # 2. Initialize provider and model
    # provider = get_provider(Provider.Anthropic)
    # model = "claude-3-5-sonnet-20241022"
    provider = get_provider(Provider.OpenAI)
    model = "gpt-4o"
    # provider = get_provider(Provider.Ollama)
    # model = "qwen2.5:0.5b"
    # model = "MFDoom/deepseek-r1-tool-calling:1.5b"

    # provider = get_provider(Provider.Ollama)
    # model = "driaforall/tiny-agent-a:3b"
    # model = "qwen2.5:14b"

    # 3. Set up tools for retrieval
    retrieval_tools = [
        OpenAIWebSearchTool(str(workspace_dir)),
    ]

    # 5. Create a retrieval agent for gathering information
    agent = Agent(
        name="Research Agent",
        objective="""
        Research the competitive landscape of AI code tools in 2025.
        Such as Cursor, Windsurf, Copilot, asn open source alternatives.
        1. Use multiple openai web searches to identify a list of AI code assistant tools
        2. Summarize the findings in comprehensive markdown document.
        """,
        provider=provider,
        model=model,
        tools=retrieval_tools,
        output_type="markdown",
    )
    # 8. Solve the problem using the multi-agent coordinator
    processing_context = ProcessingContext()
    async for item in agent.execute(processing_context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    print(agent.results)
    print(f"\nWorkspace: {workspace_dir}")


if __name__ == "__main__":
    asyncio.run(main())
