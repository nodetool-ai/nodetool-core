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
from nodetool.chat.multi_agent import MultiAgentCoordinator
from nodetool.chat.providers import get_provider, Chunk
from nodetool.chat.task_planner import TaskPlanner
from nodetool.chat.tools.browser import GoogleSearchTool, BrowserTool
from nodetool.chat.tools.workspace import ReadWorkspaceFileTool
from nodetool.chat.workspace_manager import WorkspaceManager
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext

SUMMARIZER_SYSTEM_PROMPT = """You are a specialized Summarization Agent for AI industry intelligence. Your role is to:
"""


async def main():
    context = ProcessingContext()

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
        GoogleSearchTool(context.workspace_dir),
        BrowserTool(context.workspace_dir),
    ]

    # 5. Create a retrieval agent for gathering information
    agent = Agent(
        name="Research Agent",
        objective="""
        Research the competitive landscape of AI code assistant tools.
        1. Use google search and browser to identify a list of AI code assistant tools
        2. For each tool, identify the following information:   
            - Name of the tool
            - Description of the tool
            - Key features of the tool
            - Pricing information
            - User reviews
            - Comparison with other tools
        3. Summarize the findings in a table format
        """,
        provider=provider,
        model=model,
        tools=retrieval_tools,
        output_schema={
            "type": "object",
            "properties": {
                "tools": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "key_features": {"type": "string"},
                            "pricing": {"type": "string"},
                            "user_reviews": {"type": "string"},
                            "comparison_with_other_tools": {"type": "string"},
                        },
                    },
                },
            },
        },
    )
    async for item in agent.execute(context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    print(f"\nWorkspace: {context.workspace_dir}")
    print(f"Results: {agent.results}")


if __name__ == "__main__":
    asyncio.run(main())
