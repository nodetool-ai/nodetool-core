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
import json
from pathlib import Path

from rich.console import Console

from nodetool.agents.sub_task_context import SubTaskContext
from nodetool.agents.tools.openai_tools import OpenAIWebSearchTool
from nodetool.metadata.types import SubTask, Task
from nodetool.providers.openai_provider import OpenAIProvider
from nodetool.runtime.resources import ResourceScope
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk, TaskUpdate

# Create a console for rich output
console = Console()


async def run_web_search_example():
    # Configure test parameters
    context = ProcessingContext()
    workspace_dir = context.workspace_dir

    # Initialize chat provider
    provider = OpenAIProvider()
    model = "gpt-4o"

    # Create test tools
    tools = [
        OpenAIWebSearchTool(),
    ]

    # Create a processing context
    processing_context = ProcessingContext(workspace_dir=workspace_dir)

    # Create a sample task
    task = Task(
        title="Research AI Code Tools",
        description="Research and summarize the competitive landscape of AI code tools in 2025",
        subtasks=[],
    )

    # Create a sample subtask
    subtask = SubTask(
        content="Use web search to identify AI code assistant tools and summarize findings",
        output_file="ai_code_tools_summary.json",
        input_files=[],
        output_type="json",
        output_schema=json.dumps(
            {
                "type": "object",
                "properties": {
                    "tools": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "features": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
        ),
    )

    # Add the subtask to the task
    task.subtasks = [subtask]

    # Create the SubTaskContext
    subtask_context = SubTaskContext(
        task=task,
        subtask=subtask,
        processing_context=processing_context,
        tools=tools,
        model=model,
        provider=provider,
        max_iterations=10,
    )

    # Execute the subtask
    async for event in subtask_context.execute():
        if isinstance(event, Chunk):
            print(event.content, end="")
        elif isinstance(event, TaskUpdate):
            console.print(f"[green]Task Update:[/green] {event.event}")

    # Check if output file was created
    output_path = Path(workspace_dir) / subtask.output_file
    if output_path.exists():
        with open(output_path) as f:
            result = json.load(f)
        console.print("\n[bold green]SubTask Execution Successful![/bold green]")
        console.print("\n[bold]Output File Content:[/bold]")
        console.print(json.dumps(result, indent=2))
    else:
        console.print("\n[bold red]Output file was not created![/bold red]")


async def main():
    async with ResourceScope():
        await run_web_search_example()


if __name__ == "__main__":
    asyncio.run(main())
