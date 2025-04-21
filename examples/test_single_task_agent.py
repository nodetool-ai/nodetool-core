import asyncio
import os
import json
from pathlib import Path
from rich.console import Console

from nodetool.agents.agent import SingleTaskAgent
from nodetool.chat.providers.gemini_provider import GeminiProvider
from nodetool.chat.providers.openai_provider import OpenAIProvider
from nodetool.agents.sub_task_context import SubTaskContext
from nodetool.agents.tools.base import Tool
from nodetool.agents.tools.browser import BrowserTool
from nodetool.agents.tools.google import GoogleGroundedSearchTool
from nodetool.metadata.types import Task, SubTask, ToolCall
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk, TaskUpdate
from nodetool.chat.providers.base import ChatProvider

# Create a console for rich output
console = Console()


async def test_subtask_context(provider: ChatProvider, model: str):
    # Configure test parameters
    context = ProcessingContext()
    workspace_dir = context.workspace_dir

    # Sample objective
    objective = "Find information about healthy breakfast recipes"

    # Optional: Create test tools if needed
    tools = [
        GoogleGroundedSearchTool(),
        BrowserTool(use_readability=True),
    ]

    console.print(
        f"[bold green]Testing SubTaskContext with objective:[/bold green] {objective}"
    )
    console.print("[bold]Configuration:[/bold]")
    console.print(f"  - Workspace: {workspace_dir}")
    console.print(f"  - Model: {model}")

    # Create a processing context (required for SubTaskContext)
    processing_context = ProcessingContext(workspace_dir=workspace_dir)

    # Create a sample subtask
    agent = SingleTaskAgent(
        name="Healthy Breakfast Recipes",
        objective="Search Google for healthy breakfast recipes and create a summary of the top 5 options",
        input_files=[],
        output_type="json",
        provider=provider,
        model=model,
        tools=tools,
        output_schema={
            "type": "object",
            "properties": {
                "recipes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "ingredients": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "preparation_time": {"type": "string"},
                            "calories": {"type": "string"},
                            "description": {"type": "string"},
                        },
                    },
                },
            },
        },
    )

    console.print("\n[bold yellow]Running SingleTaskAgent...[/bold yellow]")

    try:
        # Execute the subtask and process the results
        async for event in agent.execute(processing_context):
            # Different types of events that can be yielded
            if isinstance(event, Chunk):
                print(event.content, end="")
            elif isinstance(event, ToolCall):
                console.print(
                    f"[magenta]Tool Call:[/magenta] {event.name} with args: {json.dumps(event.args)[:100]}..."
                )
            elif isinstance(event, TaskUpdate):
                console.print(f"[green]Task Update:[/green] {event.event}")
            else:
                console.print(f"[yellow]Other Event:[/yellow] {type(event)}")

        console.print("\n[bold green]SubTask Execution Successful![/bold green]")
        console.print("\n[bold]Results:[/bold]")
        console.print(json.dumps(agent.get_results(), indent=2))

    except Exception as e:
        console.print(f"[bold red]Error during agent execution:[/bold red] {str(e)}")
        raise


# Run the test
if __name__ == "__main__":
    asyncio.run(test_subtask_context(provider=OpenAIProvider(), model="gpt-4o"))
    asyncio.run(
        test_subtask_context(provider=GeminiProvider(), model="gemini-2.0-flash")
    )
