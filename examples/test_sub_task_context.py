import asyncio
import os
import json
from pathlib import Path
from rich.console import Console

from nodetool.chat.providers.gemini_provider import GeminiProvider
from nodetool.chat.providers.openai_provider import OpenAIProvider
from nodetool.agents.sub_task_context import SubTaskContext
from nodetool.agents.tools.base import Tool
from nodetool.agents.tools.browser import BrowserTool
from nodetool.agents.tools.google import GoogleGroundedSearchTool
from nodetool.metadata.types import Task, SubTask, ToolCall
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk, TaskUpdate

# Create a console for rich output
console = Console()


async def test_subtask_context():
    # Configure test parameters
    context = ProcessingContext()
    workspace_dir = context.workspace_dir

    # Sample objective
    objective = "Find information about healthy breakfast recipes"

    # Initialize chat provider (use your preferred provider)
    # provider = OpenAIProvider()
    # model = "gpt-4o"  # Use an appropriate model

    provider = GeminiProvider()
    model = "gemini-2.0-flash"

    # Optional: Create test tools if needed
    tools = [
        GoogleGroundedSearchTool(workspace_dir),
        BrowserTool(workspace_dir),
    ]

    console.print(
        f"[bold green]Testing SubTaskContext with objective:[/bold green] {objective}"
    )
    console.print("[bold]Configuration:[/bold]")
    console.print(f"  - Workspace: {workspace_dir}")
    console.print(f"  - Model: {model}")

    # Create a processing context (required for SubTaskContext)
    processing_context = ProcessingContext(workspace_dir=workspace_dir)

    # Create a sample task
    task = Task(
        title="Research Healthy Breakfast Options",
        description="Find and summarize information about healthy breakfast recipes",
        subtasks=[],  # We'll add the subtask directly to the SubTaskContext
    )

    # Create a sample subtask
    subtask = SubTask(
        content="Search Google for healthy breakfast recipes and create a summary of the top 5 options",
        output_file="breakfast_recipes.json",
        input_files=[],
        output_type="json",
        output_schema=json.dumps(
            {
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
        ),
    )

    # Add the subtask to the task
    task.subtasks = [subtask]

    console.print("\n[bold yellow]Creating SubTaskContext...[/bold yellow]")

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

    console.print("\n[bold yellow]Executing SubTask...[/bold yellow]")

    try:
        # Execute the subtask and process the results
        async for event in subtask_context.execute():
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

        # Check if output file was created
        output_path = Path(workspace_dir) / subtask.output_file
        if output_path.exists():
            with open(output_path, "r") as f:
                result = json.load(f)

            console.print("\n[bold green]SubTask Execution Successful![/bold green]")
            console.print("\n[bold]Output File Content:[/bold]")
            console.print(json.dumps(result, indent=2))
        else:
            console.print("\n[bold red]Output file was not created![/bold red]")

    except Exception as e:
        console.print(f"[bold red]Error during subtask execution:[/bold red] {str(e)}")
        raise


# Run the test
if __name__ == "__main__":
    asyncio.run(test_subtask_context())
