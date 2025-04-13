import asyncio
import os
import json
from pathlib import Path
from rich.console import Console

from nodetool.chat.providers.anthropic_provider import AnthropicProvider
from nodetool.chat.providers.base import ChatProvider
from nodetool.chat.providers.gemini_provider import GeminiProvider
from nodetool.chat.providers.openai_provider import OpenAIProvider
from nodetool.agents.task_planner import TaskPlanner
from nodetool.agents.tools.base import Tool
from nodetool.agents.tools.browser import BrowserTool
from nodetool.agents.tools.google import GoogleGroundedSearchTool

# Create a console for rich output
console = Console()


async def test_task_planner(provider: ChatProvider, model: str):
    # Configure test parameters
    workspace_dir = "/tmp/nodetool-test"  # Create a test workspace directory
    os.makedirs(workspace_dir, exist_ok=True)

    # Sample objective
    objective = "Search for the best recipes for chicken wings and extract the ingredients and instructions for each recipe"

    # Sample input files (if any)
    input_files = []

    # Optional: Create test tools if needed
    tools = [
        GoogleGroundedSearchTool(workspace_dir),
        BrowserTool(workspace_dir),
    ]

    console.print(
        f"[bold green]Testing TaskPlanner with objective:[/bold green] {objective}"
    )
    console.print("[bold]Configuration:[/bold]")
    console.print(f"  - Workspace: {workspace_dir}")
    console.print(f"  - Model: {model}")
    console.print(f"  - Input files: {input_files}")

    # Create TaskPlanner instance with different configurations for testing
    planner = TaskPlanner(
        provider=provider,
        model=model,
        objective=objective,
        workspace_dir=workspace_dir,
        tools=tools,
        input_files=input_files,
        enable_analysis_phase=False,
        enable_data_contracts_phase=True,
        use_structured_output=True,
        verbose=True,
        output_schema={
            "type": "object",
            "properties": {
                "recipes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "image_url": {"type": "string"},
                            "ingredients": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "quantity": {"type": "string"},
                                        "unit": {"type": "string"},
                                    },
                                },
                            },
                            "instructions": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    )

    console.print("\n[bold yellow]Starting plan generation...[/bold yellow]")

    try:
        # Generate the task plan
        task = await planner.create_task(objective)

        # Display the generated plan
        console.print("[bold green]Plan generated successfully![/bold green]")
        console.print("\n[bold]Generated Task Plan:[/bold]")

        console.print(f"[bold cyan]Task Title:[/bold cyan] {task.title}")
        console.print(
            f"[bold cyan]Number of Subtasks:[/bold cyan] {len(task.subtasks)}"
        )

        for i, subtask in enumerate(task.subtasks, 1):
            console.print(f"\n[bold magenta]Subtask {i}:[/bold magenta]")
            console.print(f"Content: {subtask.content}")
            console.print(f"Output File: {subtask.output_file}")
            console.print(f"Input Files: {subtask.input_files}")
            console.print(f"Output Type: {subtask.output_type}")

            # Print output schema in a readable format if it's complex
            if subtask.output_type == "json":
                console.print("Output Schema:")
                console.print(json.dumps(subtask.output_schema, indent=2))
            else:
                console.print(f"Output Schema: {subtask.output_schema}")

    except Exception as e:
        console.print(f"[bold red]Error during plan generation:[/bold red] {str(e)}")
        raise


# Run the test
if __name__ == "__main__":
    asyncio.run(
        test_task_planner(
            provider=AnthropicProvider(), model="claude-3-5-sonnet-20241022"
        )
    )
    asyncio.run(test_task_planner(provider=OpenAIProvider(), model="gpt-4o"))
    asyncio.run(test_task_planner(provider=GeminiProvider(), model="gemini-2.0-flash"))
