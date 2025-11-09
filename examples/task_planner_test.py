"""
Task Planner Test Example

This example demonstrates the TaskPlanner and TaskExecutor capabilities.

Key Features Demonstrated:
1. Automatic task planning from objectives
2. Task execution with progress tracking
3. Inspecting generated plans before execution
"""

import asyncio
from nodetool.metadata.types import Provider
from nodetool.providers import get_provider
from nodetool.providers.huggingface_provider import HuggingFaceProvider
from rich.console import Console

from nodetool.providers.base import BaseProvider
from nodetool.agents.task_planner import TaskPlanner
from nodetool.agents.tools import (
    BrowserTool,
    GoogleSearchTool,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.runtime.resources import ResourceScope
import dotenv


dotenv.load_dotenv()

# Create a console for rich output
console = Console()


async def test_task_planner(provider: BaseProvider, model: str):
    # Sample objective that encourages dynamic subtask addition
    objective = """
    Search for the best recipes for chicken wings and extract the ingredients and instructions for each recipe.

    """

    # Optional: Create test tools if needed
    tools = [
        GoogleSearchTool(),
        BrowserTool(),
    ]

    console.print(
        f"[bold green]Testing TaskPlanner with objective:[/bold green] {objective.strip()}"
    )
    console.print("[bold]Configuration:[/bold]")
    console.print(f"  - Model: {model}")
    console.print("  - Dynamic subtasks: Disabled")

    context = ProcessingContext()
    # Create TaskPlanner instance with different configurations for testing
    planner = TaskPlanner(
        provider=provider,
        model=model,
        objective=objective,
        workspace_dir=context.workspace_dir,
        execution_tools=tools,
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
        async for chunk in planner.create_task(context, objective):
            pass

        # Display the generated plan
        console.print("[bold green]Plan generated successfully![/bold green]")
        console.print("\n[bold]Generated Task Plan:[/bold]")

        print(planner.task_plan.to_markdown())

    except Exception as e:
        console.print(f"[bold red]Error during plan generation:[/bold red] {str(e)}")
        raise


async def main():
    async with ResourceScope():
        await test_task_planner(
            provider=await get_provider(Provider.HuggingFaceCerebras),
            model="openai/gpt-oss-120b",
        )


if __name__ == "__main__":
    asyncio.run(main())
