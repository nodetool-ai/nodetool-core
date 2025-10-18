"""
Example demonstrating dynamic subtask addition in the NodeTool agent system.

This example shows how agents can dynamically add new subtasks during execution
using the AddSubtaskTool and ListSubtasksTool.
"""

import asyncio
from nodetool.providers.openai_provider import OpenAIProvider
from nodetool.agents.task_executor import TaskExecutor
from nodetool.agents.tools import BrowserTool, GoogleSearchTool
from nodetool.metadata.types import Task, SubTask
from nodetool.workflows.processing_context import ProcessingContext
import dotenv
from rich.console import Console

dotenv.load_dotenv()

console = Console()


async def demonstrate_dynamic_subtasks():
    """
    Demonstrate how agents can dynamically add subtasks during execution.

    The agent is given an initial task and has access to tools that allow it to:
    1. Add new subtasks when needed (AddSubtaskTool)
    2. List current subtasks (ListSubtasksTool)
    3. Browse the web and search (BrowserTool, GoogleSearchTool)
    """

    # Create an initial task with a simple starting subtask
    task = Task(
        id="research_task",
        title="Research AI Agent Frameworks",
        description="Research and compare different AI agent frameworks, then create a summary",
        subtasks=[
            SubTask(
                id="initial_research",
                content=(
                    "Research AI agent frameworks. "
                    "If you find multiple interesting frameworks, use the add_subtask tool "
                    "to create additional subtasks for deeper research on each one."
                ),
                output_schema="""{
                    "type": "object",
                    "properties": {
                        "frameworks_found": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "initial_findings": {"type": "string"}
                    }
                }""",
                max_tool_calls=20,
            )
        ],
    )

    # Create processing context
    context = ProcessingContext()

    # Set up provider and tools
    provider = OpenAIProvider()
    model = "gpt-4o-mini"

    tools = [
        GoogleSearchTool(),
        BrowserTool(),
    ]

    console.print("[bold green]Starting Dynamic Subtask Demonstration[/bold green]")
    console.print(f"[bold]Initial Task:[/bold] {task.title}")
    console.print(f"[bold]Initial Subtasks:[/bold] {len(task.subtasks)}\n")

    # Create task executor
    # Note: AddSubtaskTool and ListSubtasksTool are automatically added
    executor = TaskExecutor(
        provider=provider,
        model=model,
        processing_context=context,
        tools=tools,
        task=task,
        max_steps=20,
        max_subtask_iterations=15,
        parallel_execution=False,
    )

    # Execute the task
    console.print("[bold yellow]Executing task...[/bold yellow]\n")

    async for event in executor.execute_tasks(context):
        # You can process events here if needed
        pass

    # Display results
    console.print("\n[bold green]Task Execution Complete![/bold green]")
    console.print(f"[bold]Final Subtask Count:[/bold] {len(task.subtasks)}")

    if len(task.subtasks) > 1:
        console.print(
            f"\n[bold cyan]The agent dynamically added {len(task.subtasks) - 1} new subtask(s)![/bold cyan]\n"
        )

    # Show all subtasks and their completion status
    console.print("[bold]All Subtasks:[/bold]")
    for i, subtask in enumerate(task.subtasks, 1):
        status = "✓" if subtask.completed else "✗"
        console.print(f"  {i}. [{status}] {subtask.id}: {subtask.content[:100]}...")

    # Show results from the processing context
    console.print("\n[bold]Subtask Results:[/bold]")
    for subtask in task.subtasks:
        result = context.get(subtask.id)
        if result:
            console.print(f"\n[bold cyan]{subtask.id}:[/bold cyan]")
            console.print(f"  {str(result)[:300]}...")


async def demonstrate_with_explicit_subtask_addition():
    """
    Example showing explicit subtask addition during execution.

    This demonstrates a simpler case where we programmatically add subtasks.
    """

    # Create a basic task
    task = Task(
        id="analysis_task",
        title="Multi-Step Analysis",
        description="Analyze a topic in multiple stages",
        subtasks=[
            SubTask(
                id="stage_1",
                content="Perform initial analysis",
                output_schema='{"type": "object", "properties": {"findings": {"type": "string"}}}',
                max_tool_calls=10,
            )
        ],
    )

    # After the task is created, we can programmatically add more subtasks
    # This simulates what the agent does internally with AddSubtaskTool
    from nodetool.agents.tools.task_tools import AddSubtaskTool

    context = ProcessingContext()
    add_tool = AddSubtaskTool(task=task)

    # Add a dependent subtask
    result = await add_tool.process(
        context,
        {
            "content": "Perform detailed analysis based on stage 1 findings",
            "input_tasks": ["stage_1"],
            "max_tool_calls": 15,
        },
    )

    console.print(
        f"\n[bold green]Added new subtask:[/bold green] {result['subtask_id']}"
    )
    console.print(f"Total subtasks now: {len(task.subtasks)}")

    # The TaskExecutor will now execute both subtasks in order
    # (stage_2 will wait for stage_1 to complete due to the dependency)


if __name__ == "__main__":
    console.print(
        "[bold blue]NodeTool Dynamic Subtask Example[/bold blue]\n",
        "=" * 60,
        "\n",
    )

    # Run the main demonstration
    asyncio.run(demonstrate_dynamic_subtasks())

    # Optionally run the programmatic example
    # asyncio.run(demonstrate_with_explicit_subtask_addition())
