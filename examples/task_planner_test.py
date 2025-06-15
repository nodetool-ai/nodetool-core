import asyncio
from nodetool.chat.providers.openai_provider import OpenAIProvider
from rich.console import Console

from nodetool.chat.providers.base import ChatProvider
from nodetool.chat.providers.ollama_provider import OllamaProvider
from nodetool.agents.task_planner import TaskPlanner
from nodetool.agents.tools import (
    BrowserTool,
    GoogleSearchTool,
)
from nodetool.workflows.processing_context import ProcessingContext
import dotenv


dotenv.load_dotenv()

# Create a console for rich output
console = Console()


async def test_task_planner(provider: ChatProvider, model: str):
    # Sample objective
    objective = "Search for the best recipes for chicken wings and extract the ingredients and instructions for each recipe"

    # Optional: Create test tools if needed
    tools = [
        GoogleSearchTool(),
        BrowserTool(),
    ]

    console.print(
        f"[bold green]Testing TaskPlanner with objective:[/bold green] {objective}"
    )
    console.print("[bold]Configuration:[/bold]")
    console.print(f"  - Model: {model}")

    context = ProcessingContext()
    # Create TaskPlanner instance with different configurations for testing
    planner = TaskPlanner(
        provider=provider,
        model=model,
        objective=objective,
        workspace_dir=context.workspace_dir,
        execution_tools=tools,
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
        async for chunk in planner.create_task(context, objective):
            pass

        # Display the generated plan
        console.print("[bold green]Plan generated successfully![/bold green]")
        console.print("\n[bold]Generated Task Plan:[/bold]")

        print(planner.task_plan.to_markdown())

    except Exception as e:
        console.print(f"[bold red]Error during plan generation:[/bold red] {str(e)}")
        raise


# Run the test
if __name__ == "__main__":
    asyncio.run(test_task_planner(provider=OllamaProvider(), model="qwen3:0.6b"))
    # asyncio.run(test_task_planner(provider=OpenAIProvider(), model="gpt-4o-mini"))
    # asyncio.run(
    #     test_task_planner(
    #         provider=AnthropicProvider(), model="claude-3-5-sonnet-20241022"
    #     )
    # )
    # asyncio.run(test_task_planner(provider=GeminiProvider(), model="gemini-2.0-flash"))
