"""Workflow creation command using GraphPlanner."""

import traceback

from rich.panel import Panel
from rich.syntax import Syntax

from nodetool.agents.graph_planner import GraphPlanner, print_visual_graph
from nodetool.chat.chat_cli import ChatCLI
from nodetool.models.workflow import Workflow
from nodetool.providers import get_provider
from nodetool.workflows.types import Chunk, PlanningUpdate

from .base import Command


class CreateWorkflowCommand(Command):
    """Command to create a new workflow using GraphPlanner from natural language objective."""

    def __init__(self):
        super().__init__(
            "create-workflow",
            "Create a new workflow from objective: /create-workflow <objective>",
            ["cw", "create_workflow"],
        )

    async def _get_workflow_name_and_description(self, cli: ChatCLI, objective: str) -> tuple[str, str]:
        """Get workflow name and description from user."""
        cli.console.print("\n[bold cyan]Workflow Details[/bold cyan]")

        # Suggest a name based on objective
        suggested_name = objective[:50].replace('"', "").strip()
        if len(objective) > 50:
            suggested_name += "..."

        try:
            name = await cli.session.prompt_async(f"Workflow name [{suggested_name}]: ")
            if not name.strip():
                name = suggested_name

            description = await cli.session.prompt_async(f"Description [{objective}]: ")
            if not description.strip():
                description = objective

        except (KeyboardInterrupt, EOFError):
            name = suggested_name
            description = objective

        return name.strip(), description.strip()

    async def execute(self, cli: ChatCLI, args: list[str]) -> bool:
        if not args:
            cli.console.print("[bold red]Usage:[/bold red] /create-workflow <objective>")
            cli.console.print('Example: /create-workflow "Process sales data and generate summary report"')
            return False

        # Check if model is selected
        if not cli.selected_model:
            cli.console.print("[bold red]Error:[/bold red] No model selected. Use /model to select one.")
            return False

        objective = " ".join(args)

        # Display objective
        objective_panel = Panel(
            f"[bold green]Objective:[/bold green] {objective}",
            title="Workflow Creation",
            border_style="cyan",
        )
        cli.console.print(objective_panel)

        try:
            # Create GraphPlanner with empty schemas for now
            provider = await get_provider(cli.selected_model.provider)
            planner = GraphPlanner(
                provider=provider,
                model=cli.selected_model.id,
                objective=objective,
                input_schema=[],  # Empty for now
                output_schema=[],  # Empty for now
                verbose=True,
            )

            cli.console.print("\n[bold green]Starting workflow creation...[/bold green]")

            # Create the graph with progress updates
            streaming_output = ""
            lines_printed = 0

            try:
                async for update in planner.create_graph(cli.context):
                    if isinstance(update, PlanningUpdate):
                        status_color = (
                            "green"
                            if update.status == "Success"
                            else "yellow"
                            if update.status == "Starting"
                            else "red"
                        )
                        cli.console.print(f"[bold {status_color}]{update.phase}:[/bold {status_color}] {update.status}")
                        if update.content:
                            cli.console.print(f"  {update.content}")
                    elif isinstance(update, Chunk):
                        # Collect streaming content and track lines
                        streaming_output += update.content
                        chunk_lines = update.content.count("\n")
                        cli.console.print(update.content, highlight=False, end="")
                        lines_printed += chunk_lines

                # Clear the streamed output and display final result
                if lines_printed > 0:
                    # Move cursor up and clear streamed content
                    cli.console.print(f"\033[{lines_printed}A", end="")
                    cli.console.print("\033[0J", end="")

                # Print the final complete result if there was streaming content
                if streaming_output.strip():
                    cli.console.print(streaming_output.strip())

            except Exception as e:
                cli.console.print(f"[bold red]Error during planning:[/bold red] {e}")
                cli.console.print(
                    Syntax(
                        traceback.format_exc(),
                        "python",
                        theme="monokai",
                        line_numbers=True,
                    )
                )
                return False

            # Check if graph was created successfully
            if not planner.graph:
                cli.console.print("[bold red]Error:[/bold red] Failed to create workflow graph")
                return False

            # Display the visual graph
            cli.console.print("\n[bold green]Generated Workflow Graph:[/bold green]")
            print_visual_graph(planner.graph)

            # Get workflow name and description
            (
                workflow_name,
                workflow_description,
            ) = await self._get_workflow_name_and_description(cli, objective)

            # Save the workflow
            try:
                workflow = Workflow.create(
                    user_id=cli.context.user_id,
                    name=workflow_name,
                    description=workflow_description,
                    graph=planner.graph.model_dump(),
                )
                workflow.save()

                cli.console.print("\n[bold green]✓ Workflow saved successfully![/bold green]")
                cli.console.print(f"[bold cyan]Name:[/bold cyan] {workflow_name}")
                cli.console.print(f"[bold cyan]Description:[/bold cyan] {workflow_description}")
                cli.console.print(f"[bold cyan]ID:[/bold cyan] {workflow.id}")
                cli.console.print(f'\nUse [bold]/workflow "{workflow_name}"[/bold] to run this workflow')

            except Exception as e:
                cli.console.print(f"[bold red]Error saving workflow:[/bold red] {e}")
                cli.console.print(
                    Syntax(
                        traceback.format_exc(),
                        "python",
                        theme="monokai",
                        line_numbers=True,
                    )
                )
                return False

        except KeyboardInterrupt:
            cli.console.print("\n[bold yellow]Workflow creation cancelled[/bold yellow]")
        except Exception as e:
            cli.console.print(f"[bold red]Unexpected error:[/bold red] {e}")
            cli.console.print(Syntax(traceback.format_exc(), "python", theme="monokai", line_numbers=True))

        return False
