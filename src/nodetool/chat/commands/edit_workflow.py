"""Workflow editing command using GraphPlanner."""

import traceback
from typing import List
from nodetool.chat.chat_cli import ChatCLI
from rich.syntax import Syntax
from rich.panel import Panel
from .base import Command
from nodetool.agents.graph_planner import GraphPlanner
from nodetool.providers import get_provider
from nodetool.workflows.types import PlanningUpdate, Chunk
from nodetool.models.workflow import Workflow
from nodetool.agents.graph_planner import print_visual_graph


class EditWorkflowCommand(Command):
    """Command to edit an existing workflow using GraphPlanner."""

    def __init__(self) -> None:
        super().__init__(
            "edit-workflow",
            "Edit an existing workflow: /edit-workflow <workflow_name> <new_objective>",
            ["ew", "edit_workflow"],
        )

    async def _get_workflow_name_and_description(
        self,
        cli: ChatCLI,
        objective: str,
        original_name: str,
        original_description: str,
    ) -> tuple[str, str]:
        """Get updated workflow name and description from user."""
        cli.console.print("\n[bold cyan]Update Workflow Details[/bold cyan]")

        try:
            name = await cli.session.prompt_async(f"Workflow name [{original_name}]: ")
            if not name.strip():
                name = original_name

            description = await cli.session.prompt_async(f"Description [{objective}]: ")
            if not description.strip():
                description = objective

        except (KeyboardInterrupt, EOFError):
            name = original_name
            description = objective

        return name.strip(), description.strip()

    async def execute(self, cli: ChatCLI, args: List[str]) -> bool:
        if len(args) < 2:
            cli.console.print(
                "[bold red]Usage:[/bold red] /edit-workflow <workflow_name> <new_objective>"
            )
            cli.console.print(
                'Example: /edit-workflow "Sales Report" "Process sales data and include quarterly trends"'
            )
            return False

        # Check if model is selected
        if not cli.selected_model:
            cli.console.print(
                "[bold red]Error:[/bold red] No model selected. Use /model to select one."
            )
            return False

        workflow_name = args[0]
        objective = " ".join(args[1:])

        # Find the existing workflow
        try:
            workflows, _ = Workflow.paginate(user_id=cli.context.user_id, limit=1000)

            # Find workflow with matching name
            found_workflow = None
            for workflow in workflows:
                if workflow.name == workflow_name:
                    found_workflow = workflow
                    break

            if not found_workflow:
                cli.console.print(
                    f"[bold red]Error:[/bold red] Workflow '{workflow_name}' not found"
                )
                cli.console.print("Available workflows:")
                for wf in workflows[:10]:  # Show first 10 workflows
                    cli.console.print(f"  - {wf.name}")
                if len(workflows) > 10:
                    cli.console.print(f"  ... and {len(workflows) - 10} more")
                return False

            cli.console.print(
                f"[bold green]Found workflow:[/bold green] {found_workflow.name}"
            )
            if found_workflow.description:
                cli.console.print(
                    f"[bold cyan]Current Description:[/bold cyan] {found_workflow.description}"
                )

        except Exception as e:
            cli.console.print(f"[bold red]Error loading workflow:[/bold red] {e}")
            return False

        # Get the existing graph
        try:
            existing_graph = found_workflow.get_api_graph()
        except Exception as e:
            cli.console.print(f"[bold red]Error getting workflow graph:[/bold red] {e}")
            return False

        # Display objective
        objective_panel = Panel(
            f"[bold green]Original:[/bold green] {found_workflow.description or 'No description'}\n"
            f"[bold cyan]New Objective:[/bold cyan] {objective}",
            title="Workflow Editing",
            border_style="cyan",
        )
        cli.console.print(objective_panel)

        try:
            # Create GraphPlanner with the existing graph
            provider = get_provider(cli.selected_model.provider)
            planner = GraphPlanner(
                provider=provider,
                model=cli.selected_model.id,
                objective=objective,
                input_schema=[],  # Empty for now - will be inferred
                output_schema=[],  # Empty for now - will be inferred
                existing_graph=existing_graph,  # Pass the existing graph
                verbose=True,
            )

            cli.console.print("\n[bold green]Starting workflow editing...[/bold green]")
            cli.console.print(
                "[bold yellow]Note:[/bold yellow] The AI will modify the existing workflow based on your new objective."
            )

            # Create the updated graph with progress updates
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
                        cli.console.print(
                            f"[bold {status_color}]{update.phase}:[/bold {status_color}] {update.status}"
                        )
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
                cli.console.print(
                    "[bold red]Error:[/bold red] Failed to edit workflow graph"
                )
                return False

            # Display the updated visual graph
            cli.console.print("\n[bold green]Updated Workflow Graph:[/bold green]")
            print_visual_graph(planner.graph)

            # Get updated workflow name and description
            (
                workflow_name_updated,
                workflow_description_updated,
            ) = await self._get_workflow_name_and_description(
                cli,
                objective,
                found_workflow.name,
                found_workflow.description or "",
            )

            # Update the existing workflow
            try:
                found_workflow.name = workflow_name_updated
                found_workflow.description = workflow_description_updated
                found_workflow.graph = planner.graph.model_dump()
                found_workflow.save()

                cli.console.print(
                    "\n[bold green]✓ Workflow updated successfully![/bold green]"
                )
                cli.console.print(
                    f"[bold cyan]Name:[/bold cyan] {workflow_name_updated}"
                )
                cli.console.print(
                    f"[bold cyan]Description:[/bold cyan] {workflow_description_updated}"
                )
                cli.console.print(f"[bold cyan]ID:[/bold cyan] {found_workflow.id}")
                cli.console.print(
                    f'\nUse [bold]/workflow "{workflow_name_updated}"[/bold] to run this updated workflow'
                )

            except Exception as e:
                cli.console.print(f"[bold red]Error updating workflow:[/bold red] {e}")
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
            cli.console.print("\n[bold yellow]Workflow editing cancelled[/bold yellow]")
        except Exception as e:
            cli.console.print(f"[bold red]Unexpected error:[/bold red] {e}")
            cli.console.print(
                Syntax(
                    traceback.format_exc(), "python", theme="monokai", line_numbers=True
                )
            )

        return False
