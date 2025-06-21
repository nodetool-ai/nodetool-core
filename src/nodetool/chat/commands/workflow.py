"""Workflow execution command."""

import json
import traceback
from typing import List
from nodetool.chat.chat_cli import ChatCLI
from rich.syntax import Syntax
from .base import Command
from nodetool.models.workflow import Workflow
from nodetool.types.graph import get_input_schema, get_output_schema
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.run_job_request import RunJobRequest


class RunWorkflowCommand(Command):
    """Command to run a workflow by name from the database."""

    def __init__(self):
        super().__init__(
            "workflow",
            "Run a workflow by name: /workflow <workflow_name> [input_values_json]",
            ["wf"],
        )

    async def execute(self, cli: ChatCLI, args: List[str]) -> bool:
        if not args:
            cli.console.print(
                "[bold red]Usage:[/bold red] /workflow <workflow_name> [input_values_json]"
            )
            cli.console.print(
                "Example: /workflow \"My Workflow\" '{\"input1\": 5, \"input2\": 3}'"
            )
            return False

        workflow_name = args[0]
        input_values = {}

        # Parse input values if provided
        if len(args) > 1:
            try:
                input_values = json.loads(" ".join(args[1:]))
            except json.JSONDecodeError as e:
                cli.console.print(
                    f"[bold red]Error parsing input JSON:[/bold red] {e}"
                )
                return False

        # Find workflow by name
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

            cli.console.print(f"[bold green]Found workflow:[/bold green] {found_workflow.name}")
            if found_workflow.description:
                cli.console.print(f"[bold cyan]Description:[/bold cyan] {found_workflow.description}")
                
        except Exception as e:
            cli.console.print(
                f"[bold red]Error loading workflow:[/bold red] {e}"
            )
            return False

        # Get the graph from the workflow
        try:
            graph = found_workflow.get_api_graph()
        except Exception as e:
            cli.console.print(
                f"[bold red]Error getting workflow graph:[/bold red] {e}"
            )
            return False

        # Get input and output schemas
        try:
            input_schema = get_input_schema(graph)
            output_schema = get_output_schema(graph)
            
            cli.console.print("\n[bold cyan]Input Schema:[/bold cyan]")
            cli.console.print(Syntax(
                json.dumps(input_schema, indent=2),
                "json",
                theme="monokai",
                line_numbers=False,
            ))
            
            cli.console.print("\n[bold cyan]Output Schema:[/bold cyan]")
            cli.console.print(Syntax(
                json.dumps(output_schema, indent=2),
                "json",
                theme="monokai",
                line_numbers=False,
            ))
            
        except Exception as e:
            cli.console.print(
                f"[bold yellow]Warning:[/bold yellow] Could not generate schemas: {e}"
            )

        # Update graph with input values if provided
        if input_values:
            try:
                for node in graph.nodes:
                    if node.type.startswith("nodetool.input."):
                        node_name = node.data.get("name", node.id)
                        if node_name in input_values:
                            node.data["value"] = input_values[node_name]
                            cli.console.print(
                                f"[bold green]Set {node_name} = {input_values[node_name]}[/bold green]"
                            )
            except Exception as e:
                cli.console.print(
                    f"[bold yellow]Warning:[/bold yellow] Error setting input values: {e}"
                )

        # Create workflow request
        try:
            req = RunJobRequest(
                user_id=cli.context.user_id,
                auth_token=cli.context.auth_token,
                workflow_id=found_workflow.id,
                graph=graph,
            )
            
            cli.console.print(f"\n[bold green]Running workflow '{workflow_name}'...[/bold green]")
            cli.console.print("-" * 50)
            
            # Run the workflow
            async for msg in run_workflow(req, context=cli.context, use_thread=False):
                if hasattr(msg, 'content'):
                    cli.console.print(msg.content) # type: ignore
                else:
                    cli.console.print(str(msg))
                    
            cli.console.print("-" * 50)
            cli.console.print("[bold green]Workflow completed![/bold green]")
            
        except Exception as e:
            cli.console.print(
                f"[bold red]Error running workflow:[/bold red] {e}"
            )
            cli.console.print(
                Syntax(
                    traceback.format_exc(),
                    "python",
                    theme="monokai",
                    line_numbers=True,
                )
            )

        return False