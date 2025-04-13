import asyncio
from nodetool.chat.providers import ChatProvider, get_provider
from nodetool.agents.tools.base import Tool
from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.metadata.types import (
    Message,
    Provider,
    ToolCall,
)

import json
import os
from typing import Any, Sequence

from nodetool.packages.registry import Registry
from nodetool.metadata.node_metadata import NodeMetadata
from nodetool.types.graph import Graph
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.processing_context import ProcessingContext
import time
import networkx as nx
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
)

from nodetool.workflows.property import Property
from nodetool.workflows.run_job_request import RunJobRequest

# Create Rich console for prettier output
console = Console()

# Simplify the DEFAULT_PLANNING_SYSTEM_PROMPT
DEFAULT_PLANNING_SYSTEM_PROMPT = """
You are a task planning agent that creates optimized, executable workflows.

RESPOND WITH TOOL CALLS TO CREATE WORKFLOWS.

KEY PLANNING PRINCIPLES:
1. Break complex goals into nodes with clear responsibilities
2. Connect nodes by directly referencing source nodes in property values
3. Ensure the workflow forms a Directed Acyclic Graph (DAG)
4. Use appropriate node types based on their description
5. Provide all necessary properties 

NODE PROPERTIES:
- The properties of a node are objects with key-value pairs
- The properties are specific to the node type
- The properties are optional and can be omitted if not needed
- If not provided, each property should have a default value
- ADD PROPERTIES TO THE NODE THAT ARE NEEDED TO COMPLETE THE TASK
- For model properties, always use gpt-4o
- A PROPERTY CAN BE SET EITHER AS A REFERENCE OR A VALUE, BUT NOT BOTH
- To reference another node's output, use the format: { "source": "nodeId", "sourceHandle": "outputName" }

DEPENDENCY GRAPH:
- The dependency graph is a directed graph of dependencies between nodes
- NODES must not have circular dependencies
- NODES must be connected via property references
"""


class GetNodeMetadataTool(Tool):
    name = "get_node_metadata"
    description = "Get detailed metadata for a specific node type"
    input_schema = {
        "type": "object",
        "properties": {
            "node_type": {
                "type": "string",
                "description": "The node type identifier to get metadata for",
            }
        },
        "required": ["node_type"],
    }

    def __init__(self, workspace_dir: str, node_types: list[NodeMetadata]):
        super().__init__(workspace_dir=workspace_dir)
        self.workspace_dir = workspace_dir
        self.node_types = node_types

    async def process(self, context: ProcessingContext, params: dict) -> dict:
        node_type = params["node_type"]

        # Find the node metadata that matches the node_type
        node_metadata = next(
            (node for node in self.node_types if node.node_type == node_type), None
        )

        if node_metadata:
            result = node_metadata.model_dump()

            # Add formatted property details for easier understanding
            property_details = []
            for prop in node_metadata.properties:
                # Get type information
                type_info = prop.type.type
                if prop.type.is_enum_type() and prop.type.values:
                    type_info += (
                        f" (options: {', '.join(str(v) for v in prop.type.values)})"
                    )
                elif prop.type.is_list_type() and prop.type.type_args:
                    type_info += f" of {prop.type.type_args[0].type}"

                # Required/optional status
                status = "required" if not prop.type.optional else "optional"

                property_details.append(
                    {
                        "name": prop.name,
                        "type": type_info,
                        "status": status,
                        "description": prop.description,
                    }
                )

            result["formatted_properties"] = property_details
            return result
        else:
            return {
                "error": f"No node found with type: {node_type}",
                "node_type": node_type,
            }


class CreateWorkflowTool(Tool):
    """
    Workflow Creator - Tool for generating a workflow with nodes
    """

    name = "create_workflow"
    description = "Create a workflow with nodes"

    def __init__(self, workspace_dir: str, input_schema: dict):
        super().__init__(workspace_dir=workspace_dir)
        self.workspace_dir = workspace_dir
        self.input_schema = input_schema

    async def process(self, context: ProcessingContext, params: dict):
        return params


class SelectNodesTool(Tool):
    """
    Node Selector - Tool for selecting the final set of node types for a workflow
    """

    name = "select_nodes"
    description = "Select the final set of node types for the workflow"
    input_schema = {
        "type": "object",
        "properties": {
            "selected_nodes": {
                "type": "array",
                "description": "The types of nodes to include in the workflow",
                "items": {
                    "type": "string",
                    "description": "Node type identifier",
                },
            },
            "justification": {
                "type": "string",
                "description": "Explanation of why these nodes were selected",
            },
        },
        "required": ["selected_nodes", "justification"],
    }

    def __init__(self, workspace_dir: str):
        super().__init__(workspace_dir=workspace_dir)
        self.workspace_dir = workspace_dir

    async def process(self, context: ProcessingContext, params: dict) -> dict:
        # Simply return the parameters
        return params


# Default prompts for workflow planning
DEFAULT_WORKFLOW_SYSTEM_PROMPT = """
You are a workflow planning agent that creates optimized, executable graph workflows.

RESPOND WITH TOOL CALLS TO CREATE WORKFLOWS.

KEY WORKFLOW PLANNING PRINCIPLES:
1. Break complex goals into nodes with clear responsibilities
2. Connect nodes by directly referencing source nodes in property values
3. Ensure the workflow forms a Directed Acyclic Graph (DAG)
4. Create self-contained nodes with minimal coupling
5. Use appropriate node types based on their metadata
6. Provide all necessary properties for each node

NODE CONNECTIONS:
- To connect nodes, reference source nodes directly in property values
- Use the format: { "source": "nodeId", "sourceHandle": "outputName" }
- Ensure the sourceHandle is a valid output of the source node
- This replaces the need for separate edge definitions

NODE SEARCH INSTRUCTIONS:
- ALWAYS use the search_nodes tool to find appropriate nodes for your workflow
- Search by keywords, node types, or namespaces to find relevant nodes
- Examine node metadata including properties, inputs, and outputs
- Multiple searches with different queries may be needed to discover all relevant nodes
- You MUST search for nodes before creating a workflow

WORKFLOW GRAPH:
- The workflow is a directed graph of nodes connected via property references
- Nodes must not have circular dependencies
- All required inputs for each node must be connected
- Node properties must be set with appropriate values
"""


DEFAULT_WORKFLOW_AGENT_PROMPT = """
Objective: {objective}

IMPORTANT: You MUST use the search_nodes tool to find appropriate nodes before creating a workflow.

HOW TO SEARCH FOR NODES:
1. Use search_nodes tool with relevant keywords related to your objective
2. Try different queries to discover all required node types
3. Examine node properties, inputs, and outputs carefully

Available namespaces to search within:
{node_types_info}

Think carefully about:
1. How to structure the workflow to accomplish the objective
2. Which node types to use and how to configure their properties
3. How to connect nodes properly to form a valid DAG
4. What properties to set for each node, and add properties that are needed to complete the task
5. Properties are defined in the data field of the node
6. Ensuring all required inputs for each node are connected

Create a workflow that is clear, executable, and achieves the objective.
"""


class WorkflowPlanner:
    """
    🧩 The Workflow Architect - Creates executable graph workflows

    This component designs workflow graphs with nodes and edges to accomplish complex tasks.
    It uses metadata from node types to inform the LLM about available nodes and their properties.

    Features:
    - Node type metadata integration
    - Edge connection validation
    - DAG structure validation
    - Workflow persistence through JSON storage
    - Detailed LLM trace logging for debugging and analysis
    """

    def __init__(
        self,
        provider: ChatProvider,
        model: str,
        objective: str,
        workspace_dir: str,
        node_types: list[NodeMetadata],
        tools: Sequence[Tool] = [],
        system_prompt: str | None = None,
        agent_prompt: str | None = None,
        enable_tracing: bool = True,
    ):
        """
        Initialize the WorkflowPlanner.

        Args:
            provider (ChatProvider): An LLM provider instance
            model (str): The model to use with the provider
            objective (str): The objective to accomplish
            workspace_dir (str): The workspace directory path
            node_types (list[type[BaseNode]]): Available node types for the workflow
            tools (List[Tool], optional): Additional tools available for workflow creation
            system_prompt (str, optional): Custom system prompt
            agent_prompt (str, optional): Custom agent prompt
            enable_tracing (bool, optional): Whether to enable LLM trace logging
        """
        self.provider = provider
        self.model = model
        self.objective = objective
        self.workspace_dir = workspace_dir
        self.node_types = node_types
        self.system_prompt = (
            system_prompt if system_prompt else DEFAULT_WORKFLOW_SYSTEM_PROMPT
        )
        self.agent_prompt = (
            agent_prompt if agent_prompt else DEFAULT_WORKFLOW_AGENT_PROMPT
        )
        self.tools = list(tools) or []
        self.enable_tracing = enable_tracing
        self.workflow = None

        # Setup tracing
        if self.enable_tracing:
            self.traces_dir = os.path.join(self.workspace_dir, "traces")
            os.makedirs(self.traces_dir, exist_ok=True)
            sanitized_objective = "".join(
                c if c.isalnum() else "_" for c in self.objective[:40]
            )
            self.trace_file_path = os.path.join(
                self.traces_dir, f"trace_workflow_planner_{sanitized_objective}.jsonl"
            )
            self._log_trace_event(
                "workflow_planner_initialized",
                {"objective": self.objective, "model": self.model},
            )

    def _log_trace_event(self, event_type: str, data: dict) -> None:
        """
        Log an event to the trace file.

        Args:
            event_type (str): Type of event (message, tool_call, etc.)
            data (dict): Event data to log
        """
        if not self.enable_tracing:
            return

        trace_entry = {"timestamp": time.time(), "event": event_type, "data": data}

        with open(self.trace_file_path, "a") as f:
            f.write(json.dumps(trace_entry) + "\n")

    def _get_available_namespaces(self) -> list[str]:
        """
        Extract unique namespaces from available node types.

        Returns:
            list[str]: List of unique namespaces
        """
        namespaces = set()
        for node_type in self.node_types:
            try:
                namespace = node_type.namespace
                if namespace:
                    namespaces.add(namespace)
            except (AttributeError, Exception):
                continue

        return sorted(list(namespaces))

    def _get_available_node_types_list(self) -> str:
        """
        Get a comma-separated list of all available node types.

        Returns:
            str: Comma-separated list of node types
        """
        node_types = [node_type.node_type for node_type in self.node_types]
        return ", ".join(node_types)

    async def _build_agent_workflow_prompt(self) -> str:
        """
        Build the prompt for the workflow planning agent.

        Returns:
            str: The formatted agent prompt
        """
        namespaces = self._get_available_namespaces()
        namespaces_info = "Available namespaces:\n" + "\n".join(
            [f"- {ns}" for ns in namespaces]
        )

        return self.agent_prompt.format(
            objective=self.objective,
            node_types_info=namespaces_info,
        )

    def _validate_workflow(self, nodes, edges=None) -> list[str]:
        """
        Validate a workflow by checking node connections and DAG structure.

        Args:
            nodes: List of nodes in the workflow
            edges: Optional list of edges (for backward compatibility)

        Returns:
            list[str]: List of validation error messages
        """
        validation_errors = []

        # Create a mapping of node IDs to nodes
        node_ids = {node["id"] for node in nodes}
        node_dict = {node["id"]: node for node in nodes}

        # Build a graph to check for cycles
        G = nx.DiGraph()
        for node in nodes:
            G.add_node(node["id"])

        # Validate node types
        validation_errors.extend(self._validate_node_types(nodes))

        # Validate node properties and build dependency graph
        property_validation_errors, G = self._validate_node_properties(
            nodes, node_ids, node_dict, G
        )
        validation_errors.extend(property_validation_errors)

        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(G))
            if cycles:
                validation_errors.append(f"Workflow contains cycles: {cycles}")
        except nx.NetworkXNoCycle:
            pass

        return validation_errors

    def _validate_node_types(self, nodes) -> list[str]:
        """
        Validate that all nodes have valid types.

        Args:
            nodes: List of nodes in the workflow

        Returns:
            list[str]: List of validation error messages
        """
        validation_errors = []

        # Check for valid node types
        node_type_names = [node_type.node_type for node_type in self.node_types]
        for node in nodes:
            if node["type"] not in node_type_names:
                validation_errors.append(f"Invalid node type: {node['type']}")

        return validation_errors

    def _validate_node_properties(
        self, nodes, node_ids, node_dict, G
    ) -> tuple[list[str], nx.DiGraph]:
        """
        Validate node properties and build dependency graph.

        Args:
            nodes: List of nodes in the workflow
            node_ids: Set of node IDs
            node_dict: Dictionary mapping node IDs to nodes
            G: DiGraph to add edges to

        Returns:
            tuple[list[str], nx.DiGraph]: List of validation errors and updated graph
        """
        validation_errors = []

        # Check node property references
        for node in nodes:
            if node.get("data"):
                node_type = node["type"]
                node_id = node["id"]
                node_metadata = next(
                    (nt for nt in self.node_types if nt.node_type == node_type), None
                )

                if node_metadata is None:
                    validation_errors.append(
                        f"Node {node_id} has invalid type: {node_type}"
                    )
                else:
                    # Validate property types and references
                    properties = node.get("data", {})
                    for prop_name, prop_value in properties.items():
                        prop_metadata = next(
                            (
                                p
                                for p in node_metadata.properties
                                if p.name == prop_name
                            ),
                            None,
                        )

                        if prop_metadata is None:
                            validation_errors.append(
                                f"Node {node_id} has unknown property: {prop_name}"
                            )
                        else:
                            # Check if this is a node reference
                            if (
                                isinstance(prop_value, dict)
                                and "source" in prop_value
                                and "sourceHandle" in prop_value
                            ):
                                errors, G = self._validate_node_reference(
                                    node_id,
                                    prop_name,
                                    prop_value,
                                    prop_metadata,
                                    node_ids,
                                    node_dict,
                                    G,
                                )
                                validation_errors.extend(errors)
                            else:
                                # Validate normal property type
                                try:
                                    self._validate_property_type(
                                        prop_metadata, prop_value, node_id, prop_name
                                    )
                                except ValueError as e:
                                    validation_errors.append(str(e))

        return validation_errors, G

    def _validate_node_reference(
        self, node_id, prop_name, prop_value, prop_metadata, node_ids, node_dict, G
    ) -> tuple[list[str], nx.DiGraph]:
        """
        Validate a node reference property and update the dependency graph.

        Args:
            node_id: ID of the node containing the property
            prop_name: Name of the property
            prop_value: Value of the property (a node reference)
            prop_metadata: Metadata for the property
            node_ids: Set of all node IDs
            node_dict: Dictionary mapping node IDs to nodes
            G: DiGraph to add edges to

        Returns:
            tuple[list[str], nx.DiGraph]: List of validation errors and updated graph
        """
        validation_errors = []
        source_node_id = prop_value["source"]
        source_handle = prop_value["sourceHandle"]

        # Check if source node exists
        if source_node_id not in node_ids:
            validation_errors.append(
                f"Node {node_id} property '{prop_name}' references non-existent node: {source_node_id}"
            )
        else:
            # Add edge to dependency graph
            G.add_edge(source_node_id, node_id)

            # Verify sourceHandle is valid
            source_node = node_dict[source_node_id]
            source_node_type = source_node["type"]
            source_node_metadata = next(
                (nt for nt in self.node_types if nt.node_type == source_node_type),
                None,
            )

            if source_node_metadata:
                if not any(
                    output.name == source_handle
                    for output in source_node_metadata.outputs
                ):
                    validation_errors.append(
                        f"Node {node_id} property '{prop_name}' references invalid sourceHandle: {source_handle}"
                    )
                else:
                    # Get output type from source node
                    source_output = next(
                        (
                            output
                            for output in source_node_metadata.outputs
                            if output.name == source_handle
                        ),
                        None,
                    )

                    # Get property type from target node
                    target_property = prop_metadata

                    # Check type compatibility
                    if source_output and target_property:
                        from nodetool.metadata import typecheck

                        if not typecheck(
                            source_output.type,
                            target_property.type,
                        ):
                            validation_errors.append(
                                f"Type mismatch: Node {node_id} property '{prop_name}' of type '{target_property.type.type}' cannot accept output '{source_handle}' of type '{source_output.type.type}' from node {source_node_id}"
                            )

        return validation_errors, G

    def _validate_property_type(
        self, prop: Property, value: Any, node_id: str, prop_name: str
    ) -> None:
        """
        Validate a property value against its type metadata.

        Args:
            prop: The property metadata
            value: The value to validate
            node_id: The ID of the node (for error messages)
            prop_name: The name of the property (for error messages)

        Raises:
            ValueError: If the value doesn't match the expected type
        """
        type_metadata = prop.type

        # Handle None/null values
        if value is None:
            if not type_metadata.optional:
                raise ValueError(
                    f"Node {node_id}: Property '{prop_name}' cannot be null"
                )
            return

        # Validate based on type
        if type_metadata.is_primitive_type():
            self._validate_primitive_type(type_metadata, value, node_id, prop_name)
        elif type_metadata.is_enum_type():
            self._validate_enum_type(type_metadata, value, node_id, prop_name)
        elif type_metadata.is_list_type():
            self._validate_list_type(type_metadata, value, node_id, prop_name)
        elif type_metadata.is_dict_type():
            self._validate_dict_type(type_metadata, value, node_id, prop_name)
        elif type_metadata.is_union_type():
            self._validate_union_type(type_metadata, value, node_id, prop_name)
        elif type_metadata.is_asset_type():
            self._validate_asset_type(type_metadata, value, node_id, prop_name)
        # Add more type validations as needed

    def _validate_primitive_type(
        self, type_metadata: TypeMetadata, value: Any, node_id: str, prop_name: str
    ) -> None:
        """Validate primitive type values (int, float, bool, str)"""
        if type_metadata.type == "int" and not isinstance(value, int):
            raise ValueError(
                f"Node {node_id}: Property '{prop_name}' expected type 'int', got '{type(value).__name__}'"
            )
        elif type_metadata.type == "float" and not (isinstance(value, (int, float))):
            raise ValueError(
                f"Node {node_id}: Property '{prop_name}' expected type 'float', got '{type(value).__name__}'"
            )
        elif type_metadata.type == "bool" and not isinstance(value, bool):
            raise ValueError(
                f"Node {node_id}: Property '{prop_name}' expected type 'bool', got '{type(value).__name__}'"
            )
        elif type_metadata.type in ("str", "text") and not isinstance(value, str):
            raise ValueError(
                f"Node {node_id}: Property '{prop_name}' expected type 'string', got '{type(value).__name__}'"
            )

    def _validate_enum_type(
        self, type_metadata: TypeMetadata, value: Any, node_id: str, prop_name: str
    ) -> None:
        """Validate enum type values"""
        if type_metadata.values and value not in type_metadata.values:
            raise ValueError(
                f"Node {node_id}: Property '{prop_name}' expected one of {type_metadata.values}, got '{value}'"
            )

    def _validate_list_type(
        self, type_metadata: TypeMetadata, value: Any, node_id: str, prop_name: str
    ) -> None:
        """Validate list type values"""
        if not isinstance(value, list):
            raise ValueError(
                f"Node {node_id}: Property '{prop_name}' expected type 'list', got '{type(value).__name__}'"
            )

        # If we have type args, validate each item in the list
        if type_metadata.type_args and value:
            item_type = type_metadata.type_args[0]
            for i, item in enumerate(value):
                try:
                    # Recursively validate each item
                    self._validate_property_type(
                        Property(name=f"{prop_name}[{i}]", type=item_type),
                        item,
                        node_id,
                        f"{prop_name}[{i}]",
                    )
                except ValueError as e:
                    raise ValueError(str(e))

    def _validate_dict_type(
        self, type_metadata: TypeMetadata, value: Any, node_id: str, prop_name: str
    ) -> None:
        """Validate dict type values"""
        if not isinstance(value, dict):
            raise ValueError(
                f"Node {node_id}: Property '{prop_name}' expected type 'dict', got '{type(value).__name__}'"
            )

    def _validate_union_type(
        self, type_metadata: TypeMetadata, value: Any, node_id: str, prop_name: str
    ) -> None:
        """Validate union type values"""
        # For union types, try each possible type until one matches
        if not type_metadata.type_args:
            return

        errors = []
        for type_arg in type_metadata.type_args:
            try:
                self._validate_property_type(
                    Property(name=prop_name, type=type_arg), value, node_id, prop_name
                )
                return  # If validation succeeds, we're done
            except ValueError as e:
                errors.append(str(e))

        # If we get here, none of the union types matched
        raise ValueError(
            f"Node {node_id}: Property '{prop_name}' failed to match any of the expected types: {[t.type for t in type_metadata.type_args]}"
        )

    def _validate_asset_type(
        self, type_metadata: TypeMetadata, value: Any, node_id: str, prop_name: str
    ) -> None:
        """Validate asset type values (image, audio, video, etc.)"""
        if not isinstance(value, dict) or "uri" not in value:
            raise ValueError(
                f"Node {node_id}: Property '{prop_name}' expected an asset with 'uri' field, got {value}"
            )

    async def _execute_tool(self, tools: list[Tool], tool_call: ToolCall) -> ToolCall:
        """
        Execute a tool call using the available tools.

        Args:
            tool_call: The tool call to execute

        Returns:
            The tool call with the result attached
        """
        tools = [
            GetNodeMetadataTool(
                workspace_dir=self.workspace_dir, node_types=self.node_types
            )
        ] + self.tools

        for tool in tools:
            if tool.name == tool_call.name:
                try:
                    # Create a processing context if it doesn't exist
                    processing_context = ProcessingContext(
                        workspace_dir=self.workspace_dir
                    )
                    result = await tool.process(processing_context, tool_call.args)

                    self._log_trace_event(
                        "tool_executed",
                        {
                            "name": tool_call.name,
                            "args": tool_call.args,
                            "result": result,
                        },
                    )

                    return ToolCall(
                        id=tool_call.id,
                        name=tool_call.name,
                        args=tool_call.args,
                        result=result,
                    )
                except Exception as e:
                    error_result = {"error": str(e)}

                    # Log tool execution error in trace
                    self._log_trace_event(
                        "tool_execution_error",
                        {
                            "name": tool_call.name,
                            "args": tool_call.args,
                            "error": str(e),
                        },
                    )

                    return ToolCall(
                        id=tool_call.id,
                        name=tool_call.name,
                        args=tool_call.args,
                        result=error_result,
                    )

        # Tool not found
        return ToolCall(
            id=tool_call.id,
            name=tool_call.name,
            args=tool_call.args,
            result={"error": f"Tool '{tool_call.name}' not found"},
        )

    async def _first_stage_planning(self) -> tuple[list[NodeMetadata], str]:
        """
        First stage of planning: examine and select appropriate node types over multiple iterations.

        Returns:
            tuple[list[NodeMetadata], str]: List of selected node types and the justification
        """
        console.print(
            Panel.fit(
                f"[bold blue]Starting first stage planning[/bold blue]: selecting node types for objective: [yellow]{self.objective}[/yellow]",
                border_style="blue",
            )
        )

        # Setup conversation for first stage
        first_stage_system_prompt = """
        You are a workflow planning agent that selects appropriate nodes for a workflow.
        
        YOUR TASK:
        1. Examine the available node types to find those needed to accomplish the objective
        2. Use the get_node_metadata tool to inspect node functionality and properties
        3. Select a final set of node types that will be used to build the workflow
        4. Return this final selection with the select_nodes tool
        
        INSTRUCTIONS:
        - First, examine the node types provided grouped by namespace
        - For any node that seems relevant, use get_node_metadata to examine its details
        - You can get metadata for multiple nodes to understand their capabilities
        - After examining relevant nodes, make your selection with the select_nodes tool
        - Select a complete set of nodes that together can accomplish the objective
        """

        # Include all available node types in the prompt
        all_node_types = self._get_available_node_types_list()

        agent_prompt = f"""
        Objective: {self.objective}
        
        Please examine the available nodes and select the most appropriate ones for this objective:
        
        1. Review the available node types below organized by namespace.
        2. Use the get_node_metadata tool to examine any nodes that seem relevant.
        3. After sufficient examination, use the select_nodes tool to choose the final set of nodes.
        
        Available node types (grouped by namespace):
        
        {self._format_node_types_by_namespace()}
        
        Remember to select a complete set of nodes that together can accomplish the objective.
        """

        history = [
            Message(role="system", content=first_stage_system_prompt),
            Message(role="user", content=agent_prompt),
        ]

        # Create tools
        select_nodes_tool = SelectNodesTool(workspace_dir=self.workspace_dir)

        get_node_metadata_tool = GetNodeMetadataTool(
            workspace_dir=self.workspace_dir,
            node_types=self.node_types,
        )

        tools = [select_nodes_tool, get_node_metadata_tool]

        console.print(
            "[cyan]Executing first stage planning with multiple iterations allowed[/cyan]"
        )

        max_iterations = 20
        current_iteration = 0
        selected_nodes = []
        justification = ""
        examined_nodes = set()  # Track which nodes have been examined

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Planning in progress...", total=max_iterations
            )

            # Execute multiple message exchanges with tools enabled
            console.print(
                f"[cyan]>>> STARTING FIRST STAGE:[/cyan] Planning workflow for objective: [yellow]{self.objective}[/yellow]"
            )
            console.print(
                f"[dim]>>> Available tools:[/dim] {[tool.name for tool in tools]}"
            )

            while current_iteration < max_iterations and not selected_nodes:
                progress.update(
                    task,
                    completed=current_iteration,
                    description=f"[cyan]Planning iteration {current_iteration + 1}/{max_iterations}[/cyan]",
                )

                # Generate message
                message = await self.provider.generate_message(
                    messages=history,
                    model=self.model,
                    tools=tools,
                )

                # Log the planning step
                self._log_trace_event(
                    "first_stage_planning_iteration",
                    {
                        "iteration": current_iteration + 1,
                        "message": message.content,
                        "tool_calls": [tc.name for tc in (message.tool_calls or [])],
                    },
                )

                # Add model's message to history
                history.append(message)

                # Process tool calls
                new_node_metadata_examined = False
                node_metadata_calls = []

                # First process metadata tool calls
                for tool_call in message.tool_calls or []:
                    if tool_call.name == "get_node_metadata":
                        node_type = tool_call.args.get("node_type", "")
                        if node_type not in examined_nodes:
                            examined_nodes.add(node_type)
                            new_node_metadata_examined = True

                        executed_tool_call = await self._execute_tool(tools, tool_call)
                        node_metadata_calls.append(executed_tool_call)

                # Now process all tool calls
                for tool_call in message.tool_calls or []:
                    if tool_call.name == "get_node_metadata":
                        # Already processed above
                        pass
                    elif tool_call.name == "select_nodes":
                        # Process the final node selection
                        console.print(
                            "[dim]>>> Processing select_nodes tool call[/dim]"
                        )
                        node_types = tool_call.args.get("selected_nodes", [])
                        justification = tool_call.args.get("justification", "")

                        console.print(
                            f"[dim]>>> Selected node types:[/dim] [green]{node_types}[/green]"
                        )
                        # Create a table for selected nodes
                        table = Table(
                            title="Selected Node Types",
                            show_header=True,
                            header_style="bold magenta",
                        )
                        table.add_column("Node Type", style="dim")
                        for node_type in node_types:
                            table.add_row(node_type)
                        console.print(table)

                        # Show justification
                        truncated_justification = justification[:100] + (
                            "..." if len(justification) > 100 else ""
                        )
                        console.print(
                            f"[bold cyan]Selection justification:[/bold cyan] {truncated_justification}"
                        )

                        # Log node selection in trace
                        self._log_trace_event(
                            "nodes_selected",
                            {
                                "iteration": current_iteration + 1,
                                "selected_nodes": node_types,
                                "justification": justification,
                            },
                        )

                        # Get the full NodeMetadata objects for the selected node types
                        selected_nodes = [
                            node
                            for node in self.node_types
                            if node.node_type in node_types
                        ]

                        executed_tool_call = await self._execute_tool(tools, tool_call)
                        history.append(
                            Message(
                                role="tool",
                                content=json.dumps(executed_tool_call.result),
                                tool_call_id=executed_tool_call.id,
                                name=executed_tool_call.name,
                            )
                        )

                # Add all tool responses to history
                for executed_call in node_metadata_calls:
                    history.append(
                        Message(
                            role="tool",
                            content=json.dumps(executed_call.result),
                            tool_call_id=executed_call.id,
                            name=executed_call.name,
                        )
                    )

                # If we haven't selected nodes yet and examined new metadata, continue the conversation
                if not selected_nodes:
                    if new_node_metadata_examined or current_iteration == 0:
                        # If we've examined new nodes or it's the first iteration, continue prompting
                        if current_iteration < max_iterations - 1:
                            next_prompt = f"""
                            Thank you for examining those nodes. 
                            
                            So far you have examined: {list(examined_nodes)}
                            
                            Please continue your analysis:
                            1. Examine additional relevant nodes if needed using get_node_metadata
                            2. When you have enough information, use select_nodes to choose the final set
                            
                            Remember that the objective is: {self.objective}
                            """

                            history.append(Message(role="user", content=next_prompt))
                            current_iteration += 1
                        else:
                            # Final iteration, force a selection
                            final_prompt = """
                            We've reached the maximum number of iterations. 
                            Please make your final node selection now using the select_nodes tool.
                            Choose the best combination of nodes based on the metadata you've examined.
                            """

                            history.append(Message(role="user", content=final_prompt))
                            current_iteration += 1
                    else:
                        # If no new nodes were examined and no selection made, encourage selection
                        selection_prompt = """
                        You've already examined several node types but haven't made a selection yet.
                        Based on what you've learned so far, please use the select_nodes tool to choose
                        the final set of node types for this workflow.
                        """

                        history.append(Message(role="user", content=selection_prompt))
                        current_iteration += 1
                else:
                    # We have our nodes, exit the loop
                    progress.update(task, completed=max_iterations)
                    break

        # If we've exhausted iterations and still don't have nodes, raise an error
        if not selected_nodes:
            console.print(
                "[bold red]Error: Failed to complete node selection after maximum iterations[/bold red]"
            )
            raise ValueError(
                f"Failed to complete node selection after {max_iterations} iterations"
            )

        console.print(
            Panel.fit(
                f"[bold green]First stage planning complete:[/bold green] Selected [yellow]{len(selected_nodes)}[/yellow] node types after [cyan]{current_iteration + 1}[/cyan] iterations",
                border_style="green",
            )
        )

        return selected_nodes, justification

    def _format_node_types_by_namespace(self) -> str:
        """
        Format node types grouped by namespace for better visualization.

        Returns:
            str: Formatted string of node types by namespace
        """
        # Group nodes by namespace
        namespace_groups = {}
        for node in self.node_types:
            namespace = node.namespace or "Uncategorized"
            if namespace not in namespace_groups:
                namespace_groups[namespace] = []
            namespace_groups[namespace].append(node)

        # Format the output
        output = []
        for namespace, nodes in sorted(namespace_groups.items()):
            output.append(f"Namespace: {namespace}")
            for node in sorted(nodes, key=lambda n: n.node_type):
                output.append(f"  - {node.node_type}: {node.description[:60]}...")

        return "\n".join(output)

    def _build_schema_for_selected_nodes(
        self, selected_nodes: list[NodeMetadata]
    ) -> dict:
        """
        Build a JSON schema for the create_workflow tool based on selected node types.

        Args:
            selected_nodes: List of selected NodeMetadata objects

        Returns:
            dict: JSON schema for create_workflow tool
        """
        # Base schema for the create_workflow tool
        schema = {
            "type": "object",
            "required": ["nodes"],
            "properties": {
                "nodes": {
                    "type": "array",
                    "description": "The nodes of the workflow",
                    "items": {
                        "oneOf": [],  # Will be filled with node-specific schemas
                    },
                },
            },
        }

        # Add individual node type schemas to the oneOf array
        for node in selected_nodes:
            node_schema = {
                "type": "object",
                "properties": {
                    "type": {"const": node.node_type},
                    "id": {"type": "string"},
                    "data": {
                        prop.name: self._create_property_schema(prop)
                        for prop in node.properties
                    },
                },
                "required": ["type", "id"],
            }
            # Add this node type schema to the oneOf array
            schema["properties"]["nodes"]["items"]["oneOf"].append(node_schema)

        # Log the generated schema
        self._log_trace_event("nodes_schema_generated", {"schema": schema})

        return schema

    def _create_property_schema(self, prop: Property) -> dict:
        """
        Create a JSON schema for a property that allows either the original type or a node reference.

        Args:
            prop: The property metadata

        Returns:
            dict: JSON schema for the property
        """
        # Get the original schema for the property
        original_schema = prop.get_json_schema()

        # Create a schema for node references
        reference_schema = {
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "sourceHandle": {"type": "string"},
            },
            "required": ["source", "sourceHandle"],
        }

        # Return a oneOf schema allowing either the original type or a reference
        return {"oneOf": [original_schema, reference_schema]}

    async def _second_stage_planning(
        self, selected_nodes: list[NodeMetadata], planning_summary: str
    ) -> dict:
        """
        Second stage of planning: create the workflow with selected node types.

        Args:
            selected_nodes: List of NodeMetadata objects selected in first stage
            planning_summary: Enhanced reasoning about node selection

        Returns:
            dict: The created workflow with nodes
        """
        console.print(
            Panel.fit(
                f"[bold blue]Starting second stage planning[/bold blue] with [yellow]{len(selected_nodes)}[/yellow] selected node types",
                border_style="blue",
            )
        )

        # Build schema for create_workflow tool based on selected nodes
        custom_schema = self._build_schema_for_selected_nodes(selected_nodes)

        # Log that we're using this schema for the create_workflow tool
        self._log_trace_event(
            "second_stage_planning_started",
            {
                "selected_node_count": len(selected_nodes),
                "using_schema": True,
                "schema_size": len(json.dumps(custom_schema)),
            },
        )

        # Create create_workflow tool with custom schema
        create_workflow_tool = CreateWorkflowTool(
            workspace_dir=self.workspace_dir, input_schema=custom_schema
        )

        # Setup conversation for second stage
        second_stage_system_prompt = """
        You are a workflow planning agent that creates optimized, executable graph workflows.
        
        YOUR TASK:
        Create a workflow using ONLY the node types that were selected in the previous stage.
        
        KEY WORKFLOW PLANNING PRINCIPLES:
        1. Break complex goals into nodes with clear responsibilities
        2. Connect nodes properly using source and target handles
        3. Ensure the workflow forms a Directed Acyclic Graph (DAG)
        4. Create self-contained nodes with minimal coupling
        5. Use appropriate properties for each node
        6. Provide all necessary properties for nodes to work correctly
        7. ALWAYS include an output node (like OutputNode, JsonOutputNode, etc.) to capture the final result
           of the workflow. Every workflow MUST have a way to access its results.
        """

        def get_node_info(node: NodeMetadata) -> str:
            info = f"""# {node.node_type}\n{node.description}\n## Properties\n"""
            for prop in node.properties:
                info += repr(prop)
            info += "\n## Outputs\n"
            for output in node.outputs:
                info += repr(output)
            return info + "\n"

        selected_node_info = "\n".join([get_node_info(node) for node in selected_nodes])

        # Pretty print selected node info and properties info
        console.print(
            "\n[bold cyan]===== SELECTED NODE TYPES INFORMATION =====[/bold cyan]"
        )
        console.print(f"[blue]{selected_node_info}[/blue]")
        console.print("\n[bold cyan]===== PLANNING SUMMARY =====[/bold cyan]")
        console.print(f"[yellow]{planning_summary}[/yellow]")
        console.print(
            "\n[bold cyan]==========================================[/bold cyan]\n"
        )

        agent_prompt = f"""
        Objective: {self.objective}
        
        Now, create a workflow using only these selected node types:
        {selected_node_info}
        
        Here is the reasoning about why these nodes were selected and how they should work together:
        {planning_summary}
        
        For each node:
        1. Provide an appropriate ID
        2. Select the correct node type from the available options
        3. Configure all required properties correctly
        4. Connect nodes properly with edges to form a DAG
        5. ALWAYS include an output node to make the workflow results accessible
        
        Use the create_workflow tool to generate the final workflow.
        """

        history = [
            Message(role="system", content=second_stage_system_prompt),
            Message(role="user", content=agent_prompt),
        ]

        # Execute second stage planning
        max_retries = 2  # Limit to 2 iterations
        current_retry = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Creating workflow...", total=None)

            while current_retry < max_retries:
                progress.update(
                    task,
                    description=f"[cyan]Second stage planning attempt {current_retry + 1}/{max_retries}[/cyan]",
                )

                message = await self.provider.generate_message(
                    messages=history,
                    model=self.model,
                    tools=[create_workflow_tool],
                )

                history.append(message)

                if not message.tool_calls:
                    error_msg = "No tool calls found in the message"
                    console.print(f"[bold red]Error:[/bold red] {error_msg}")
                    history.append(
                        Message(
                            role="user",
                            content=f"Error: {error_msg}. Please use the create_workflow tool.",
                        )
                    )
                    current_retry += 1
                    continue

                tool_call = next(
                    (tc for tc in message.tool_calls if tc.name == "create_workflow"),
                    None,
                )

                if not tool_call:
                    console.print(
                        "[bold red]Error:[/bold red] create_workflow tool not used"
                    )
                    history.append(
                        Message(
                            role="user",
                            content="Please use the create_workflow tool to create the workflow.",
                        )
                    )
                    current_retry += 1
                    continue

                nodes = tool_call.args.get("nodes", [])
                console.print(
                    f"[green]Workflow generated with {len(nodes)} nodes[/green]"
                )

                # Validate workflow
                validation_errors = self._validate_workflow(nodes)

                if validation_errors and current_retry < max_retries - 1:
                    console.print(
                        f"[bold yellow]Found {len(validation_errors)} validation errors:[/bold yellow]"
                    )

                    error_table = Table(
                        show_header=True,
                        header_style="bold red",
                        title="Validation Errors",
                    )
                    error_table.add_column("#", style="dim")
                    error_table.add_column("Error")

                    for i, error in enumerate(
                        validation_errors[:5], 1
                    ):  # Show first 5 errors
                        error_table.add_row(str(i), error)

                    if len(validation_errors) > 5:
                        error_table.add_row(
                            "...", f"and {len(validation_errors) - 5} more errors"
                        )

                    console.print(error_table)

                    self._log_trace_event(
                        "workflow_validation_errors",
                        {"retry": current_retry, "errors": validation_errors},
                    )

                    # Request corrections
                    retry_prompt = "Please fix the following errors in the workflow:\n"
                    for error in validation_errors:
                        retry_prompt += f"- {error}\n"

                    # Add a tool response message before sending the user message
                    history.append(
                        Message(
                            role="tool",
                            content=json.dumps(
                                {
                                    "errors": validation_errors,
                                    "status": "validation_failed",
                                }
                            ),
                            tool_call_id=tool_call.id,
                            name=tool_call.name,
                        )
                    )

                    history.append(Message(role="user", content=retry_prompt))
                    current_retry += 1
                else:
                    # Save workflow
                    workflow = {"nodes": nodes}

                    progress.stop()

                    if validation_errors:
                        console.print(
                            f"[bold yellow]Warning:[/bold yellow] Created workflow with {len(validation_errors)} unresolved validation issues (using final attempt)",
                        )
                    else:
                        console.print(
                            Panel.fit(
                                f"[bold green]Successfully created valid workflow[/bold green] with [yellow]{len(nodes)}[/yellow] nodes",
                                border_style="green",
                            )
                        )

                    # Log success
                    self._log_trace_event(
                        "workflow_created",
                        {"node_count": len(nodes)},
                    )

                    return workflow

        # If we get here, we've exhausted retries
        console.print(
            "[bold red]Failed to create a valid workflow after all retry attempts[/bold red]"
        )
        raise ValueError(
            f"Failed to create a valid workflow after {max_retries} attempts"
        )

    def _get_available_namespaces_info(self) -> str:
        """
        Get formatted string of available namespaces.

        Returns:
            str: Formatted string with namespace information
        """
        namespaces = self._get_available_namespaces()
        return "\n".join([f"- {ns}" for ns in namespaces])

    async def create_workflow(self) -> dict:
        """
        Create a workflow to accomplish the objective using the two-stage planning approach.

        Returns:
            dict: The created workflow with nodes
        """
        # Stage 1: Search for and select node types
        selected_nodes, justification = await self._first_stage_planning()

        if not selected_nodes:
            raise ValueError("Failed to select any node types in first stage planning")

        # Generate enhanced planning summary
        planning_summary = await self._generate_planning_summary(
            selected_nodes, justification
        )

        # Log selected nodes
        self._log_trace_event(
            "first_stage_complete",
            {
                "selected_node_count": len(selected_nodes),
                "selected_node_types": [node.node_type for node in selected_nodes],
                "planning_summary": planning_summary,
            },
        )

        # Stage 2: Create workflow with selected node types
        workflow = await self._second_stage_planning(selected_nodes, planning_summary)

        # Store the workflow
        self.workflow = transform_workflow(workflow)

        # Visualize the workflow
        self.visualize_workflow(self.workflow)

        return self.workflow

    async def _generate_planning_summary(
        self, selected_nodes: list[NodeMetadata], justification: str
    ) -> str:
        """
        Generate an enhanced summary of the first stage planning reasoning.

        Args:
            selected_nodes: List of selected NodeMetadata objects
            justification: Initial justification from node selection

        Returns:
            str: Enhanced planning summary
        """
        # Create a prompt to generate an enhanced reasoning summary
        system_prompt = """
        You are a workflow planning assistant. Your task is to analyze the selected nodes and 
        initial justification, then provide an enhanced reasoning about why these nodes are appropriate 
        for the workflow objective and how they should be connected.
        
        Provide a clear and concise summary that explains:
        1. Why each node type was selected
        2. How the nodes will work together to achieve the objective
        3. Any potential challenges or considerations in implementing this workflow
        4. Suggestions for node organization and connections
        """

        selected_node_info = "\n".join(
            [f"- {node.node_type}: {node.description}" for node in selected_nodes]
        )

        user_prompt = f"""
        Objective: {self.objective}
        
        Selected nodes:
        {selected_node_info}
        
        Initial justification:
        {justification}
        
        Please provide an enhanced reasoning about this node selection that will help in creating
        an effective workflow.
        """

        history = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

        # Generate the enhanced reasoning
        message = await self.provider.generate_message(
            messages=history,
            model=self.model,
        )

        # Log the enhanced reasoning
        self._log_trace_event("enhanced_planning_summary", {"summary": message.content})

        return str(message.content)

    def _get_node_properties_info(self, selected_nodes: list[NodeMetadata]) -> str:
        """
        Get formatted information about properties for each selected node type.

        Args:
            selected_nodes: List of selected NodeMetadata objects

        Returns:
            str: Formatted string with node property information
        """
        property_info = []

        for node in selected_nodes:
            node_info = [f"Properties for {node.node_type}:"]

            if not node.properties:
                node_info.append("  No configurable properties")
            else:
                for prop in node.properties:
                    # Get type information
                    type_info = prop.type.type
                    if prop.type.is_enum_type() and prop.type.values:
                        type_info += (
                            f" (options: {', '.join(str(v) for v in prop.type.values)})"
                        )
                    elif prop.type.is_list_type() and prop.type.type_args:
                        type_info += f" of {prop.type.type_args[0].type}"

                    # Add required/optional status
                    status = "required" if not prop.type.optional else "optional"

                    # Add description if available
                    description = f": {prop.description}" if prop.description else ""

                    node_info.append(
                        f"  - {prop.name} ({type_info}, {status}){description}"
                    )

            property_info.append("\n".join(node_info))

        return "\n\n".join(property_info)

    def visualize_workflow(self, workflow: dict | None = None) -> None:
        """
        Visualize the workflow as a network graph in the terminal using rich.

        Args:
            workflow: Optional workflow to visualize. If None, uses self.workflow.
        """
        if workflow is None:
            workflow = self.workflow

        if workflow is None:
            console.print("[bold red]No workflow to visualize[/bold red]")
            return

        nodes = {node["id"]: node for node in workflow["nodes"]}

        # Create a directed graph representation
        G = nx.DiGraph()
        for node in workflow["nodes"]:
            G.add_node(node["id"], type=node["type"])

        for edge in workflow.get("edges", []):
            G.add_edge(
                edge["source"],
                edge["target"],
                sourceHandle=edge.get("sourceHandle", "output"),
                targetHandle=edge.get("targetHandle", "input"),
            )

        console.print(
            Panel.fit(
                f"[bold blue]Workflow Visualization[/bold blue] - [yellow]{len(nodes)}[/yellow] nodes, "
                f"[yellow]{len(workflow.get('edges', []))}[/yellow] edges",
                border_style="blue",
            )
        )

        # Create a network-style visualization
        table = Table(show_header=False, box=None, padding=0)

        # Calculate node positions using networkx layout algorithm
        try:
            # Try to use a hierarchical layout
            pos = (
                nx.nx_agraph.graphviz_layout(G, prog="dot")
                if nx.nx_agraph.graphviz_layout
                else nx.spring_layout(G)
            )
        except:
            # Fall back to spring layout if graphviz is not available
            pos = nx.spring_layout(G, seed=42)

        # Normalize positions to fit in the terminal
        x_values = [coord[0] for coord in pos.values()]
        y_values = [coord[1] for coord in pos.values()]

        if x_values and y_values:  # Make sure we have positions
            min_x, max_x = min(x_values), max(x_values)
            min_y, max_y = min(y_values), max(y_values)

            # Normalize to terminal size
            width = 60
            height = 20

            normalized_pos = {}
            for node_id, (x, y) in pos.items():
                # Normalize coordinates
                if max_x > min_x:
                    norm_x = int((x - min_x) / (max_x - min_x) * (width - 1))
                else:
                    norm_x = 0

                if max_y > min_y:
                    norm_y = int((y - min_y) / (max_y - min_y) * (height - 1))
                else:
                    norm_y = 0

                normalized_pos[node_id] = (norm_x, norm_y)

            # Create a grid for the graph
            grid = [[" " for _ in range(width)] for _ in range(height)]

            # Place nodes on the grid
            node_positions = {}
            for node_id, (x, y) in normalized_pos.items():
                node_positions[node_id] = (x, y)
                grid[y][x] = "●"

            # Draw edges with simple lines
            for source, target in G.edges():
                x1, y1 = node_positions[source]
                x2, y2 = node_positions[target]

                # Draw a simple line between nodes
                self._draw_line(grid, x1, y1, x2, y2)

            # Convert grid to a string and print it
            for row in grid:
                table.add_row("".join(row))

            console.print(table)

            # Add node legend
            legend = Table(show_header=True, header_style="bold cyan")
            legend.add_column("Node ID")
            legend.add_column("Type")

            for node_id, node in nodes.items():
                legend.add_row(f"[green]{node_id}[/green]", node["type"])

            console.print(Panel.fit(legend, title="Node Legend", border_style="blue"))

        # Print node details
        self._print_node_details(nodes)

    def _draw_line(self, grid, x1, y1, x2, y2):
        """
        Draw a simple line between two points on the grid using Bresenham's algorithm.

        Args:
            grid: 2D grid to draw on
            x1, y1: Coordinates of the start point
            x2, y2: Coordinates of the end point
        """
        # Implementation of Bresenham's line algorithm
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            # Skip the endpoints as they contain the node markers
            if (
                (x1, y1) != (x2, y2)
                and 0 <= y1 < len(grid)
                and 0 <= x1 < len(grid[0])
                and grid[y1][x1] == " "
            ):
                if dx > dy:
                    grid[y1][x1] = "─"  # Horizontal line
                else:
                    grid[y1][x1] = "│"  # Vertical line

            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

    def _print_node_details(self, nodes):
        """
        Print detailed information about each node.

        Args:
            nodes: Dictionary of node ID to node data
        """
        console.print("\n[bold cyan]Node Details:[/bold cyan]")

        for node_id, node in nodes.items():
            panel = Panel(
                Text.from_markup(f"[bold]Type:[/bold] {node['type']}"),
                title=f"[green]{node_id}[/green]",
                border_style="blue",
            )
            console.print(panel)

            if node.get("data"):
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Property")
                table.add_column("Value")

                for prop, value in node["data"].items():
                    if isinstance(value, dict) and "source" in value:
                        val_str = f"[cyan]Reference:[/cyan] {value['source']} → {value['sourceHandle']}"
                    else:
                        val_str = str(value)
                        if len(val_str) > 50:
                            val_str = val_str[:47] + "..."
                    table.add_row(prop, val_str)

                console.print(table)


def transform_workflow(workflow: dict) -> dict:
    """
    Transform the workflow from direct node references to explicit edges.
    """
    nodes = workflow["nodes"]
    edges = []
    for node in nodes:
        for name, prop in node["data"].items():
            if isinstance(prop, dict) and "source" in prop:
                edges.append(
                    {
                        "source": prop["source"],
                        "sourceHandle": prop["sourceHandle"],
                        "target": node["id"],
                        "targetHandle": name,
                    }
                )
    return {"nodes": nodes, "edges": edges}


if __name__ == "__main__":
    registry = Registry()
    installed_packages = registry.list_installed_packages()
    nodes = []
    for package in installed_packages:
        if package.nodes:
            nodes.extend(package.nodes)

    async def main():
        context = ProcessingContext()
        planner = WorkflowPlanner(
            provider=get_provider(Provider.Anthropic),
            model="claude-3-7-sonnet-20250219",
            # provider=get_provider(Provider.OpenAI),
            # model="gpt-4o",
            # objective="Fetch the top 10 posts from the Hacker News website and extract the titles",
            # objective="Draw a red circle on a white background",
            objective="""
            Research the competitive landscape of AI code assistant tools.
            1. Use google search and browser to identify a list of AI code assistant tools
            2. For each tool, identify the following information:   
                - Name of the tool
                - Description of the tool
                - Key features of the tool
                - Pricing information
            - User reviews
            - Comparison with other tools
            3. Summarize the findings in a table format""",
            workspace_dir=context.workspace_dir,
            node_types=nodes,
            enable_tracing=True,
        )
        workflow = await planner.create_workflow()
        console.print(json.dumps(workflow, indent=2), style="blue")

        # Explicitly visualize the workflow (it's already called in create_workflow but adding here for clarity)
        planner.visualize_workflow(workflow)

        graph = Graph(**workflow)

        async for msg in run_workflow(RunJobRequest(graph=graph)):
            console.print(msg, style="green")

    asyncio.run(main())
