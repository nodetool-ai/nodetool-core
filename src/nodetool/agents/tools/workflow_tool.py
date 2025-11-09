"""
Tool for executing specific workflows as agent tools.

This module provides WorkflowTool, which allows workflows to be used as tools
by agents. Each WorkflowTool instance is configured with a specific workflow
and uses its input schema for tool parameters.
"""

from nodetool.config.logging_config import get_logger
import re
from typing import Any, Dict
from uuid import uuid4

from nodetool.agents.tools.base import Tool
from nodetool.config.environment import Environment
from nodetool.types.workflow import Workflow
from nodetool.models.workflow import Workflow as WorkflowModel
from nodetool.types.graph import Edge, Node, get_input_schema, get_output_schema
from nodetool.workflows.base_node import BaseNode, ToolResultNode
from nodetool.workflows.graph import Graph
from nodetool.workflows.processing_context import (
    AssetOutputMode,
    ProcessingContext,
)
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.types import (
    JobUpdate,
    OutputUpdate,
    ToolResultUpdate,
)

log = get_logger(__name__)
# Log level is controlled by env (DEBUG/NODETOOL_LOG_LEVEL)


def from_model(workflow: WorkflowModel):
    api_graph = workflow.get_api_graph()

    return Workflow(
        id=workflow.id,
        access=workflow.access,
        created_at=workflow.created_at.isoformat(),
        updated_at=workflow.updated_at.isoformat(),
        name=workflow.name,
        package_name=workflow.package_name,
        tags=workflow.tags,
        description=workflow.description or "",
        thumbnail=workflow.thumbnail or "",
        graph=api_graph,
        input_schema=get_input_schema(api_graph),
        output_schema=get_output_schema(api_graph),
        settings=workflow.settings,
        run_mode=workflow.run_mode,
    )


def sanitize_workflow_name(name: str) -> str:
    """
    Convert workflow name to tool name format.

    Args:
        node_name: The workflow name string.

    Returns:
        The sanitized tool name.
    """
    # replace non alphanumeric characters
    node_name = re.sub(r"[^a-zA-Z0-9]", "", name)
    max_length = 64
    if len(node_name) > max_length:
        return node_name[:max_length]
    else:
        return node_name


class GraphTool(Tool):
    """Tool that executes a specific graph using its input schema."""

    def __init__(
        self,
        graph: Graph,
        name: str,
        description: str,
        initial_edges: list[Edge],
        initial_nodes: list[BaseNode],
    ):
        self.graph = graph
        self.name = name
        self.description = description
        self.initial_edges = initial_edges
        self.initial_nodes = initial_nodes

        def get_property_schema(node: BaseNode, handle: str) -> dict[str, Any]:
            for prop in node.properties():
                if prop.name == handle:
                    schema = prop.type.get_json_schema()
                    schema["description"] = prop.description or ""
                    return schema
            raise ValueError(
                f"Property {handle} not found on node {node.get_node_type()} {node.id}"
            )

        self.input_schema = {
            "type": "object",
            "properties": {
                edge.targetHandle: get_property_schema(node, edge.targetHandle)
                for edge, node in zip(self.initial_edges, self.initial_nodes)
            },
        }

    async def process(self, context: ProcessingContext, params: Dict[str, Any]) -> Any:
        from nodetool.types.graph import Graph as ApiGraph
        from nodetool.workflows.workflow_runner import WorkflowRunner

        initial_edges_by_target = {edge.target: edge for edge in self.initial_edges}
        excluded_source_ids = {edge.source for edge in self.initial_edges}

        def properties_for_node(node: BaseNode) -> dict[str, Any]:
            props = node.node_properties()
            if node.id in initial_edges_by_target:
                edge = initial_edges_by_target[node.id]
                if edge.targetHandle not in params:
                    raise ValueError(f"Missing required parameter: {edge.targetHandle}")
                props[edge.targetHandle] = params[edge.targetHandle]
            return props

        # Build the base node list for the API graph
        nodes = [
            Node(
                id=node.id,
                type=node.get_node_type(),
                data=properties_for_node(node),
                parent_id=node.parent_id,
                ui_properties=node.ui_properties,
                dynamic_properties=node.dynamic_properties,
                dynamic_outputs=node.dynamic_outputs,
            )
            for node in self.graph.nodes
            if node.id not in excluded_source_ids
        ]

        # Start with existing edges
        edges = [
            Edge(
                id=edge.id,
                source=edge.source,
                target=edge.target,
                sourceHandle=edge.sourceHandle,
                targetHandle=edge.targetHandle,
                ui_properties=edge.ui_properties,
            )
            for edge in self.graph.edges
            if edge.source not in excluded_source_ids
            and edge.target not in excluded_source_ids
        ]

        # Check if there's already a ToolResultNode in the graph
        has_tool_result_node = any(
            isinstance(node, ToolResultNode) for node in self.graph.nodes
            if node.id not in excluded_source_ids
        )

        # If there is only one node in the graph and no ToolResult node,
        # automatically add a ToolResult node and connect all outputs from the single node to it.
        if len([n for n in self.graph.nodes if n.id not in excluded_source_ids]) == 1 and not has_tool_result_node:
            single_node = next(n for n in self.graph.nodes if n.id not in excluded_source_ids)
            result_node_id = uuid4().hex

            # Append ToolResult node to API graph
            nodes.append(
                Node(
                    id=result_node_id,
                    type=ToolResultNode.get_node_type(),
                    data={},
                    parent_id=None,
                    ui_properties={},
                    dynamic_properties={},
                    dynamic_outputs={},
                )
            )

            # Create edges from each output of the single node to the ToolResult node
            for output_slot in single_node.outputs_for_instance():
                edges.append(
                    Edge(
                        id=uuid4().hex,
                        source=single_node.id,
                        target=result_node_id,
                        sourceHandle=output_slot.name,
                        targetHandle=output_slot.name,
                        ui_properties={},
                    )
                )
            has_tool_result_node = True

        try:
            req = RunJobRequest(
                user_id=context.user_id,
                auth_token=context.auth_token,
                graph=ApiGraph(
                    nodes=nodes,
                    edges=edges,
                ),
            )
            assert req.graph is not None
            # Use an isolated message queue for the subgraph to avoid draining
            # the parent's queue. We will forward all messages back to the
            # parent context so external listeners still see NodeUpdate/OutputUpdate.
            import queue as _queue

            isolated_queue: _queue.Queue = _queue.Queue()
            sub_context = ProcessingContext(
                user_id=context.user_id,
                auth_token=context.auth_token,
                graph=Graph.from_dict(req.graph.model_dump()),
                message_queue=isolated_queue,
                device=context.device,
                workspace_dir=context.workspace_dir,
                asset_output_mode=getattr(
                    context, "asset_output_mode", AssetOutputMode.TEMP_URL
                ),
            )

            # Collect all messages from workflow execution
            result = {}
            runner = WorkflowRunner(job_id=uuid4().hex, disable_caching=True)
            
            # Determine if we need to capture outputs from leaf nodes
            # (when there's no ToolResultNode)
            need_leaf_output = not has_tool_result_node
            
            # Find leaf nodes (nodes with no outgoing edges) if needed
            leaf_node_id: str | None = None
            leaf_output_slot: str | None = None
            if need_leaf_output:
                # Build set of node IDs that have outgoing edges in the final graph
                nodes_with_outgoing = {edge.source for edge in edges}
                # Find nodes that are in the graph but have no outgoing edges
                # (these are the leaf/terminal nodes)
                leaf_nodes = [
                    node for node in self.graph.nodes
                    if node.id not in excluded_source_ids
                    and node.id not in nodes_with_outgoing
                ]
                
                if len(leaf_nodes) == 1:
                    leaf_node = leaf_nodes[0]
                    outputs = leaf_node.outputs_for_instance()
                    # Check for default output: "output" slot or exactly one output slot
                    default_output = None
                    for output_slot in outputs:
                        if output_slot.name == "output":
                            default_output = output_slot.name
                            break
                    if default_output is None and len(outputs) == 1:
                        default_output = outputs[0].name
                    
                    if default_output:
                        leaf_node_id = leaf_node.id
                        leaf_output_slot = default_output
                    else:
                        raise ValueError(
                            f"Tool graph has no ToolResult node and the leaf node "
                            f"({leaf_node.get_node_type()}) has no default output. "
                            f"Available outputs: {[o.name for o in outputs]}. "
                            f"Either add a ToolResult node or ensure the leaf node has an 'output' slot."
                        )
                elif len(leaf_nodes) > 1:
                    raise ValueError(
                        f"Tool graph has no ToolResult node and multiple leaf nodes "
                        f"({len(leaf_nodes)}). Cannot determine which node's output to use. "
                        f"Either add a ToolResult node or ensure there's only one leaf node with a default output."
                    )
                elif len(leaf_nodes) == 0:
                    raise ValueError(
                        "Tool graph has no ToolResult node and no leaf nodes found. "
                        "Either add a ToolResult node or ensure the graph has terminal nodes."
                    )
            
            async for msg in run_workflow(
                request=req,
                runner=runner,
                context=sub_context,
                use_thread=True,
                send_job_updates=False,
                initialize_graph=False,
                validate_graph=False,
            ):
                # Forward all subgraph messages to the parent context
                # but not JobUpdate to prevent early termination
                if not isinstance(msg, JobUpdate):
                    context.post_message(msg)
                if isinstance(msg, ToolResultUpdate):
                    update: ToolResultUpdate = msg
                    if update.result is not None:
                        for key, value in update.result.items():
                            if hasattr(value, "model_dump"):
                                value = value.model_dump()
                            if result.get(key) is None:
                                result[key] = value
                            elif isinstance(result[key], list):
                                result[key].append(value)
                            elif isinstance(result[key], str):
                                result[key] += value
                            else:
                                result[key] = value
                elif need_leaf_output and isinstance(msg, OutputUpdate):
                    # Capture output from leaf node if no ToolResult node
                    if msg.node_id == leaf_node_id and msg.output_name == leaf_output_slot:
                        value = msg.value
                        if hasattr(value, "model_dump"):
                            value = value.model_dump()
                        # Use output slot name as key, or "output" if it's the default
                        key = leaf_output_slot if leaf_output_slot != "output" else "output"
                        if result.get(key) is None:
                            result[key] = value
                        elif isinstance(result[key], list):
                            result[key].append(value)
                        elif isinstance(result[key], str):
                            result[key] += value
                        else:
                            result[key] = value

            print("result", result)

            # Return the collected results
            return result

        except Exception as e:
            return str(e)


class WorkflowTool(Tool):
    """Tool that executes a specific workflow using its input schema."""

    def __init__(self, workflow: Workflow):
        """
        Initialize the WorkflowTool with a specific workflow.

        Args:
            workflow: The Workflow model instance to use for this tool
        """
        self.workflow = workflow

        # Set tool metadata from workflow
        self.name = f"workflow_{workflow.tool_name}"
        self.description = workflow.description or ""

        assert workflow.input_schema is not None, "Workflow input schema is required"
        self.input_schema = workflow.input_schema

    async def process(self, context: ProcessingContext, params: Dict[str, Any]) -> Any:
        """
        Execute the workflow with the provided parameters.

        Args:
            context: The processing context
            params: Input parameters matching the workflow's input schema

        Returns:
            The workflow execution results
        """
        try:
            req = RunJobRequest(
                user_id=context.user_id,
                auth_token=context.auth_token,
                workflow_id=self.workflow.id,
                graph=self.workflow.graph,
                params=params,
            )

            # Collect all messages from workflow execution
            results = {}
            async for msg in run_workflow(req, context=context, use_thread=True):
                if isinstance(msg, OutputUpdate):
                    value = msg.value
                    if hasattr(value, "model_dump"):
                        value = value.model_dump()
                    results[msg.node_name] = value

            log.debug(f"Workflow tool {self.name} returned: {results}")

            return results

        except Exception as e:
            return {
                "error": str(e),
            }

    def user_message(self, params: Dict[str, Any]) -> str:
        """
        Returns a user message for the workflow tool.
        """
        return f"Executing workflow '{self.workflow.name}' with parameters: {params}"


async def create_workflow_tools(user_id: str, limit: int = 1000) -> list[WorkflowTool]:
    """
    Create WorkflowTool instances for all workflows accessible to a user.

    Args:
        user_id: The user ID to get workflows for
        limit: Maximum number of workflows to load

    Returns:
        List of WorkflowTool instances
    """
    if not Environment.has_database():
        return []
    workflows, _ = await WorkflowModel.paginate(user_id=user_id, limit=limit)
    return [WorkflowTool(from_model(workflow)) for workflow in workflows]


async def create_workflow_tool_by_name(
    user_id: str, workflow_name: str
) -> WorkflowTool | None:
    """
    Create a WorkflowTool instance for a specific workflow by name.

    Args:
        user_id: The user ID to get workflows for
        workflow_name: Name of the workflow to find

    Returns:
        WorkflowTool instance if found, None otherwise
    """
    try:
        workflow = await WorkflowModel.find(user_id, workflow_name)
        if not workflow:
            return None
        return WorkflowTool(from_model(workflow))
    except Exception as e:
        print(
            f"Warning: Could not load workflow '{workflow_name}' for user {user_id}: {e}"
        )
        return None
