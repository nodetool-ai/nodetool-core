"""
Tool for executing specific workflows as agent tools.

This module provides WorkflowTool, which allows workflows to be used as tools
by agents. Each WorkflowTool instance is configured with a specific workflow
and uses its input schema for tool parameters.
"""

import json
import logging
import re
from typing import Any, Dict
from uuid import uuid4

from nodetool.agents.tools.base import Tool
from nodetool.common.environment import Environment
from nodetool.types.workflow import Workflow
from nodetool.models.workflow import Workflow as WorkflowModel
from nodetool.types.graph import Edge, Node, get_input_schema, get_output_schema
from nodetool.types.job import JobUpdate
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.graph import Graph
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.types import NodeUpdate, OutputUpdate

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


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

        log.debug(f"inital edges: {initial_edges}")
        log.debug(f"inital nodes: {initial_nodes}")
        log.debug(f"input schema: {self.input_schema}")

    async def process(self, context: ProcessingContext, params: Dict[str, Any]) -> Any:
        from nodetool.types.graph import Graph as ApiGraph

        log.debug(f"calling tool {self.name}")
        log.debug(f"initial edges: {self.initial_edges}")
        log.debug(f"initial nodes: {self.initial_nodes}")

        initial_edges_by_target = {edge.target: edge for edge in self.initial_edges}

        def properties_for_node(node: BaseNode) -> dict[str, Any]:
            props = node.node_properties()
            if node.id in initial_edges_by_target:
                edge = initial_edges_by_target[node.id]
                if not edge.targetHandle in params:
                    raise ValueError(f"Missing required parameter: {edge.targetHandle}")
                props[edge.targetHandle] = params[edge.targetHandle]
            return props

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
        ]

        try:
            req = RunJobRequest(
                user_id=context.user_id,
                auth_token=context.auth_token,
                graph=ApiGraph(
                    nodes=nodes,
                    edges=[
                        Edge(
                            id=edge.id,
                            source=edge.source,
                            target=edge.target,
                            sourceHandle=edge.sourceHandle,
                            targetHandle=edge.targetHandle,
                            ui_properties=edge.ui_properties,
                        )
                        for edge in self.graph.edges
                    ],
                ),
            )

            # Collect all messages from workflow execution
            results = {}
            async for msg in run_workflow(req, context=context, use_thread=True):
                if isinstance(msg, NodeUpdate):
                    update: NodeUpdate = msg
                    if update.node_name == "ToolResultNode":
                        results = update.result

            log.debug(f"tool results before upload: {results}")
            results = await context.upload_assets_to_temp(results)
            log.debug(f"tool results: {results}")

            # Return the collected results
            return results

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
        self.name = f"workflow_{workflow.id}"
        self.description = f"Execute workflow: {workflow.name}"
        if workflow.description:
            self.description += f" - {workflow.description}"

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

            results = await context.upload_assets_to_temp(results)

            # Return the collected results
            return {
                "workflow_name": self.workflow.name,
                "results": results,
                "status": "completed",
            }

        except Exception as e:
            return {
                "workflow_name": self.workflow.name,
                "error": str(e),
                "status": "failed",
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
