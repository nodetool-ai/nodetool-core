"""
Tool for executing specific workflows as agent tools.

This module provides WorkflowTool, which allows workflows to be used as tools
by agents. Each WorkflowTool instance is configured with a specific workflow
and uses its input schema for tool parameters.
"""

import json
from typing import Any, Dict
from uuid import uuid4

from nodetool.agents.tools.base import Tool
from nodetool.metadata.types import AssetRef
from nodetool.models.asset import Asset
from nodetool.models.workflow import Workflow
from nodetool.types.graph import get_input_schema, get_output_schema
from nodetool.types.job import JobUpdate
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.types import NodeUpdate, OutputUpdate


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
        self.name = f"workflow_{workflow.name.lower().replace(' ', '_').replace('-', '_')}"
        self.description = f"Execute workflow: {workflow.name}"
        if workflow.description:
            self.description += f" - {workflow.description}"
        
        graph = workflow.get_api_graph()
        schema = get_input_schema(graph)
        self.input_schema = schema

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
            # Get the workflow graph
            graph = self.workflow.get_api_graph()
            
            # Create workflow request
            req = RunJobRequest(
                user_id=context.user_id,
                auth_token=context.auth_token,
                workflow_id=self.workflow.id,
                graph=graph,
                params=params
            )
            
            # Collect all messages from workflow execution
            results = {}
            async for msg in run_workflow(req, context=context, use_thread=True):
                if isinstance(msg, OutputUpdate):
                    value = context.upload_assets_to_temp(msg.value)
                    if hasattr(value, "model_dump"):
                        value = value.model_dump()
                    results[msg.node_name] = value
            
            # Return the collected results
            return {
                "workflow_name": self.workflow.name,
                "results": results,
                "status": "completed"
            }
            
        except Exception as e:
            return {
                "workflow_name": self.workflow.name,
                "error": str(e),
                "status": "failed"
            }

    def user_message(self, params: Dict[str, Any]) -> str:
        """
        Returns a user message for the workflow tool.
        """
        return f"Executing workflow '{self.workflow.name}' with parameters: {params}"


def create_workflow_tools(user_id: str, limit: int = 1000) -> list[WorkflowTool]:
    """
    Create WorkflowTool instances for all workflows accessible to a user.
    
    Args:
        user_id: The user ID to get workflows for
        limit: Maximum number of workflows to load
        
    Returns:
        List of WorkflowTool instances
    """
    try:
        workflows, _ = Workflow.paginate(user_id=user_id, limit=limit)
        return [WorkflowTool(workflow) for workflow in workflows if workflow.run_mode == "tool"]
    except Exception as e:
        print(f"Warning: Could not load workflows for user {user_id}: {e}")
        return []


def create_workflow_tool_by_name(user_id: str, workflow_name: str) -> WorkflowTool | None:
    """
    Create a WorkflowTool instance for a specific workflow by name.
    
    Args:
        user_id: The user ID to get workflows for
        workflow_name: Name of the workflow to find
        
    Returns:
        WorkflowTool instance if found, None otherwise
    """
    try:
        workflows, _ = Workflow.paginate(user_id=user_id, limit=1000)
        
        for workflow in workflows:
            if workflow.name == workflow_name:
                return WorkflowTool(workflow)
        
        return None
    except Exception as e:
        print(f"Warning: Could not load workflow '{workflow_name}' for user {user_id}: {e}")
        return None