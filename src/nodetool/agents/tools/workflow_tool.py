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
from nodetool.common.environment import Environment
from nodetool.types.workflow import Workflow
from nodetool.models.workflow import Workflow as WorkflowModel
from nodetool.types.graph import get_input_schema, get_output_schema
from nodetool.types.job import JobUpdate
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.types import NodeUpdate, OutputUpdate

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
    if not Environment.has_database():
        return []
    workflows, _ = WorkflowModel.paginate(user_id=user_id, limit=limit)
    return [WorkflowTool(from_model(workflow)) for workflow in workflows]


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
        workflows, _ = WorkflowModel.paginate(user_id=user_id, limit=1000)
        
        for workflow in workflows:
            if workflow.name == workflow_name:
                return WorkflowTool(from_model(workflow))
        
        return None
    except Exception as e:
        print(f"Warning: Could not load workflow '{workflow_name}' for user {user_id}: {e}")
        return None