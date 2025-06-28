"""Tool for editing workflows through natural language objectives via WebSocket."""

import logging
from typing import Any, Dict
from nodetool.agents.tools.base import Tool
from nodetool.workflows.processing_context import ProcessingContext

log = logging.getLogger(__name__)


class EditWorkflowTool(Tool):
    """
    Tool that triggers workflow editing using GraphPlanner.

    This tool detects user intent to edit existing workflows and initiates the
    workflow editing process through the WebSocket chat system. It leverages
    the existing GraphPlanner infrastructure to modify workflow graphs based on
    natural language objectives while preserving existing structure.
    """

    name = "edit_workflow"
    description = "Edits an existing workflow from a natural language objective. Use this tool when the user wants to modify, update, enhance, or change an existing workflow. The tool will use AI to modify the existing workflow graph based on the provided objective while preserving relevant existing nodes and connections."
    input_schema = {
        "type": "object",
        "properties": {
            "objective": {
                "type": "string",
                "description": """
                A clear description of what changes should be made to the workflow.
                This should be a complete sentence describing the modifications, enhancements, or updates needed.
                """,
            },
            "workflow_id": {
                "type": "string",
                "description": "The ID of the existing workflow to edit (optional - will be inferred from context if not provided)",
            },
        },
        "required": ["objective"],
    }
    example = """
    Example:
    "Add sentiment analysis to the existing customer feedback workflow"
    "Modify the data processing pipeline to include outlier detection"  
    "Update the report generation workflow to include charts and visualizations"
    "Enhance the image processing workflow with watermark functionality"
    """

    async def process(
        self, context: ProcessingContext, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a workflow editing request.

        This method doesn't actually edit the workflow directly. Instead, it
        returns metadata that the WebSocket runner can use to trigger the
        workflow editing process using the existing GraphPlanner with an
        existing_graph parameter.

        Args:
            context: The processing context
            params: Tool arguments, should contain 'objective' and optionally 'workflow_id'

        Returns:
            Dict containing workflow editing trigger information
        """
        objective = params.get("objective", "").strip()
        workflow_id = params.get("workflow_id", "").strip()

        if not objective:
            return {
                "error": "No objective provided. Please specify what changes should be made to the workflow.",
                "success": False,
            }

        log.info(f"Workflow editing requested with objective: {objective}")
        if workflow_id:
            log.info(f"Target workflow ID: {workflow_id}")

        # Return trigger information for the WebSocket runner to handle
        return {
            "action": "edit_workflow",
            "objective": objective,
            "workflow_id": workflow_id if workflow_id else None,
            "message": f"Initiating workflow editing for: {objective}",
            "success": True,
        }

    def user_message(self, args: Dict[str, Any]) -> str:
        """Return a user-friendly message about the tool execution."""
        objective = args.get("objective", "")
        workflow_id = args.get("workflow_id", "")
        if workflow_id:
            return f"Editing workflow {workflow_id} for: {objective}"
        else:
            return f"Editing workflow for: {objective}"
