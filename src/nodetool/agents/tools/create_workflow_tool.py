"""Tool for creating workflows through natural language objectives via WebSocket."""

import logging
from typing import Any, Dict
from pydantic import Field
from nodetool.agents.tools.base import Tool
from nodetool.workflows.processing_context import ProcessingContext

log = logging.getLogger(__name__)


class CreateWorkflowTool(Tool):
    """
    Tool that triggers workflow creation using GraphPlanner.
    
    This tool detects user intent to create workflows and initiates the
    workflow creation process through the WebSocket chat system. It leverages
    the existing GraphPlanner infrastructure to convert natural language
    objectives into executable workflow graphs.
    """
    name = "create_workflow"
    description = "Creates a new workflow from a natural language objective. Use this tool when the user wants to create, build, generate, or design a workflow. The tool will use AI to convert the objective into a structured workflow graph with appropriate nodes and connections."
    input_schema = {
        "type": "object",
        "properties": {
            "objective": {"type": "string", "description": """
                          A clear description of what the workflow should accomplish.
                          This should be a complete sentence describing the goal, process, or task that the workflow needs to perform.
                          """
            }
        },
        "required": ["objective"]
    }
    example = """
    Example:
    "Process sales data from CSV file and generate quarterly summary report"  
    "Analyze customer feedback, extract sentiment, and create insights dashboard"
    "Download images from URLs, resize them, and apply watermarks"
    "Extract text from PDF documents and translate to multiple languages"
    """

    async def process(self, context: ProcessingContext, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a workflow creation request.
        
        This method doesn't actually create the workflow directly. Instead, it
        returns metadata that the WebSocket runner can use to trigger the
        workflow creation process using the existing _process_create_workflow method.
        
        Args:
            context: The processing context
            params: Tool arguments, should contain 'objective'
            
        Returns:
            Dict containing workflow creation trigger information
        """
        objective = params.get("objective", "").strip()
        
        if not objective:
            return {
                "error": "No objective provided. Please specify what the workflow should accomplish.",
                "success": False
            }
        
        log.info(f"Workflow creation requested with objective: {objective}")
        
        # Return trigger information for the WebSocket runner to handle
        return {
            "action": "create_workflow",
            "objective": objective,
            "message": f"Initiating workflow creation for: {objective}",
            "success": True,
        }

    def user_message(self, args: Dict[str, Any]) -> str:
        """Return a user-friendly message about the tool execution."""
        objective = args.get("objective", "")
        return f"Creating workflow for: {objective}"
