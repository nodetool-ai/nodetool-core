"""
Finish Step Tool for signaling step completion.

This tool provides a reliable mechanism for Claude models to signal step completion
with a validated result, replacing the error-prone JSON parsing approach.
"""

from typing import Any

from nodetool.config.logging_config import get_logger
from nodetool.workflows.processing_context import ProcessingContext

from .base import Tool

logger = get_logger(__name__)


class FinishStepTool(Tool):
    """
    Tool for signaling step completion with a validated result.

    This tool is automatically injected into the StepExecutor when a step has
    an output schema. The LLM calls this tool to signal completion, which is
    more reliable than parsing JSON blocks from the response.

    The input schema is dynamically generated based on the step's output schema,
    allowing Anthropic's strict tool validation to enforce the result format.
    """

    name: str = "finish_step"
    description: str = """Call this tool when you have completed the step and have the final result ready.

This is the ONLY way to properly signal step completion. Do not output raw JSON blocks.

When to use:
- You have gathered all necessary information
- You have processed/transformed the data as required
- Your result matches the expected output schema

The result must conform to the step's declared output schema."""

    # Will be set dynamically based on step's output_schema
    _result_schema: dict[str, Any] | None
    _input_schema: dict[str, Any]

    def __init__(self, result_schema: dict[str, Any] | None = None):
        """
        Initialize the finish_step tool with the expected result schema.

        Args:
            result_schema: The JSON schema that the result must conform to.
                          This is typically from Step.output_schema.
        """
        self._result_schema = result_schema
        self._input_schema = self._build_input_schema()

    def _build_input_schema(self) -> dict[str, Any]:
        """Build the tool's input schema based on the result schema."""
        if self._result_schema is None:
            # If no schema, accept any object
            return {
                "type": "object",
                "properties": {
                    "result": {
                        "type": "object",
                        "description": "The result of the step. Can be any JSON object.",
                        "additionalProperties": True,
                    }
                },
                "required": ["result"],
                "additionalProperties": False,
            }

        # Build schema with result field matching the step's output schema
        result_schema_copy = dict(self._result_schema)

        # Ensure the result schema has additionalProperties: false for strict mode
        if result_schema_copy.get("type") == "object" and "additionalProperties" not in result_schema_copy:
            result_schema_copy["additionalProperties"] = False

        return {
            "type": "object",
            "properties": {
                "result": result_schema_copy,
            },
            "required": ["result"],
            "additionalProperties": False,
        }

    @property
    def input_schema(self) -> dict[str, Any]:
        """Return the dynamically generated input schema."""
        return self._input_schema

    @input_schema.setter
    def input_schema(self, value: dict[str, Any]) -> None:
        """Allow setting input_schema (for compatibility)."""
        self._input_schema = value

    def user_message(self, params: dict[str, Any]) -> str:
        """Return a user-facing message for this tool call."""
        return "Completing step with result..."

    async def process(self, context: ProcessingContext, params: dict[str, Any]) -> dict[str, Any]:
        """
        Process the finish_step tool call.

        Note: The actual completion handling is done in StepExecutor._handle_finish_step_tool.
        This method just returns the result for logging/history purposes.

        Args:
            context: The processing context.
            params: The tool parameters, containing the 'result' field.

        Returns:
            A dict indicating completion with the result.
        """
        result = params.get("result")
        logger.info(f"finish_step tool called with result type: {type(result)}")
        return {
            "status": "completed",
            "result": result,
        }
