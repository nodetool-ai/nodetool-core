"""
🧠 Simple SubTask Context: Value-based Task Execution

A simplified version of SubTaskContext that operates on Python values instead of files.
This version maintains output type handling while removing file operations and workspace management.

Key simplifications:
- Works with Python values (strings, dicts, lists) instead of files
- No workspace or file system operations
- Simplified tool system focused on value processing
- Maintains core execution loop and output type handling
- No binary file handling or artifact management
"""

import asyncio
import json
from nodetool.config.logging_config import get_logger
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Sequence, Union

from nodetool.agents.tools.base import Tool
from nodetool.chat.providers import ChatProvider
from nodetool.metadata.types import Message, ToolCall
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk

logger = get_logger(__name__)

DEFAULT_MAX_ITERATIONS = 10

# Metadata schema for results
METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "description"],
    "additionalProperties": True,
}


def json_schema_for_output_type(output_type: str) -> Dict[str, Any]:
    """Generate JSON schema for different output types."""
    if output_type == "json":
        return {"type": "object", "description": "JSON object"}
    elif output_type == "list":
        return {"type": "array", "description": "Array of values"}
    elif output_type == "string":
        return {"type": "string", "description": "Text string"}
    elif output_type == "number":
        return {"type": "number", "description": "Numeric value"}
    elif output_type == "boolean":
        return {"type": "boolean", "description": "Boolean value"}
    elif output_type == "markdown":
        return {"type": "string", "description": "Markdown formatted text"}
    elif output_type == "html":
        return {"type": "string", "description": "HTML markup"}
    elif output_type == "csv":
        return {"type": "string", "description": "CSV formatted data"}
    elif output_type == "yaml":
        return {"type": "string", "description": "YAML formatted data"}
    else:
        return {"type": "string", "description": f"Output of type {output_type}"}


class FinishTool(Tool):
    """Tool for completing a task with a direct value result."""

    name: str = "finish_task"
    description: str = (
        "Finish the task by providing the final result value and metadata."
    )

    def __init__(self, output_type: str, output_schema: Any):
        super().__init__()
        self.output_type = output_type

        # Use the provided schema or generate one from output_type
        if output_schema:
            result_schema = output_schema
        else:
            result_schema = json_schema_for_output_type(output_type)

        self.input_schema = {
            "type": "object",
            "properties": {
                "result": result_schema,
                "metadata": METADATA_SCHEMA,
            },
            "required": ["result", "metadata"],
            "additionalProperties": False,
        }

    async def process(
        self, context: ProcessingContext, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        return params


class AgentExecutor:
    """
    Agent-based task executor that operates on Python values.

    This executor manages the execution of a single agent task using LLM interactions,
    working with direct Python values instead of files.
    """

    def __init__(
        self,
        objective: str,
        output_type: str,
        processing_context: ProcessingContext,
        provider: ChatProvider,
        model: str,
        tools: Sequence[Tool],
        input_values: Optional[Dict[str, Any]] = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        output_schema: Optional[Any] = None,
    ):
        self.objective = objective
        self.output_type = output_type
        self.processing_context = processing_context
        self.provider = provider
        self.model = model
        self.input_values = input_values or {}
        self.max_iterations = max_iterations
        self.output_schema = output_schema

        # Initialize finish tool
        self.finish_tool = FinishTool(self.output_type, output_schema)

        # Combine provided tools with finish tool
        self.tools = list(tools) + [self.finish_tool]

        # Create system prompt
        self.system_prompt = self._create_system_prompt()

        # Initialize message history
        self.history = [Message(role="system", content=self.system_prompt)]

        # Execution state
        self.iterations = 0
        self.completed = False
        self.result = None
        self.metadata = None

    def _create_system_prompt(self) -> str:
        """Create system prompt for the task."""
        return f"""
You are an AI agent executing a focused objective. Produce a result of type '{self.output_type}'.

Operating mode (persistence):
- Keep going until the objective is completed; do not hand back early.
- Resolve ambiguity by making reasonable assumptions and record them in `metadata.notes`.
- Prefer tool calls and concrete actions over clarifying questions.

Tool preambles:
- First assistant message: restate the objective in one sentence and list a 1–3 step plan.
- Before each tool call, add a one-sentence rationale describing what and why.
- After tool results, update the plan only if it materially changes.

Execution protocol:
1. Focus on the objective: {self.objective}
2. Use the provided input values efficiently (available keys: {len(self.input_values)})
3. Perform the minimal steps required to generate the result
4. Ensure the final result matches the expected output type: {self.output_type}
5. Call 'finish_task' exactly once at the end with final `result` and `metadata` (title, description, sources, notes)

Safety and privacy:
- Do not reveal chain-of-thought; output only tool calls and required fields.
- Prefer deterministic, structured outputs over prose.
"""

    async def execute(self) -> AsyncGenerator[Union[Chunk, ToolCall], None]:
        """Execute the task and yield progress updates."""

        logger.debug(f"Starting agent execution: {self.objective}")

        # Create initial prompt
        prompt_parts = [
            f"**Objective:**\n{self.objective}\n",
        ]

        if self.input_values:
            input_str = "\n".join(
                [
                    f"- {key}: {self._format_value(val)}"
                    for key, val in self.input_values.items()
                ]
            )
            prompt_parts.append(f"**Input Values:**\n{input_str}\n")

        prompt_parts.append(
            "Please complete the objective using the provided input values."
        )
        task_prompt = "\n".join(prompt_parts)

        # Add to history
        self.history.append(Message(role="user", content=task_prompt))

        # Main execution loop
        while not self.completed and self.iterations < self.max_iterations:
            self.iterations += 1

            logger.debug(f"Agent iteration {self.iterations}/{self.max_iterations}")

            # Get LLM response
            response = await self.provider.generate_message(
                messages=self.history,
                tools=self.tools,
                model=self.model,
            )

            # Handle response
            if response.content:
                yield Chunk(content=str(response.content))

            # Add assistant message to history (including tool calls if any)
            assistant_message = Message(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls if response.tool_calls else None,
            )
            self.history.append(assistant_message)

            if response.tool_calls:
                for tool_call in response.tool_calls:
                    logger.debug(f"Processing tool call: {tool_call.name}")

                    # Yield tool call
                    yield ToolCall(
                        id=tool_call.id,
                        name=tool_call.name,
                        args=tool_call.args,
                        message=f"Calling {tool_call.name}",
                    )

                    # Process tool call
                    result = await self._handle_tool_call(tool_call)

                    # Add tool result to history (safe serialization)
                    try:
                        serialized = (
                            "Tool returned no output."
                            if result is None
                            else json.dumps(result)
                        )
                    except TypeError as e:
                        logger.error(
                            f"Failed to serialize tool result for history: {e}. Result: {result}"
                        )
                        serialized = json.dumps(
                            {
                                "error": f"Failed to serialize tool result: {e}",
                                "result_repr": repr(result),
                            }
                        )

                    self.history.append(
                        Message(
                            role="tool",
                            content=serialized,
                            tool_call_id=tool_call.id,
                        )
                    )

                    # Check if task completed
                    if tool_call.name == "finish_task":
                        self.completed = True
                        self.result = tool_call.args.get("result")
                        self.metadata = tool_call.args.get("metadata")
                        break

        # Handle max iterations reached
        if not self.completed:
            await self._handle_max_iterations()

    async def _handle_tool_call(self, tool_call: ToolCall) -> Dict[str, Any]:
        """Handle execution of a tool call."""
        # Find the tool
        tool = None
        for t in self.tools:
            if t.name == tool_call.name:
                tool = t
                break

        if not tool:
            return {"error": f"Tool {tool_call.name} not found"}

        try:
            # Execute the tool
            result = await tool.process(self.processing_context, tool_call.args)
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_call.name}: {e}")
            return {"error": str(e)}

    async def _handle_max_iterations(self):
        """Handle case where max iterations is reached without completion."""
        logger.warning(f"Max iterations reached for objective: {self.objective}")

        # Force completion with a default result
        self.completed = True
        self.result = f"Task incomplete after {self.max_iterations} iterations"
        self.metadata = {
            "title": "Incomplete Task",
            "description": "Task did not complete within iteration limit",
            "sources": [],
        }

    def _format_value(self, value: Any) -> str:
        """Format a value for display in prompts."""
        if isinstance(value, str):
            return value[:200] + "..." if len(value) > 200 else value
        elif isinstance(value, (dict, list)):
            formatted = json.dumps(value, indent=2)
            return formatted[:200] + "..." if len(formatted) > 200 else formatted
        else:
            return str(value)

    def get_result(self) -> Any:
        """Get the final result of the subtask."""
        return self.result

    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """Get the metadata for the result."""
        return self.metadata


async def main():
    """Test scenario for the AgentExecutor."""
    import os
    from nodetool.chat.providers.openai_provider import OpenAIProvider
    from nodetool.workflows.processing_context import ProcessingContext

    # Test input data
    input_values = {
        "sales_data": [
            {"product": "Widget A", "quantity": 10, "price": 25.99},
            {"product": "Widget B", "quantity": 5, "price": 45.50},
            {"product": "Widget A", "quantity": 8, "price": 25.99},
            {"product": "Widget C", "quantity": 3, "price": 99.99},
            {"product": "Widget B", "quantity": 12, "price": 45.50},
        ],
        "date_range": "2024-01-01 to 2024-01-31",
    }

    # Initialize provider and context
    provider = OpenAIProvider()
    processing_context = ProcessingContext()

    # Import math tools for testing
    from nodetool.agents.tools.math_tools import CalculatorTool, StatisticsTool

    # Create executor with math tools
    executor = AgentExecutor(
        objective="Analyze the provided sales data and calculate total revenue, average order value, and identify top products using mathematical calculations",
        output_type="json",
        processing_context=processing_context,
        provider=provider,
        model="gpt-4o-mini",
        tools=[CalculatorTool(), StatisticsTool()],  # Include math tools
        input_values=input_values,
        max_iterations=5,
    )

    print("🚀 Starting AgentExecutor test...")
    print(f"Objective: {executor.objective}")
    print(f"Output type: {executor.output_type}")
    print(f"Input data keys: {list(input_values.keys())}")
    print("-" * 60)

    try:
        # Execute the task
        async for update in executor.execute():
            print(update)

        # Get results
        result = executor.get_result()
        metadata = executor.get_metadata()

        print("-" * 60)
        print("✅ Task completed!")
        print(f"📊 Result: {json.dumps(result, indent=2)}")
        print(f"📝 Metadata: {json.dumps(metadata, indent=2)}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
