from typing import Any, Dict
from nodetool.metadata.types import Message, Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.tools.base import Tool


def guess_provider(model: str) -> Provider:
    if (
        model.startswith("gpt")
        or model.startswith("o1")
        or model.startswith("o3")
        or model.startswith("o4")
    ):
        return Provider.OpenAI
    elif model.startswith("claude"):
        return Provider.Anthropic
    elif model.startswith("gemini"):
        return Provider.Gemini
    else:
        return Provider.Ollama


REASONING_SYSTEM_PROMPT = """
You are a helpful assistant that articulates a reasoning step.
"""


class ReasoningTool(Tool):
    """
    🧠 Reasoning Tool - Allows the agent to articulate a reasoning step.

    This tool doesn't perform an external action but serves to make the agent's
    thought process explicit in the conversation history. It takes a reasoning
    string as input and returns it.
    """

    name: str = "reasoning_step"
    description: str = "Articulate a step in the reasoning process or thought process."
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "objective": {
                "type": "string",
                "description": "The objective of the reasoning step.",
            },
            "model": {
                "type": "string",
                "description": "The model to use for the reasoning step.",
            },
        },
        "required": ["reasoning", "model"],
        "additionalProperties": False,
    }

    async def process(
        self, context: ProcessingContext, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Processes the reasoning step. Simply returns the provided reasoning.
        """
        from nodetool.chat.providers import get_provider

        model = params.get("model")
        objective = params.get("objective")
        if not model:
            raise ValueError("Model is required")
        provider = get_provider(guess_provider(model))

        response = await provider.generate_message(
            [
                Message(
                    role="system",
                    content=REASONING_SYSTEM_PROMPT,
                ),
                Message(
                    role="user",
                    content=params.get("reasoning"),
                ),
            ],
            model,
        )

        return {"objective": objective, "reasoning": response.content, "model": model}

    def user_message(self, params: Dict[str, Any]) -> str | None:
        """Returns a user-friendly message describing the reasoning step."""
        reasoning = params.get("reasoning", "...")
        return f"Thinking: {reasoning}"
