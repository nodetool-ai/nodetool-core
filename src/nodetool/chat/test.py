# Test function to demonstrate provider usage
import json
from nodetool.chat.providers.base import Chunk
from nodetool.chat.providers.ollama import OllamaProvider
from nodetool.chat.tools.base import Tool
from nodetool.metadata.types import ToolCall
from nodetool.workflows.processing_context import ProcessingContext


class Test2Tool(Tool):
    name = "test_2"
    description = "A test tool for integration testing"
    input_schema = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Test message to echo back",
            },
        },
        "required": ["message"],
    }

    async def process(self, context: ProcessingContext, params: dict):
        return {
            "echo": params["message"],
        }


async def test_ollama_provider():
    """
    Test function to demonstrate the OllamaProvider functionality.

    This function shows basic usage patterns for both streaming and non-streaming
    generation with and without tools.
    """
    from nodetool.metadata.types import Message
    from nodetool.chat.tools.system import TestTool

    provider = OllamaProvider()

    # Create sample messages
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Call both tools"),
    ]

    try:
        response = await provider.generate_message(
            messages=messages,
            model="llama3.2:1b",
            tools=[TestTool("/tmp"), Test2Tool("/tmp")],
        )
        print(f"\nResponse: {response}")
        print(f"Usage stats: {provider.usage}")
    except Exception as e:
        print(f"Error in non-streaming test: {e}")


# Uncomment to run the test (requires running in an async context)
if __name__ == "__main__":
    import asyncio

    asyncio.run(test_ollama_provider())
