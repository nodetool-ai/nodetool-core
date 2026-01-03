"""
Example: Using MiniMax Provider in NodeTool

This example demonstrates how to use the MiniMax provider to generate
LLM responses using MiniMax's Anthropic-compatible API.

MiniMax provides access to models like:
- MiniMax-M2.1
- MiniMax-M2.1-lightning
- MiniMax-M2

Requirements:
- Set MINIMAX_API_KEY environment variable
- Get an API key from https://platform.minimaxi.com/

For CI/CD usage, configure MINIMAX_API_KEY as a GitHub secret and pass
it to the workflow environment.
"""

import asyncio

from nodetool.metadata.types import Message, Provider
from nodetool.providers import get_provider


async def example_basic_chat():
    """Example: Basic chat completion with MiniMax"""
    print("MiniMax Provider - Basic Chat Example")
    print("-" * 50)

    # Get the MiniMax provider
    provider = await get_provider(Provider.MiniMax, user_id="1")

    # Create a simple message
    messages = [Message(role="user", content="What is MiniMax AI?")]

    # Generate a response (non-streaming)
    response = await provider.generate_message(
        messages=messages,
        model="MiniMax-M2.1",
        max_tokens=150,
    )

    print(f"Response: {response.content}\n")


async def example_streaming_chat():
    """Example: Streaming chat completion with MiniMax"""
    print("MiniMax Provider - Streaming Chat Example")
    print("-" * 50)

    provider = await get_provider(Provider.MiniMax, user_id="1")

    messages = [Message(role="user", content="Count from 1 to 5.")]

    # Generate a streaming response
    print("Response: ", end="", flush=True)
    async for chunk in provider.generate_messages(messages=messages, model="MiniMax-M2.1-lightning", max_tokens=100):
        if hasattr(chunk, "content"):
            print(chunk.content, end="", flush=True)

    print("\n")


async def example_list_models():
    """Example: List available models from MiniMax"""
    print("MiniMax Provider - List Available Models")
    print("-" * 50)

    provider = await get_provider(Provider.MiniMax, user_id="1")

    # Get available models
    models = await provider.get_available_language_models()

    print(f"Available MiniMax models: {len(models)}")
    for model in models:
        print(f"  - {model.id}: {model.name}")

    print()


async def main():
    """Run all examples"""
    print("=" * 50)
    print("MiniMax Provider Examples")
    print("=" * 50)
    print()

    try:
        # List available models
        await example_list_models()

        # Run basic chat example
        await example_basic_chat()

        # Run streaming chat example
        await example_streaming_chat()
    except Exception as e:
        print(f"Error: {e}")
        print("\n(i) Make sure MINIMAX_API_KEY is set in your environment")
        print("   Get your API key from: https://platform.minimaxi.com/")


if __name__ == "__main__":
    asyncio.run(main())
