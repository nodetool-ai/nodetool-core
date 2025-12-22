"""
Example: Using OpenRouter Provider in NodeTool

This example demonstrates how to use the OpenRouter provider to access
multiple AI models through a unified API.

OpenRouter provides access to models from:
- OpenAI (GPT-4, GPT-3.5, etc.)
- Anthropic (Claude 3, Claude 2, etc.)
- Google (Gemini Pro, etc.)
- Meta (Llama 3, etc.)
- Mistral (Mistral, Mixtral, etc.)
- And many more...

Requirements:
- Set OPENROUTER_API_KEY environment variable or in .env file
- Get an API key from https://openrouter.ai/
"""

import asyncio
from nodetool.metadata.types import Message, Provider
from nodetool.providers import get_provider


async def example_basic_chat():
    """Example: Basic chat completion with OpenRouter"""
    print("Example 1: Basic Chat Completion")
    print("-" * 50)
    
    # Get the OpenRouter provider
    provider = await get_provider(Provider.OpenRouter, user_id="1")
    
    # Create a simple message
    messages = [
        Message(role="user", content="What is OpenRouter?")
    ]
    
    # Generate a response (non-streaming)
    response = await provider.generate_message(
        messages=messages,
        model="openai/gpt-3.5-turbo",  # Model format: provider/model-name
        max_tokens=150
    )
    
    print(f"Response: {response.content}\n")


async def example_streaming_chat():
    """Example: Streaming chat completion with OpenRouter"""
    print("Example 2: Streaming Chat Completion")
    print("-" * 50)
    
    provider = await get_provider(Provider.OpenRouter, user_id="1")
    
    messages = [
        Message(role="user", content="Count from 1 to 5.")
    ]
    
    # Generate a streaming response
    print("Response: ", end="", flush=True)
    async for chunk in provider.generate_messages(
        messages=messages,
        model="anthropic/claude-3-haiku",
        max_tokens=100
    ):
        if hasattr(chunk, 'content'):
            print(chunk.content, end="", flush=True)
    
    print("\n")


async def example_different_models():
    """Example: Using different models through OpenRouter"""
    print("Example 3: Accessing Different Models")
    print("-" * 50)
    
    provider = await get_provider(Provider.OpenRouter, user_id="1")
    
    # List of models to try
    models = [
        "openai/gpt-3.5-turbo",
        "anthropic/claude-3-haiku",
        "google/gemini-pro",
        "meta-llama/llama-3-8b",
    ]
    
    messages = [
        Message(role="user", content="Say hello in one word.")
    ]
    
    for model in models:
        try:
            response = await provider.generate_message(
                messages=messages,
                model=model,
                max_tokens=10
            )
            print(f"{model}: {response.content}")
        except Exception as e:
            print(f"{model}: Error - {e}")
    
    print()


async def example_tool_calling():
    """Example: Function/tool calling with OpenRouter"""
    print("Example 4: Tool Calling")
    print("-" * 50)
    
    provider = await get_provider(Provider.OpenRouter, user_id="1")
    
    # Define a simple tool
    from nodetool.agents.tools.base import Tool
    
    get_weather_tool = Tool(
        name="get_weather",
        description="Get the current weather for a location",
        input_schema={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                }
            },
            "required": ["location"]
        }
    )
    
    messages = [
        Message(role="user", content="What's the weather in London?")
    ]
    
    # Note: Only some models support tool calling
    # OpenAI GPT-4 and Anthropic Claude models support it
    response = await provider.generate_message(
        messages=messages,
        model="openai/gpt-4",
        tools=[get_weather_tool],
        max_tokens=100
    )
    
    if response.tool_calls:
        print("Tool called:")
        for tool_call in response.tool_calls:
            print(f"  - {tool_call.name}: {tool_call.args}")
    else:
        print(f"Response: {response.content}")
    
    print()


async def example_list_models():
    """Example: List available models from OpenRouter"""
    print("Example 5: List Available Models")
    print("-" * 50)
    
    provider = await get_provider(Provider.OpenRouter, user_id="1")
    
    # Get available models
    models = await provider.get_available_language_models()
    
    print(f"Available models: {len(models)}")
    print("\nFirst 10 models:")
    for model in models[:10]:
        print(f"  - {model.id}: {model.name}")
    
    print()


async def main():
    """Run all examples"""
    print("=" * 50)
    print("OpenRouter Provider Examples")
    print("=" * 50)
    print()
    
    # Note: These examples require a valid OPENROUTER_API_KEY
    # Uncomment the examples you want to run:
    
    # await example_basic_chat()
    # await example_streaming_chat()
    # await example_different_models()
    # await example_tool_calling()
    # await example_list_models()
    
    print("Note: Uncomment the examples you want to run in the main() function")
    print("Make sure OPENROUTER_API_KEY is set in your environment")


if __name__ == "__main__":
    asyncio.run(main())
