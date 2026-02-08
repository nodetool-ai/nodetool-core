"""
Example: Using MiniMax Provider in NodeTool

This example demonstrates how to use the MiniMax provider to generate
LLM responses using MiniMax's Anthropic-compatible API, images
using MiniMax's image generation API, and videos using MiniMax's
Hailuo video generation API.

MiniMax provides access to models like:
- MiniMax-M2.1
- MiniMax-M2.1-lightning
- MiniMax-M2
- MiniMax-Text-01
- image-01 (for image generation)
- image-01-live (for image generation with style options)
- MiniMax-Hailuo-2.3 (for video generation)
- MiniMax-Hailuo-2.3-Fast (for fast video generation)
- MiniMax-Hailuo-02 (for video generation)

Requirements:
- Set MINIMAX_API_KEY environment variable
- Get an API key from https://platform.minimaxi.com/

For CI/CD usage, configure MINIMAX_API_KEY as a GitHub secret and pass
it to the workflow environment.
"""

import asyncio

from nodetool.metadata.types import ImageModel, Message, Provider, VideoModel
from nodetool.providers import get_provider
from nodetool.providers.types import TextToImageParams, TextToVideoParams


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

    # Get available language models
    language_models = await provider.get_available_language_models()

    print(f"Available MiniMax language models: {len(language_models)}")
    for model in language_models:
        print(f"  - {model.id}: {model.name}")

    # Get available image models
    image_models = await provider.get_available_image_models()

    print(f"\nAvailable MiniMax image models: {len(image_models)}")
    for model in image_models:
        print(f"  - {model.id}: {model.name}")

    # Get available video models
    video_models = await provider.get_available_video_models()

    print(f"\nAvailable MiniMax video models: {len(video_models)}")
    for model in video_models:
        print(f"  - {model.id}: {model.name}")

    # Get available TTS models
    tts_models = await provider.get_available_tts_models()

    print(f"\nAvailable MiniMax TTS models: {len(tts_models)}")
    for model in tts_models:
        print(f"  - {model.id}: {model.name}")

    print()


async def example_image_generation():
    """Example: Generate an image with MiniMax"""
    print("MiniMax Provider - Image Generation Example")
    print("-" * 50)

    provider = await get_provider(Provider.MiniMax, user_id="1")

    # Create image generation parameters
    params = TextToImageParams(
        model=ImageModel(
            id="image-01",
            name="MiniMax Image-01",
            provider=Provider.MiniMax,
        ),
        prompt="A serene landscape with mountains, a lake, and a sunset",
        width=1024,
        height=1024,
    )

    print(f"Generating image with prompt: '{params.prompt}'")

    # Generate the image
    image_bytes = await provider.text_to_image(params)

    print(f"Generated image size: {len(image_bytes)} bytes")

    # Optionally save to file
    import aiofiles

    output_path = "/tmp/minimax_generated_image.png"
    async with aiofiles.open(output_path, "wb") as f:
        await f.write(image_bytes)
    print(f"Image saved to: {output_path}\n")


async def example_video_generation():
    """Example: Generate a video with MiniMax Hailuo"""
    print("MiniMax Provider - Video Generation Example")
    print("-" * 50)

    provider = await get_provider(Provider.MiniMax, user_id="1")

    # Create video generation parameters
    params = TextToVideoParams(
        model=VideoModel(
            id="MiniMax-Hailuo-2.3",
            name="Hailuo 2.3",
            provider=Provider.MiniMax,
        ),
        prompt="A cat wearing sunglasses, walking down a street at sunset",
    )

    print(f"Generating video with prompt: '{params.prompt}'")

    # Generate the video (this may take a few minutes)
    video_bytes = await provider.text_to_video(params, timeout_s=600)

    print(f"Generated video size: {len(video_bytes)} bytes")

    # Optionally save to file
    import aiofiles

    output_path = "/tmp/minimax_generated_video.mp4"
    async with aiofiles.open(output_path, "wb") as f:
        await f.write(video_bytes)
    print(f"Video saved to: {output_path}\n")


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

        # Run image generation example
        await example_image_generation()

        # Run video generation example
        await example_video_generation()
    except Exception as e:
        print(f"Error: {e}")
        print("\n(i) Make sure MINIMAX_API_KEY is set in your environment")
        print("   Get your API key from: https://platform.minimaxi.com/")


if __name__ == "__main__":
    asyncio.run(main())
