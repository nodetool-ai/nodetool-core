#!/usr/bin/env python3
"""
OpenAI Image Generation Agent Example

This script demonstrates how to use an Agent with the OpenAIImageGenerationTool
to generate an image based on a textual prompt. It will then save the
generated image to the workspace directory.
"""

import asyncio
import os

from nodetool.agents.agent import Agent
from nodetool.agents.tools.workspace_tools import WriteFileTool
from nodetool.chat.providers import get_provider
from nodetool.agents.tools import OpenAIImageGenerationTool
from nodetool.chat.providers.base import ChatProvider
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk


async def test_openai_image_generation_agent(
    provider: ChatProvider,
    model: str,
    image_prompt: str,
):
    """
    Initializes and runs the OpenAI Image Generation agent.

    Args:
        provider: The chat provider (e.g., OpenAI).
        model: The primary model for agent orchestration.
        image_prompt: The textual prompt to generate an image from.
    """
    context = ProcessingContext()
    print(f"Workspace for this run: {context.workspace_dir}")
    if not os.path.exists(context.workspace_dir):
        os.makedirs(context.workspace_dir)
        print(f"Created workspace directory: {context.workspace_dir}")

    image_agent = Agent(
        name="OpenAIImageGenerationAgent",
        objective=f"""
        You are an image generation assistant. Your primary goal is to generate an image
        based on the following prompt: '{image_prompt}'.
        Write the image to the workspace directory.
        """,
        provider=provider,
        model=model,
        tools=[
            OpenAIImageGenerationTool(),
            WriteFileTool(),
        ],
        output_schema={
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "The path to the generated image",
                },
            },
        },
    )

    print(f"Starting agent: {image_agent.name}")
    print(f"Task: Generate an image for prompt: '{image_prompt}'")

    async for item in image_agent.execute(processing_context=context):
        if isinstance(item, Chunk):
            print(f"Agent Stream Output: {item.content}", end="", flush=True)

    print("\n\n--- Agent execution finished ---")

    results = image_agent.get_results()

    print(f"Results: {results}")


if __name__ == "__main__":
    IMAGE_PROMPT = (
        "A photorealistic image of a red panda coding on a laptop in a forest"
    )

    asyncio.run(
        test_openai_image_generation_agent(
            provider=get_provider(Provider.OpenAI),
            model="gpt-4o-mini",
            image_prompt=IMAGE_PROMPT,
        )
    )
