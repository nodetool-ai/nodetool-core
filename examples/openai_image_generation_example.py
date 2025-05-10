#!/usr/bin/env python3
"""
OpenAI Image Generation Agent Example

This script demonstrates how to use an Agent with the OpenAIImageGenerationTool
to generate an image based on a textual prompt. It will then save the
generated image to the workspace directory.
"""

import asyncio
import base64
import os
import json  # For pretty printing the results
import binascii  # For catching b64decode errors

from nodetool.agents.agent import Agent
from nodetool.chat.providers import get_provider
from nodetool.agents.tools import OpenAIImageGenerationTool
from nodetool.chat.providers.base import ChatProvider
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk


async def test_openai_image_generation_agent(
    provider: ChatProvider,
    model: str,
    reasoning_model: str,
    planning_model: str,
    image_prompt: str,
):
    """
    Initializes and runs the OpenAI Image Generation agent.

    Args:
        provider: The chat provider (e.g., OpenAI).
        model: The primary model for agent orchestration.
        reasoning_model: The model used for reasoning tasks by the agent.
        planning_model: The model used for planning tasks by the agent.
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
        """,
        provider=provider,
        model=model,
        reasoning_model=reasoning_model,
        planning_model=planning_model,
        tools=[
            OpenAIImageGenerationTool(),
        ],
        output_type="png",
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
            planning_model="gpt-4o-mini",
            reasoning_model="gpt-4o-mini",
            image_prompt=IMAGE_PROMPT,
        )
    )
