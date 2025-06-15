#!/usr/bin/env python3
"""
Simple Learning Path Generator using Nodetool Agent system.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import List

from nodetool.agents.tools import GoogleSearchTool
from nodetool.agents.agent import Agent
from nodetool.chat.providers import get_provider
from nodetool.chat.providers.base import ChatProvider
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk

import dotenv

dotenv.load_dotenv()


@dataclass
class LearningModule:
    title: str
    description: str
    resources: List[str]


@dataclass
class LearningPath:
    topic: str
    overview: str
    modules: List[LearningModule]


async def generate_learning_path(provider: ChatProvider, model: str, topic: str):
    context = ProcessingContext()

    agent = Agent(
        name="Learning Path Generator",
        objective=f"Create a simple learning path for '{topic}'. Include 3-5 modules with titles, descriptions, and useful web resources.",
        provider=provider,
        model=model,
        tools=[GoogleSearchTool()],
    )

    print(f"Generating learning path for: {topic}")

    async for item in agent.execute(context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    print("\n" + "=" * 40)
    if agent.results:
        print("Generated Learning Path:")
        print(json.dumps(agent.results, indent=2))
    else:
        print("No results returned.")


if __name__ == "__main__":
    topic = "Getting Started with Docker"

    asyncio.run(
        generate_learning_path(
            provider=get_provider(Provider.OpenAI),
            model="gpt-4.1-mini",
            topic=topic,
        )
    )
