#!/usr/bin/env python3
"""
Simple Learning Path Generator using Nodetool Agent system.
"""

import asyncio
import json
from typing import List
from pydantic import BaseModel

from nodetool.agents.tools import GoogleSearchTool
from nodetool.agents.agent import Agent
from nodetool.providers import get_provider
from nodetool.providers.base import BaseProvider
from nodetool.metadata.types import Provider
from nodetool.ui.console import AgentConsole
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk

import dotenv

dotenv.load_dotenv()


class LearningModule(BaseModel):
    title: str
    description: str
    resources: List[str]


class LearningPath(BaseModel):
    topic: str
    overview: str
    modules: List[LearningModule]


async def generate_learning_path(provider: BaseProvider, model: str, topic: str):
    context = ProcessingContext()

    agent = Agent(
        name="Learning Path Generator",
        objective=f"Create a simple learning path for '{topic}'. Include 3-5 modules with titles, descriptions, and useful web resources.",
        provider=provider,
        model=model,
        tools=[GoogleSearchTool()],
        display_manager=AgentConsole(),
        output_schema=LearningPath.model_json_schema()
    )

    print(f"Generating learning path for: {topic}")

    async for item in agent.execute(context):
        pass

    print("\n" + "=" * 40)
    if agent.results:
        print("Generated Learning Path:")
        path = LearningPath.model_validate(agent.results)
        for module in path.modules:
            print(f"Module: {module.title}")
            print(f"Description: {module.description}")
            print(f"Resources: {module.resources}")
    else:
        print("No results returned.")


if __name__ == "__main__":
    topic = "Getting Started with Docker"

    asyncio.run(
        generate_learning_path(
            provider=get_provider(Provider.HuggingFaceCerebras),
            model="openai/gpt-oss-120b",
            topic=topic,
        )
    )
