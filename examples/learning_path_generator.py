#!/usr/bin/env python3
"""
Learning Path Generator using the Nodetool Agent system.

This script demonstrates using an Agent instance configured with GoogleSearchTool
to generate a structured learning path for a given topic.

This example shows how to:
1. Set up a single agent with GoogleSearchTool and an objective to create a learning path.
2. Define an output schema for the desired learning path structure.
3. Execute the agent and process its streaming output.
4. Retrieve a structured learning path based on the topic, using web search for resources.
"""

import asyncio
import json
import os  # Import os for environment variables if needed directly

from nodetool.agents.tools import (
    GoogleSearchTool,
    GoogleNewsTool,
    BrowserTool,
)

from nodetool.agents.agent import Agent
from nodetool.chat.providers import get_provider
from nodetool.chat.providers.base import ChatProvider
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk

import dotenv

# Load environment variables from .env file
# Ensure any necessary API keys (LLM Provider, potentially Search API keys if required by GoogleSearchTool) are present
dotenv.load_dotenv()


async def generate_learning_path(provider: ChatProvider, model: str, topic: str):
    """
    Generates a learning path for a given topic using an Agent with GoogleSearchTool.

    Args:
        provider: The chat provider instance.
        model: The model name to use.
        topic: The topic for which to generate the learning path.
    """
    context = ProcessingContext()

    # Prepare the list of tools for the agent
    tools_list = [
        GoogleSearchTool(),
        GoogleNewsTool(),
        BrowserTool(),
    ]

    # Define the output schema for the learning path
    learning_path_schema = {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "The learning topic."},
            "learning_path": {
                "type": "array",
                "description": "An array of modules or steps in the learning path.",
                "items": {
                    "type": "object",
                    "properties": {
                        "module_number": {
                            "type": "integer",
                            "description": "Sequential number of the module/step.",
                        },
                        "title": {
                            "type": "string",
                            "description": "Concise title for the learning module/step.",
                        },
                        "description": {
                            "type": "string",
                            "description": "Detailed description of what to learn in this module.",
                        },
                        "resources": {
                            "type": "array",
                            "description": "List of suggested learning resources (articles, videos, tutorials) found via web search.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "resource_title": {"type": "string"},
                                    "link": {"type": "string", "format": "uri"},
                                    "type": {
                                        "type": "string",
                                        "enum": [
                                            "article",
                                            "video",
                                            "tutorial",
                                            "book",
                                            "course",
                                            "documentation",
                                            "other",
                                        ],
                                    },
                                },
                                "required": ["resource_title", "link", "type"],
                            },
                            # Allow steps with no specific resources found
                            "minItems": 0,
                        },
                    },
                    "required": ["module_number", "title", "description", "resources"],
                },
            },
        },
        "required": ["topic", "learning_path"],
    }

    # Configure the agent
    agent = Agent(
        name="Learning Path Generator Agent",
        enable_analysis_phase=True,  # Analysis helps break down the topic
        enable_data_contracts_phase=True,  # Enforce the structured output
        objective=f"""
        Generate a structured, step-by-step learning path for the topic: '{topic}'.
        1. Break the topic down into logical modules or learning steps, ordered sequentially.
        2. For each module, provide a clear title and a detailed description of the concepts to cover.
        3. For each module, use the GoogleSearchTool to find 1-3 relevant and high-quality online learning resources (like articles, tutorials, videos, documentation). Include the resource title, a valid URL link, and the type of resource (article, video, etc.). Prioritize official documentation or well-regarded educational platforms if possible.
        4. Ensure the final output strictly adheres to the provided JSON schema.
        """,
        provider=provider,
        model=model,
        tools=tools_list,  # Pass the GoogleSearchTool instance
        output_schema=learning_path_schema,  # Provide the desired output structure
    )

    print(f"Executing Learning Path Generator Agent with model: {model}")
    print(f"Objective: Generate learning path for '{topic}'")
    print("-" * 30)

    async for item in agent.execute(context):
        if isinstance(item, Chunk):
            # Print the agent's thought process or intermediate steps
            print(item.content, end="", flush=True)

    print("\n" + "-" * 30)
    print(f"=== Agent Execution Complete ===")
    print(f"Workspace: {context.workspace_dir}")

    # Display the final structured results
    if agent.results:
        print("=== Generated Learning Path ===")
        # Check if results conform to the expected structure
        if isinstance(agent.results, dict) and "learning_path" in agent.results:
            print(json.dumps(agent.results, indent=2))
            path_steps = agent.results.get("learning_path", [])
            print(
                f"\nSuccessfully generated a learning path with {len(path_steps)} modules."
            )
        else:
            print("Results structure might not match the expected schema:")
            print(json.dumps(agent.results, indent=2))  # Print what was received
    else:
        print("No structured results returned by the agent.")


if __name__ == "__main__":
    # Define the topic you want to learn about
    topic_to_learn = "Getting Started with Docker"
    # topic_to_learn = "Learning Python for Data Analysis"
    # topic_to_learn = "Understanding Transformer Models in NLP"

    # --- Choose Provider and Model ---
    # Make sure the necessary API keys (e.g., OPENAI_API_KEY) are in your .env file
    # Note: GoogleSearchTool might require additional API keys (e.g., SERP API or Google Cloud API keys)
    # depending on its implementation within nodetool. Check nodetool documentation.

    # --- Example: Run with OpenAI GPT-4o Mini ---
    print("\n>>> Running with OpenAI GPT-4o Mini...")
    try:
        asyncio.run(
            generate_learning_path(
                provider=get_provider(Provider.OpenAI),
                model="gpt-4o-mini",  # Or "gpt-4o", "gpt-3.5-turbo" etc.
                topic=topic_to_learn,
            )
        )
    except Exception as e:
        print(f"Error running with OpenAI: {e}")

    # --- Example: Run with Google Gemini ---
    # Make sure GEMINI_API_KEY is in your .env file
    # print("\n>>> Running with Google Gemini Flash...")
    # try:
    #     asyncio.run(
    #         generate_learning_path(
    #             provider=get_provider(Provider.Gemini),
    #             model="gemini-1.5-flash-latest", # Or other Gemini models
    #             topic=topic_to_learn
    #         )
    #     )
    # except Exception as e:
    #      print(f"Error running with Gemini: {e}")

    # --- Example: Run with Anthropic Claude ---
    # Make sure ANTHROPIC_API_KEY is in your .env file
    # print("\n>>> Running with Anthropic Claude Sonnet...")
    # try:
    #     asyncio.run(
    #         generate_learning_path(
    #             provider=get_provider(Provider.Anthropic),
    #             # Use the latest model available or a specific one
    #             model="claude-3-5-sonnet-20240620",
    #             topic=topic_to_learn
    #         )
    #     )
    # except Exception as e:
    #      print(f"Error running with Anthropic: {e}")

    # --- Example: Run with Ollama (if running locally) ---
    # print("\n>>> Running with Ollama (e.g., llama3)...")
    # try:
    #     # Ensure Ollama server is running locally
    #     asyncio.run(
    #         generate_learning_path(
    #             provider=get_provider(Provider.Ollama),
    #             model="llama3", # Replace with your desired Ollama model
    #             topic=topic_to_learn
    #         )
    #     )
    # except Exception as e:
    #      print(f"Error running with Ollama: {e}")
