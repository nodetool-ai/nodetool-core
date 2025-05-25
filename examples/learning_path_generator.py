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
            "overview": {
                "type": "string",
                "description": "A brief overview (2-3 sentences) of what the learner will achieve after completing this learning path.",
            },
            "estimated_total_time": {
                "type": "string",
                "description": "Total estimated time to complete the entire learning path (e.g., '4-6 weeks', '40-60 hours')",
            },
            "prerequisites": {
                "type": "array",
                "description": "List of prerequisites or recommended background knowledge",
                "items": {"type": "string"},
                "minItems": 0,
            },
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
                            "description": "Clear, descriptive title for the learning module.",
                        },
                        "description": {
                            "type": "string",
                            "description": "Comprehensive description (100-200 words) of what to learn in this module.",
                        },
                        "learning_objectives": {
                            "type": "array",
                            "description": "Specific learning objectives for this module",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 5,
                        },
                        "resources": {
                            "type": "array",
                            "description": "List of suggested learning resources (articles, videos, tutorials) found via web search.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "resource_title": {
                                        "type": "string",
                                        "description": "Clear, descriptive title of the resource",
                                    },
                                    "link": {
                                        "type": "string",
                                        "format": "uri",
                                        "description": "Direct URL to the resource",
                                    },
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
                                        "description": "Type of learning resource",
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "Brief description (1-2 sentences) of what this resource covers and why it's valuable for this module",
                                    },
                                    "difficulty": {
                                        "type": "string",
                                        "enum": [
                                            "beginner",
                                            "intermediate",
                                            "advanced",
                                        ],
                                        "description": "Difficulty level of the resource",
                                    },
                                    "estimated_time": {
                                        "type": "string",
                                        "description": "Estimated time to complete (e.g., '30 minutes', '2 hours', '1 week')",
                                    },
                                },
                                "required": [
                                    "resource_title",
                                    "link",
                                    "type",
                                    "description",
                                ],
                            },
                            "minItems": 2,
                            "maxItems": 4,
                        },
                        "practice_exercises": {
                            "type": "array",
                            "description": "Suggested practice exercises or mini-projects for this module",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 3,
                        },
                    },
                    "required": [
                        "module_number",
                        "title",
                        "description",
                        "learning_objectives",
                        "resources",
                        "practice_exercises",
                    ],
                },
                "minItems": 5,
                "maxItems": 8,
            },
        },
        "required": [
            "topic",
            "overview",
            "estimated_total_time",
            "prerequisites",
            "learning_path",
        ],
    }

    # Configure the agent
    agent = Agent(
        name="Learning Path Generator Agent",
        enable_analysis_phase=True,  # Analysis helps break down the topic
        enable_data_contracts_phase=True,  # Enforce the structured output
        objective=f"""
        Create a comprehensive, structured learning path for mastering '{topic}'. 

        Your task is to design a progressive curriculum that takes a learner from beginner to proficient level. Follow these guidelines:

        1. **Topic Analysis**: First, thoroughly research '{topic}' using GoogleSearchTool to understand:
           - Core concepts and prerequisites
           - Common learning progressions
           - Industry best practices and standards
           - Typical challenges learners face

        2. **Learning Path Structure**: Design 5-8 sequential modules that:
           - Start with foundational concepts (no prior knowledge assumed)
           - Build progressively in complexity
           - Include practical, hands-on components
           - Cover both theoretical understanding and practical application
           - End with advanced topics or real-world project ideas

        3. **Module Design**: For each module, provide:
           - A clear, descriptive title (not just "Module 1")
           - A comprehensive description (100-200 words) explaining:
             * What concepts will be covered
             * Why this module is important
             * What the learner will be able to do after completing it
             * Any prerequisites from previous modules

        4. **Resource Curation**: For each module, use GoogleSearchTool and BrowserTool to find 2-4 high-quality resources:
           - Prioritize: official documentation, reputable educational platforms (Coursera, edX, freeCodeCamp), well-known tech blogs, and video tutorials
           - Verify links are accessible and content matches the module's objectives
           - Include a mix of resource types (articles for theory, videos for visual learning, tutorials for hands-on practice)
           - Focus on free or freely accessible resources when possible

        5. **Quality Standards**:
           - Ensure logical flow between modules
           - Balance theory with practical application
           - Include both quick wins (early modules) and challenging content (later modules)
           - Consider different learning styles (visual, textual, hands-on)

        6. **Output Requirements**: Structure your response according to the provided JSON schema, ensuring all required fields are populated with meaningful, detailed content.

        Remember: You're creating a learning roadmap that someone could follow independently to gain real competency in '{topic}'.
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
    print("=== Agent Execution Complete ===")
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
