#!/usr/bin/env python3
"""
Reddit Scraping Agent with Task Planner: Indie Hacker Journey Deconstructor

This script creates a Reddit agent that:
1. Uses TaskPlanner to generate a dynamic plan for scraping Reddit for indie hacker stories.
2. Uses GoogleSearchTool to find Reddit posts detailing product journeys, launches, successes, and failures.
3. Uses BrowserTool to visit each URL and extract the content (post and key comments).
4. Analyzes these narratives to extract actionable insights, tools used, strategies, and key learnings.
5. Organizes and saves the results as a structured report.
"""

import asyncio
import json  # For potentially more structured output if desired later

from nodetool.agents.agent import Agent
from nodetool.chat.providers import get_provider
from nodetool.agents.tools import BrowserTool, GoogleSearchTool
from nodetool.chat.providers.base import ChatProvider
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk


async def test_reddit_journey_deconstructor_agent(  # Renamed for clarity
    provider: ChatProvider,
    model: str,
    reasoning_model: str,
    planning_model: str,
):
    context = ProcessingContext()
    search_agent = Agent(
        name="Reddit Indie Hacker Journey Deconstructor",  # New Agent Name
        objective="""
        You are an expert research analyst focused on the indie hacker and startup ecosystem on Reddit.
        Your mission is to:
        1. Use Google Search to discover posts on subreddits like r/indiehackers, r/SideProject, r/startups, r/EntrepreneurRideAlong, and r/SaaS where users share their:
            a. Product development journeys from idea to launch.
            b. Stories of acquiring their first users or achieving growth milestones.
            c. Detailed accounts of successes, including strategies and tools used.
            d. Candid reflections on failures and lessons learned.
            e. Revenue milestones (e.g., first $1, $100 MRR, $1k MRR, etc.).
        2. For each relevant post identified:
            a. Browse the post and its most insightful comments to gather comprehensive details.
            b. Extract the core narrative, key decisions, strategies implemented, tools/technologies mentioned, challenges faced, and outcomes (both positive and negative).
        3. Analyze the gathered information to deconstruct each journey by identifying:
            a. The type of product or service.
            b. Key timeline markers or phases (e.g., idea validation, MVP development, launch, scaling).
            c. Specific marketing or growth tactics employed.
            d. Pivotal moments or decisions.
            e. Quantifiable results if shared (e.g., revenue, user numbers, time to X).
            f. Explicitly stated lessons learned or advice offered.
            g. Tools, software, or platforms mentioned that were critical to their journey.
        4. Generate a structured Markdown report. Each journey deconstruction should be a distinct section, containing:
            - Title: (e.g., "Deconstruction: [Original Post Title/Product Name]")
            - Source URL: [Link to Reddit Post]
            - Product Niche: [e.g., SaaS, Mobile App, E-commerce, AI Tool]
            - Journey Summary: [Brief overview of the story]
            - Key Strategies/Tactics: [Bulleted list]
            - Tools Mentioned: [Bulleted list]
            - Key Learnings/Takeaways: [Bulleted list of insights from the post]
            - Reported Outcomes/Metrics: [e.g., Reached $1K MRR in 6 months, Acquired 500 users post-launch, Failed to find product-market fit]
        """,
        provider=provider,
        model=model,
        reasoning_model=reasoning_model,
        planning_model=planning_model,
        enable_analysis_phase=True,
        enable_data_contracts_phase=True,  # Ensures the agent tries to stick to the output structure
        tools=[
            GoogleSearchTool(),
            BrowserTool(),
        ],
        output_type="markdown",  # Markdown is good for readable reports
    )

    # Execute each task in the plan
    print(f"Starting agent: {search_agent.name}\nObjective: {search_agent.objective}\n")
    async for item in search_agent.execute(context):
        if isinstance(item, Chunk):
            # Assuming item.content is part of the Markdown report
            print(item.content, end="", flush=True)

    final_report = search_agent.get_results()
    if final_report:
        print("\n\n--- FINAL COMPILED REPORT ---")
        print(final_report)

    print(f"\n\n--- Workspace Directory for artifacts ---")
    print(context.workspace_dir)
    # You would typically find the full report or structured data in the workspace directory.


if __name__ == "__main__":
    # Ensure you have your OpenAI API key set in your environment variables
    # or configure the provider appropriately.
    asyncio.run(
        test_reddit_journey_deconstructor_agent(
            provider=get_provider(
                Provider.OpenAI
            ),  # Or Provider.Gemini, Provider.Anthropic
            model="gpt-4o-mini",
            planning_model="o4-mini",
            reasoning_model="o4-mini",
        )
    )
