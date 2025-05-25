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

from nodetool.agents.agent import Agent
from nodetool.chat.providers import get_provider
from nodetool.agents.tools import BrowserTool, GoogleSearchTool
from nodetool.chat.providers.base import ChatProvider
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk

objective = """
Find and analyze indie hacker success stories from Reddit to extract actionable insights.

Tasks:
1. Search Reddit (r/indiehackers, r/SideProject, r/startups, etc.) for posts about:
   - Product launches and growth stories
   - Revenue milestones ($1, $100 MRR, $1k MRR+)
   - Success strategies and failures

2. For each relevant post:
   - Read the post and top comments
   - Extract key details: product type, timeline, strategies, tools used, outcomes

3. Output results as a markdown table with these columns:
   | Product | Strategy | Tools | Results | URL |
   
   Each row should contain one indie hacker story with concise, actionable information in each column.
"""


async def test_reddit_journey_deconstructor_agent(  # Renamed for clarity
    provider: ChatProvider,
    model: str,
    reasoning_model: str,
    planning_model: str,
):
    context = ProcessingContext()
    search_agent = Agent(
        name="Reddit Indie Hacker Journey Deconstructor",  # New Agent Name
        objective=objective,
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

    print("\n\n--- Workspace Directory for artifacts ---")
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
            planning_model="gpt-4o-mini",
            reasoning_model="gpt-4o-mini",
        )
    )
