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
from nodetool.providers import get_provider
from nodetool.agents.tools import BrowserTool, GoogleSearchTool
from nodetool.providers.base import BaseProvider
from nodetool.metadata.types import Provider
from nodetool.ui.console import AgentConsole
from nodetool.workflows.processing_context import ProcessingContext

objective = """
Find example for AI workflows on Reddit.

Tasks:
1. Use Google Search to find examples of AI workflows on Reddit.
2. For each url, append .json to the url and use the BrowserTool to fetch the content
3. Output results as a markdown file

Report structure:
## Workflow: [Workflow Name]
Url: [Workflow URL]
Summary: [Summary of the Workflow]
Comments: [Key Comments from the Workflow]
"""


async def test_reddit_journey_deconstructor_agent(  # Renamed for clarity
    provider: BaseProvider,
    model: str,
):
    context = ProcessingContext()
    search_agent = Agent(
        name="AI Workflow Examples on Reddit",
        objective=objective,
        provider=provider,
        model=model,
        enable_analysis_phase=False,
        enable_data_contracts_phase=True,
        tools=[
            GoogleSearchTool(),
            BrowserTool(),
        ],
        display_manager=AgentConsole(),
    )

    # Execute each task in the plan
    print(f"Starting agent: {search_agent.name}\nObjective: {search_agent.objective}\n")
    async for item in search_agent.execute(context):
        pass

    final_report = search_agent.get_results()
    if final_report:
        print("\n\n--- FINAL COMPILED REPORT ---")
        print(final_report)

    print(f"\nWorkspace: {context.workspace_dir}")


if __name__ == "__main__":
    asyncio.run(
        test_reddit_journey_deconstructor_agent(
            provider=get_provider(
                Provider.HuggingFaceCerebras
            ),
            model="openai/gpt-oss-120b",
        )
    )
