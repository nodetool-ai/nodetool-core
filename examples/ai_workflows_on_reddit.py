#!/usr/bin/env python3
"""
Reddit Agent using the discover → process → aggregate pattern.

Runtime plan:
- `discover_posts` (mode="discover") gathers Reddit post URLs using search/navigation tools.
- `process_posts` (mode="process") iterates over the discovered posts using templated natural-language
  instructions (e.g., "Fetch {post_url}.json via the browser and summarize it").
- `aggregate_report` (mode="aggregate") formats the collected results into the final markdown.

Discovery happens once at runtime (driven by the discover subtask); the executor automatically expands
the process subtask template for each discovered item.
"""

import asyncio

from nodetool.agents.agent import Agent
from nodetool.providers import get_provider
from nodetool.agents.tools import BrowserTool, GoogleSearchTool
from nodetool.metadata.types import Chunk, Provider
from nodetool.runtime.resources import ResourceScope
from nodetool.workflows.processing_context import ProcessingContext

objective = """
Goal: Find examples of AI workflows on Reddit and compile a markdown report of subreddits, posts, and top comments.

1) discover_posts (mode="discover")
   - Gather up to 10 recent Reddit posts relevant to "AI workflows".

2) process_posts (mode="process")
   - For each post, append ".json" and fetch via BrowserTool.
   - For each post, extract the post title, author, date, and full text.

3) aggregate_report (mode="aggregate")
    - Aggregate the posts into a markdown string.
    - If no posts are discovered or fetched, return a short markdown section explaining the limitation.
"""


async def main():
    async with ResourceScope():
        context = ProcessingContext()
        provider = await get_provider(Provider.HuggingFaceCerebras)
        model = "openai/gpt-oss-120b"
        search_agent = Agent(
            name="AI Workflow Examples on Reddit",
            objective=objective,
            provider=provider,
            model=model,
            enable_analysis_phase=True,
            enable_data_contracts_phase=True,
            tools=[BrowserTool(), GoogleSearchTool()],
            output_schema={"type": "string"},
        )

        # Execute each task in the plan
        print(f"Starting agent: {search_agent.name}\nObjective: {search_agent.objective}\n")
        async for item in search_agent.execute(context):
            if isinstance(item, Chunk):
                print(item.content, end="", flush=True)

        final_report = search_agent.get_results()
        if final_report:
            print("\n\n--- FINAL COMPILED REPORT ---")
            print(final_report)

        print(f"\nWorkspace: {context.workspace_dir}")


if __name__ == "__main__":
    asyncio.run(main())
