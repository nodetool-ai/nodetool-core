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
from nodetool.agents.tools import BrowserTool, GoogleSearchTool
from nodetool.metadata.types import Chunk, Provider
from nodetool.providers import get_provider
from nodetool.runtime.resources import ResourceScope
from nodetool.workflows.processing_context import ProcessingContext

objective = """
Goal: Find examples of AI workflows on Reddit and compile a markdown report of subreddits, posts, and top comments.

1) discover_posts (mode="discover")
   - Gather up to 50 recent Reddit posts relevant to "AI workflows".
   - Try many different search queries to find posts.
   - Discover related keywords to find posts.

2) process_posts (mode="process")
   - For each post, strip trailing slash and append ".json" and fetch via BrowserTool.
   - For example, if the post url is https://www.reddit.com/r/AI/comments/1234567890/, the url to fetch is https://www.reddit.com/r/AI/comments/1234567890.json
   - For each post, extract the post title, selftext, and comments.

3) aggregate_report (mode="aggregate")
    - Aggregate the posts into structured markdown.
    - Summarize the posts into a short summary.
    - Summarize each post+comments as separate markdown sections.
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
