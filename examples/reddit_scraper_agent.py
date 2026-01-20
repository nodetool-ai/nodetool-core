#!/usr/bin/env python3
"""
Reddit Scraping Agent for Community Research and Analysis

This script creates an intelligent Reddit research agent that:
1. Systematically searches for specific topics, issues, or trends in a subreddit
2. Uses GoogleSearchTool with targeted queries to find relevant Reddit posts
3. Uses BrowserTool to extract post content and comments
4. Analyzes findings to identify patterns and recurring themes
5. Generates a structured report with actionable insights

The agent outputs both JSON structured data and a markdown report, making it easy
to integrate findings into research, product development, or community analysis workflows.

Usage:
    python reddit_scraper_agent.py

By default, the script analyzes r/webdev for common developer challenges, but you can
easily modify it to research any subreddit or topic by changing the SUBREDDIT and
FOCUS_AREA variables.

Popular subreddits for analysis:
- r/webdev - Web development challenges and trends
- r/programming - General programming issues and discussions
- r/learnprogramming - Common learning obstacles and questions
- r/startups - Startup challenges and pain points
- r/smallbusiness - Small business owner concerns
- r/marketing - Marketing challenges and strategies
"""

import asyncio

import dotenv

from nodetool.agents.agent import Agent
from nodetool.agents.tools import BrowserTool, GoogleSearchTool
from nodetool.metadata.types import Provider
from nodetool.providers import get_provider
from nodetool.providers.base import BaseProvider
from nodetool.runtime.resources import ResourceScope
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk

# Load environment variables
dotenv.load_dotenv()


async def analyze_reddit_subreddit(
    provider: BaseProvider,
    model: str,
    subreddit: str = "n8n",
    focus_area: str = "customer pain points and issues",
):
    context = ProcessingContext()
    search_agent = Agent(
        name="Reddit Search Agent",
        objective=f"""
        You are an expert Reddit researcher specializing in customer feedback analysis.

        Your mission is to conduct comprehensive research on the r/{subreddit} subreddit to identify {focus_area}.

        **OUTPUT FORMAT**: Provide your final results in this simple JSON structure:
        {{
            "summary": "Brief 3-5 sentence overview of key findings",
            "posts_analyzed": number_of_posts_you_analyzed,
            "key_issues": ["main problem 1", "main problem 2", "main problem 3", ...],
            "recommendations": ["suggested improvement 1", "suggested improvement 2", ...]
        }}

        **RESEARCH APPROACH**:

        1. **Search Strategy**:
           - Use GoogleSearchTool with targeted queries like:
             * "site:reddit.com/r/{subreddit} problem"
             * "site:reddit.com/r/{subreddit} issue"
             * "site:reddit.com/r/{subreddit} help"
           - Find at least 10-15 relevant posts from the last 6 months

        2. **Data Collection**:
           - For each Reddit post URL found:
             * Use BrowserTool to visit the page but append ".json" to the url to get the content as a JSON file
             * Extract the post title, author, date, and full text
             * Capture the top 5-10 most relevant comments

        3. **Content Analysis**:
           - Categorize pain points by type:
             * Technical issues (bugs, errors, crashes)
             * Feature limitations or missing features
             * Documentation/learning curve problems
           - Identify recurring themes across multiple posts

        4. **Final Output**:
           - Write a brief summary (3-5 sentences) of your overall findings
           - List the main issues you discovered (focus on the most common/critical ones)
           - Provide actionable recommendations for improvement
           - Use the exact JSON format shown above

        Be thorough but concise. Focus on actionable insights that could help the {subreddit} community or related products/services.
        """,
        provider=provider,
        model=model,
        tools=[
            GoogleSearchTool(),
            BrowserTool(),
        ],
        output_schema={
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of findings",
                },
                "posts_analyzed": {
                    "type": "integer",
                    "description": "Number of posts analyzed",
                },
                "key_issues": {
                    "type": "array",
                    "description": "Main problems found",
                    "items": {"type": "string"},
                },
                "recommendations": {
                    "type": "array",
                    "description": "Suggested improvements",
                    "items": {"type": "string"},
                },
            },
        },
    )

    # 7. Execute each task in the plan
    async for item in search_agent.execute(context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    # Get the structured results
    results = search_agent.get_results()

    print("\n\n=== REDDIT RESEARCH COMPLETE ===")
    print(f"Workspace: {context.workspace_dir}")

    if results:
        print(f"\nPosts analyzed: {results.get('posts_analyzed', 'N/A')}")

        # Display key issues
        print("\n--- Key Issues Found ---")
        for i, issue in enumerate(results.get("key_issues", []), 1):
            print(f"\n{i}. {issue}")

        # Display recommendations
        print("\n--- Recommendations ---")
        for i, rec in enumerate(results.get("recommendations", []), 1):
            print(f"\n{i}. {rec}")

        # Save results to file
        import asyncio
        import json
        from pathlib import Path

        output_dir = Path(context.workspace_dir)
        output_file = output_dir / "reddit_analysis_report.json"

        def write_json_file():
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

        await asyncio.to_thread(write_json_file)

        print(f"\n\nReport saved to: {output_file}")
        print(f"Summary: {results.get('summary', 'No summary available')}")


async def main():
    # Configure the subreddit and focus area to analyze
    SUBREDDIT = "webdev"  # Popular, active subreddit with lots of technical discussions
    FOCUS_AREA = "common challenges, frustrations, and technical issues faced by web developers"  # Broader focus for more results

    print(f"Starting Reddit analysis for r/{SUBREDDIT}")
    print(f"Focus: {FOCUS_AREA}")
    print("-" * 60)

    async with ResourceScope():
        try:
            await analyze_reddit_subreddit(
                provider=await get_provider(Provider.HuggingFaceCerebras),
                model="openai/gpt-oss-120b",
                subreddit=SUBREDDIT,
                focus_area=FOCUS_AREA,
            )
        except Exception as e:
            print(f"Error during analysis: {e}")

        # Alternative subreddit examples for different use cases:

        # For analyzing programming learning challenges:
        # SUBREDDIT = "learnprogramming"
        # FOCUS_AREA = "common obstacles beginners face when learning to code"

        # For startup product development insights:
        # SUBREDDIT = "startups"
        # FOCUS_AREA = "technical challenges and product development issues"

        # For small business software needs:
        # SUBREDDIT = "smallbusiness"
        # FOCUS_AREA = "software pain points and automation needs"

        # For data science tool feedback:
        # SUBREDDIT = "datascience"
        # FOCUS_AREA = "tool limitations and workflow challenges"

        # Alternative provider examples:

        # Example: Run with Anthropic Claude (better for nuanced analysis)
        # await analyze_reddit_subreddit(
        #     provider=await get_provider(Provider.Anthropic),
        #     model="claude-3-5-sonnet-20241022",
        #     planning_model="claude-3-5-sonnet-20241022",
        #     reasoning_model="claude-3-5-sonnet-20241022",
        #     subreddit=SUBREDDIT,
        #     focus_area=FOCUS_AREA,
        # )

        # Example: Run with Google Gemini (good for large-scale analysis)
        # await analyze_reddit_subreddit(
        #     provider=await get_provider(Provider.Gemini),
        #     model="gemini-2.0-flash",
        #     planning_model="gemini-2.0-flash",
        #     reasoning_model="gemini-2.0-flash",
        #     subreddit=SUBREDDIT,
        #     focus_area=FOCUS_AREA,
        # )


if __name__ == "__main__":
    asyncio.run(main())
