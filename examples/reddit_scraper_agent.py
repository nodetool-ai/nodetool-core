#!/usr/bin/env python3
"""
Reddit Scraping Agent for Community Research and Analysis

This script creates an intelligent Reddit research agent that:
1. Systematically searches for specific topics, issues, or trends in a subreddit
2. Uses GoogleSearchTool with targeted queries to find relevant Reddit posts
3. Uses BrowserTool to extract post content and comments
4. Analyzes findings to identify patterns and recurring themes
5. Generates a structured report with actionable insights

**NEW: DYNAMIC SUBTASK SUPPORT**
The agent can now dynamically add subtasks during execution! If the agent discovers
multiple distinct topics or areas that need deeper investigation, it can use the
add_subtask tool to create additional research tasks on-the-fly. This enables more
thorough and organized research that adapts to what the agent discovers.

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

from nodetool.agents.agent import Agent
from nodetool.providers import get_provider
from nodetool.agents.tools import BrowserTool, GoogleSearchTool
from nodetool.providers.base import BaseProvider
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk

import dotenv

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
             * "site:reddit.com/r/{subreddit} error"
             * "site:reddit.com/r/{subreddit} bug"
             * "site:reddit.com/r/{subreddit} doesn't work"
             * "site:reddit.com/r/{subreddit} frustrated"
             * "site:reddit.com/r/{subreddit} confused"
           - Find at least 10-15 relevant posts from the last 6 months
           - Prioritize posts with high engagement (many comments/upvotes)

        2. **Data Collection**:
           - For each Reddit post URL found:
             * Use BrowserTool to visit the page
             * Extract the post title, author, date, and full text
             * Capture the top 5-10 most relevant comments
             * Note the number of upvotes (indicates issue prevalence)
             * Record any solutions or workarounds mentioned
             * Pay attention to moderator or official responses

        3. **Content Analysis**:
           - Categorize pain points by type:
             * Technical issues (bugs, errors, crashes)
             * Feature limitations or missing features
             * Documentation/learning curve problems
             * Integration difficulties
             * Performance issues
             * UX/UI concerns
           - Identify recurring themes across multiple posts
           - Note the severity and frequency of each issue type
           - Track user sentiment and frustration levels

        4. **Final Output**:
           - Write a brief summary (3-5 sentences) of your overall findings
           - List the main issues you discovered (focus on the most common/critical ones)
           - Provide actionable recommendations for improvement
           - Use the exact JSON format shown above

        **Dynamic Subtask Addition**:
        You have access to the add_subtask tool. If during your research you discover
        distinct categories of issues that warrant deeper investigation (e.g., specific
        technical problems, integration challenges, feature requests), use add_subtask
        to create focused research tasks for each category. This allows for more thorough
        analysis and better organized results.

        Be thorough but concise. Focus on actionable insights that could help the {subreddit} community or related products/services.
        """,
        provider=provider,
        model=model,
        enable_analysis_phase=True,
        enable_data_contracts_phase=True,
        tools=[
            GoogleSearchTool(),
            BrowserTool(),
        ],
        output_schema={
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of findings"
                },
                "posts_analyzed": {
                    "type": "integer",
                    "description": "Number of posts analyzed"
                },
                "key_issues": {
                    "type": "array",
                    "description": "Main problems found",
                    "items": {"type": "string"}
                },
                "recommendations": {
                    "type": "array",
                    "description": "Suggested improvements",
                    "items": {"type": "string"}
                }
            }
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
        import json
        from pathlib import Path

        output_dir = Path(context.workspace_dir)
        output_file = output_dir / "reddit_analysis_report.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n\nReport saved to: {output_file}")
        print(f"Summary: {results.get('summary', 'No summary available')}")


if __name__ == "__main__":
    # Configure the subreddit and focus area to analyze
    SUBREDDIT = "webdev"  # Popular, active subreddit with lots of technical discussions
    FOCUS_AREA = "common challenges, frustrations, and technical issues faced by web developers"  # Broader focus for more results

    print(f"Starting Reddit analysis for r/{SUBREDDIT}")
    print(f"Focus: {FOCUS_AREA}")
    print("-" * 60)

    # Example: Run with OpenAI GPT-4o Mini
    try:
        asyncio.run(
            analyze_reddit_subreddit(
                provider=get_provider(Provider.HuggingFaceCerebras),
                model="openai/gpt-oss-120b",
                subreddit=SUBREDDIT,
                focus_area=FOCUS_AREA,
            )
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
    # asyncio.run(
    #     analyze_reddit_subreddit(
    #         provider=get_provider(Provider.Anthropic),
    #         model="claude-3-5-sonnet-20241022",
    #         planning_model="claude-3-5-sonnet-20241022",
    #         reasoning_model="claude-3-5-sonnet-20241022",
    #         subreddit=SUBREDDIT,
    #         focus_area=FOCUS_AREA,
    #     )
    # )

    # Example: Run with Google Gemini (good for large-scale analysis)
    # asyncio.run(
    #     analyze_reddit_subreddit(
    #         provider=get_provider(Provider.Gemini),
    #         model="gemini-2.0-flash",
    #         planning_model="gemini-2.0-flash",
    #         reasoning_model="gemini-2.0-flash",
    #         subreddit=SUBREDDIT,
    #         focus_area=FOCUS_AREA,
    #     )
    # )
