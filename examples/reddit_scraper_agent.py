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

from nodetool.agents.agent import Agent
from nodetool.chat.providers import get_provider
from nodetool.agents.tools import BrowserTool, GoogleSearchTool
from nodetool.chat.providers.base import ChatProvider
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk

import dotenv

# Load environment variables
dotenv.load_dotenv()


async def analyze_reddit_subreddit(
    provider: ChatProvider,
    model: str,
    reasoning_model: str,
    planning_model: str,
    subreddit: str = "n8n",
    focus_area: str = "customer pain points and issues",
):
    context = ProcessingContext()
    search_agent = Agent(
        name="Reddit Search Agent",
        objective=f"""
        You are an expert Reddit researcher specializing in customer feedback analysis.
        
        Your mission is to conduct comprehensive research on the r/{subreddit} subreddit to identify {focus_area}.
        
        Follow this structured approach:
        
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
        
        4. **Report Generation**:
           Create a comprehensive analysis with:
           - Executive summary of key findings
           - Top 5 most critical pain points with evidence
           - Categorized list of all identified issues
           - Direct quotes from users illustrating each pain point
           - Potential solutions or feature requests mentioned by users
           - Recommendations for product improvements with priority levels
        
        Be thorough but concise. Focus on actionable insights that could help the {subreddit} community or related products/services.
        """,
        provider=provider,
        model=model,
        reasoning_model=reasoning_model,
        planning_model=planning_model,
        enable_analysis_phase=True,
        enable_data_contracts_phase=True,
        tools=[
            GoogleSearchTool(),
            BrowserTool(),
        ],
        output_type="json",
        output_schema={
            "type": "object",
            "properties": {
                "executive_summary": {
                    "type": "string",
                    "description": "High-level summary of key findings (3-5 sentences)",
                },
                "total_posts_analyzed": {
                    "type": "integer",
                    "description": "Number of Reddit posts analyzed",
                },
                "date_range": {
                    "type": "string",
                    "description": "Date range of posts analyzed",
                },
                "top_pain_points": {
                    "type": "array",
                    "description": "Top 5 most critical pain points identified",
                    "items": {
                        "type": "object",
                        "properties": {
                            "issue": {
                                "type": "string",
                                "description": "Brief description of the pain point",
                            },
                            "category": {
                                "type": "string",
                                "enum": [
                                    "technical",
                                    "feature",
                                    "documentation",
                                    "integration",
                                    "performance",
                                    "other",
                                ],
                                "description": "Category of the issue",
                            },
                            "frequency": {
                                "type": "integer",
                                "description": "Number of posts mentioning this issue",
                            },
                            "severity": {
                                "type": "string",
                                "enum": ["critical", "high", "medium", "low"],
                                "description": "Severity level based on user impact",
                            },
                            "example_quotes": {
                                "type": "array",
                                "description": "2-3 direct quotes from users experiencing this issue",
                                "items": {"type": "string"},
                                "minItems": 1,
                                "maxItems": 3,
                            },
                            "post_links": {
                                "type": "array",
                                "description": "Links to posts discussing this issue",
                                "items": {"type": "string", "format": "uri"},
                                "minItems": 1,
                                "maxItems": 3,
                            },
                        },
                        "required": [
                            "issue",
                            "category",
                            "frequency",
                            "severity",
                            "example_quotes",
                            "post_links",
                        ],
                    },
                    "minItems": 3,
                    "maxItems": 5,
                },
                "all_issues": {
                    "type": "object",
                    "description": "All issues categorized by type",
                    "properties": {
                        "technical": {"type": "array", "items": {"type": "string"}},
                        "feature": {"type": "array", "items": {"type": "string"}},
                        "documentation": {"type": "array", "items": {"type": "string"}},
                        "integration": {"type": "array", "items": {"type": "string"}},
                        "performance": {"type": "array", "items": {"type": "string"}},
                        "other": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "user_solutions": {
                    "type": "array",
                    "description": "Workarounds or solutions mentioned by users",
                    "items": {
                        "type": "object",
                        "properties": {
                            "problem": {"type": "string"},
                            "solution": {"type": "string"},
                            "source_url": {"type": "string", "format": "uri"},
                        },
                    },
                },
                "recommendations": {
                    "type": "array",
                    "description": "Actionable recommendations for product improvement",
                    "items": {
                        "type": "object",
                        "properties": {
                            "recommendation": {"type": "string"},
                            "priority": {
                                "type": "string",
                                "enum": ["urgent", "high", "medium", "low"],
                            },
                            "rationale": {"type": "string"},
                        },
                    },
                    "minItems": 3,
                    "maxItems": 7,
                },
                "markdown_report": {
                    "type": "string",
                    "description": "Full markdown-formatted report with all findings",
                },
            },
            "required": [
                "executive_summary",
                "total_posts_analyzed",
                "date_range",
                "top_pain_points",
                "all_issues",
                "recommendations",
                "markdown_report",
            ],
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
        print(f"\nTotal posts analyzed: {results.get('total_posts_analyzed', 'N/A')}")
        print(f"Date range: {results.get('date_range', 'N/A')}")

        # Display top pain points
        print("\n--- Top Pain Points ---")
        for i, pain_point in enumerate(results.get("top_pain_points", []), 1):
            print(f"\n{i}. {pain_point['issue']} (Category: {pain_point['category']})")
            print(
                f"   Frequency: {pain_point['frequency']} posts | Severity: {pain_point['severity']}"
            )

        # Display recommendations
        print("\n--- Recommendations ---")
        for i, rec in enumerate(results.get("recommendations", []), 1):
            print(f"\n{i}. [{rec['priority'].upper()}] {rec['recommendation']}")

        # Save full report to file
        import json
        from pathlib import Path

        output_dir = Path(context.workspace_dir)
        output_file = output_dir / "reddit_analysis_report.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n\nFull report saved to: {output_file}")

        # Also save markdown report if available
        if results.get("markdown_report"):
            md_file = output_dir / "reddit_analysis_report.md"
            with open(md_file, "w") as f:
                f.write(results["markdown_report"])
            print(f"Markdown report saved to: {md_file}")


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
                provider=get_provider(Provider.OpenAI),
                model="gpt-4o-mini",
                planning_model="gpt-4o-mini",
                reasoning_model="gpt-4o-mini",
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
