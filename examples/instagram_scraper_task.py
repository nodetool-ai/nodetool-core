#!/usr/bin/env python3
"""
Instagram Trends Analysis Task

This script demonstrates using the SubTaskContext to analyze Instagram trends,
viral content, and emerging patterns. It uses search and browser tools to gather
comprehensive data about current Instagram trends, including hashtags, content themes,
and engagement patterns.

The task outputs structured JSON data that can be used for:
- Content strategy planning
- Social media marketing insights
- Trend analysis and reporting
- Competitive intelligence

Usage:
    python instagram_scraper_task.py
"""

import asyncio
from nodetool.chat.providers import get_provider
from nodetool.agents.tools import BrowserTool, GoogleSearchTool
from nodetool.chat.providers.base import ChatProvider
from nodetool.metadata.types import Provider, Task, SubTask
from nodetool.agents.sub_task_context import SubTaskContext
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk, TaskUpdate
import json
from pathlib import Path
from datetime import datetime

import dotenv

# Load environment variables
dotenv.load_dotenv()


async def analyze_instagram_trends(
    provider: ChatProvider,
    model: str,
    focus_area: str = "general",
):
    # 1. Set up workspace directory
    context = ProcessingContext()
    workspace_dir = context.workspace_dir

    # 3. Set up browser and search tools
    tools = [
        BrowserTool(),
        GoogleSearchTool(),
    ]

    # 4. Create a comprehensive task
    task = Task(
        title="Instagram Trends Deep Analysis",
        description=f"Comprehensive analysis of current Instagram trends with focus on {focus_area}",
        subtasks=[],  # We'll add the subtask directly to the SubTaskContext
    )

    # 5. Create a detailed subtask with focus-specific prompting
    focus_prompts = {
        "general": "covering all major categories including fashion, lifestyle, tech, and entertainment",
        "fashion": "specifically related to fashion, style, outfits, and clothing brands",
        "tech": "focusing on technology, gadgets, apps, and digital trends",
        "fitness": "related to fitness, health, wellness, and workout routines",
        "food": "covering food trends, recipes, restaurants, and culinary content",
        "travel": "focusing on travel destinations, experiences, and tourism",
        "business": "related to entrepreneurship, marketing, and business growth",
    }

    focus_context = focus_prompts.get(focus_area, "across all content categories")

    subtask = SubTask(
        content=f"""
        Conduct a comprehensive Instagram trends analysis {focus_context}.
        
        Your mission is to provide actionable insights by:
        
        1. **Trend Discovery**:
           - Use GoogleSearchTool with queries like:
             * "Instagram trending hashtags {datetime.now().strftime('%B %Y')}"
             * "viral Instagram content {focus_area}"
             * "Instagram trends report latest"
             * "most popular Instagram {focus_area} accounts"
           - Search for at least 5-10 different trend categories
        
        2. **Data Collection**:
           - For each trend found:
             * Identify the main hashtag(s) associated with it
             * Understand why it's trending (cultural moment, challenge, event, etc.)
             * Find 2-3 example posts demonstrating the trend
             * Note engagement patterns and audience demographics if available
        
        3. **Trend Analysis**:
           - Categorize trends by:
             * Content type (video, photo, carousel, reels)
             * Audience appeal (Gen Z, Millennials, broad appeal)
             * Longevity (ephemeral, seasonal, evergreen)
             * Commercial potential (brand-friendly, organic only)
        
        4. **Insights Generation**:
           - Identify patterns across trends
           - Note which content formats are performing best
           - Highlight emerging micro-trends
           - Suggest content opportunities
        
        Be thorough and ensure all data is current (within the last 30 days if possible).
        Focus on trends that have genuine engagement, not just high hashtag counts.
        """,
        output_file=f"instagram_trends_{focus_area}_{datetime.now().strftime('%Y%m%d')}.json",
        input_files=[],
        output_type="json",
        output_schema=json.dumps(
            {
                "type": "object",
                "properties": {
                    "analysis_date": {
                        "type": "string",
                        "description": "Date of analysis in ISO format",
                    },
                    "focus_area": {
                        "type": "string",
                        "description": "The focus area of this analysis",
                    },
                    "trends": {
                        "type": "array",
                        "description": "List of identified Instagram trends",
                        "items": {
                            "type": "object",
                            "properties": {
                                "trend_name": {
                                    "type": "string",
                                    "description": "Name or title of the trend",
                                },
                                "primary_hashtags": {
                                    "type": "array",
                                    "description": "Main hashtags associated with this trend",
                                    "items": {"type": "string"},
                                    "minItems": 1,
                                    "maxItems": 5,
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Detailed description of the trend and why it's popular",
                                },
                                "trend_category": {
                                    "type": "string",
                                    "enum": [
                                        "challenge",
                                        "aesthetic",
                                        "meme",
                                        "seasonal",
                                        "social_cause",
                                        "product",
                                        "lifestyle",
                                        "other",
                                    ],
                                    "description": "Category of the trend",
                                },
                                "content_type": {
                                    "type": "string",
                                    "enum": [
                                        "reels",
                                        "photos",
                                        "carousel",
                                        "stories",
                                        "mixed",
                                    ],
                                    "description": "Primary content format for this trend",
                                },
                                "popularity_metrics": {
                                    "type": "object",
                                    "properties": {
                                        "engagement_level": {
                                            "type": "string",
                                            "enum": [
                                                "viral",
                                                "high",
                                                "moderate",
                                                "emerging",
                                            ],
                                            "description": "Current engagement level",
                                        },
                                        "growth_trajectory": {
                                            "type": "string",
                                            "enum": [
                                                "explosive",
                                                "steady_growth",
                                                "plateau",
                                                "declining",
                                            ],
                                            "description": "How the trend is evolving",
                                        },
                                        "estimated_reach": {
                                            "type": "string",
                                            "description": "Estimated audience reach (e.g., '1M+', '500K-1M')",
                                        },
                                    },
                                    "required": [
                                        "engagement_level",
                                        "growth_trajectory",
                                    ],
                                },
                                "target_audience": {
                                    "type": "object",
                                    "properties": {
                                        "primary_demographic": {
                                            "type": "string",
                                            "description": "Main audience demographic",
                                        },
                                        "age_range": {
                                            "type": "string",
                                            "description": "Typical age range of participants",
                                        },
                                        "interests": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Common interests of the audience",
                                        },
                                    },
                                },
                                "example_posts": {
                                    "type": "array",
                                    "description": "Representative posts for this trend",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "post_url": {
                                                "type": "string",
                                                "format": "uri",
                                            },
                                            "account_handle": {"type": "string"},
                                            "caption_snippet": {
                                                "type": "string",
                                                "description": "First 100 characters of caption",
                                            },
                                            "engagement_stats": {
                                                "type": "object",
                                                "properties": {
                                                    "likes": {"type": "string"},
                                                    "comments": {"type": "string"},
                                                    "shares": {"type": "string"},
                                                },
                                            },
                                            "post_date": {"type": "string"},
                                        },
                                        "required": ["post_url", "account_handle"],
                                    },
                                    "minItems": 1,
                                    "maxItems": 3,
                                },
                                "brand_opportunity": {
                                    "type": "object",
                                    "properties": {
                                        "commercial_viability": {
                                            "type": "string",
                                            "enum": ["high", "medium", "low"],
                                            "description": "Potential for brand participation",
                                        },
                                        "suggested_approach": {
                                            "type": "string",
                                            "description": "How brands could authentically engage",
                                        },
                                    },
                                },
                                "longevity_prediction": {
                                    "type": "string",
                                    "description": "Expected lifespan of the trend (e.g., '1-2 weeks', '1-3 months', 'evergreen')",
                                },
                            },
                            "required": [
                                "trend_name",
                                "primary_hashtags",
                                "description",
                                "trend_category",
                                "content_type",
                                "popularity_metrics",
                                "example_posts",
                            ],
                        },
                        "minItems": 5,
                        "maxItems": 10,
                    },
                    "insights_summary": {
                        "type": "object",
                        "properties": {
                            "key_findings": {
                                "type": "array",
                                "description": "Top 3-5 key insights from the analysis",
                                "items": {"type": "string"},
                                "minItems": 3,
                                "maxItems": 5,
                            },
                            "content_recommendations": {
                                "type": "array",
                                "description": "Actionable content strategy recommendations",
                                "items": {"type": "string"},
                                "minItems": 3,
                                "maxItems": 5,
                            },
                            "emerging_opportunities": {
                                "type": "array",
                                "description": "Early-stage trends to watch",
                                "items": {"type": "string"},
                                "minItems": 2,
                                "maxItems": 4,
                            },
                        },
                        "required": [
                            "key_findings",
                            "content_recommendations",
                            "emerging_opportunities",
                        ],
                    },
                },
                "required": [
                    "analysis_date",
                    "focus_area",
                    "trends",
                    "insights_summary",
                ],
            }
        ),
    )

    # Add the subtask to the task
    task.subtasks = [subtask]

    # 6. Create the SubTaskContext
    subtask_context = SubTaskContext(
        task=task,
        subtask=subtask,
        processing_context=ProcessingContext(),
        tools=tools,
        model=model,
        provider=provider,
        max_iterations=20,
    )

    # 7. Execute the subtask
    async for event in subtask_context.execute():
        if isinstance(event, Chunk):
            print(event.content, end="")
        elif isinstance(event, TaskUpdate):
            print(f"Task Update: {event.event}")

    # Check if output file was created
    output_path = Path(workspace_dir) / subtask.output_file
    if output_path.exists():
        with open(output_path, "r") as f:
            result = json.load(f)

        print("\n\n=== INSTAGRAM TRENDS ANALYSIS COMPLETE ===")
        print(f"Analysis Date: {result.get('analysis_date', 'N/A')}")
        print(f"Focus Area: {result.get('focus_area', 'N/A')}")
        print(f"Total Trends Identified: {len(result.get('trends', []))}")

        # Display trend summary
        print("\n--- Top Trends ---")
        for i, trend in enumerate(result.get("trends", [])[:5], 1):
            print(f"\n{i}. {trend.get('trend_name', 'Unnamed Trend')}")
            print(f"   Hashtags: {', '.join(trend.get('primary_hashtags', []))}")
            print(
                f"   Category: {trend.get('trend_category', 'N/A')} | Type: {trend.get('content_type', 'N/A')}"
            )
            metrics = trend.get("popularity_metrics", {})
            print(
                f"   Engagement: {metrics.get('engagement_level', 'N/A')} | Growth: {metrics.get('growth_trajectory', 'N/A')}"
            )

        # Display insights
        insights = result.get("insights_summary", {})
        if insights.get("key_findings"):
            print("\n--- Key Findings ---")
            for finding in insights["key_findings"]:
                print(f"• {finding}")

        if insights.get("content_recommendations"):
            print("\n--- Content Recommendations ---")
            for rec in insights["content_recommendations"]:
                print(f"• {rec}")

        # Save full report
        print(f"\n\nFull report saved to: {output_path}")

        # Also create a markdown summary
        md_path = output_path.with_suffix(".md")
        with open(md_path, "w") as f:
            f.write("# Instagram Trends Analysis Report\n\n")
            f.write(f"**Date:** {result.get('analysis_date', 'N/A')}\n")
            f.write(f"**Focus:** {result.get('focus_area', 'N/A')}\n\n")

            f.write("## Trends Overview\n\n")
            for trend in result.get("trends", []):
                f.write(f"### {trend.get('trend_name', 'Unnamed')}\n")
                f.write(
                    f"- **Hashtags:** {', '.join(trend.get('primary_hashtags', []))}\n"
                )
                f.write(f"- **Description:** {trend.get('description', 'N/A')}\n")
                f.write(
                    f"- **Engagement Level:** {trend.get('popularity_metrics', {}).get('engagement_level', 'N/A')}\n\n"
                )

        print(f"Markdown summary saved to: {md_path}")
    else:
        print("\nOutput file was not created!")


if __name__ == "__main__":
    # Choose your focus area for analysis
    FOCUS = (
        "general"  # Options: general, fashion, tech, fitness, food, travel, business
    )

    print("Starting Instagram Trends Analysis")
    print(f"Focus Area: {FOCUS}")
    print("-" * 60)

    # Example: Run with OpenAI
    try:
        asyncio.run(
            analyze_instagram_trends(
                provider=get_provider(Provider.OpenAI),
                model="gpt-4o-mini",
                focus_area=FOCUS,
            )
        )
    except Exception as e:
        print(f"Error during analysis: {e}")

    # Alternative examples for different focus areas:

    # Fashion trends analysis
    # asyncio.run(
    #     analyze_instagram_trends(
    #         provider=get_provider(Provider.OpenAI),
    #         model="gpt-4o-mini",
    #         focus_area="fashion"
    #     )
    # )

    # Tech and gadget trends
    # asyncio.run(
    #     analyze_instagram_trends(
    #         provider=get_provider(Provider.Gemini),
    #         model="gemini-2.0-flash",
    #         focus_area="tech"
    #     )
    # )

    # Fitness and wellness trends
    # asyncio.run(
    #     analyze_instagram_trends(
    #         provider=get_provider(Provider.Anthropic),
    #         model="claude-3-5-sonnet-20241022",
    #         focus_area="fitness"
    #     )
    # )
