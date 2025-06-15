#!/usr/bin/env python3
"""
Reddit Indie Hacker Journey Analyzer using GraphPlanner

This script demonstrates how to use GraphPlanner to create a workflow
for analyzing indie hacker success stories from Reddit. It converts the original
Agent-based implementation to use the GraphPlanner system.

The workflow will:
1. Search Reddit for indie hacker stories and product launches
2. Extract content from relevant posts and comments
3. Analyze the stories to extract actionable insights
4. Generate a structured markdown report with key findings

Usage:
    python indie_hacker_journey_graph_planner.py
"""

import asyncio
from typing import List

from nodetool.agents.graph_planner import GraphPlanner
from nodetool.chat.providers import get_provider
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.types import Chunk, PlanningUpdate

# Search terms for finding indie hacker stories
INDIE_HACKER_SEARCH_TERMS = [
    "site:reddit.com/r/indiehackers revenue milestone",
    "site:reddit.com/r/SideProject launched product",
    "site:reddit.com/r/startups first 1000 users",
    "site:reddit.com/r/indiehackers $1k MRR",
    "site:reddit.com/r/indiehackers $10k MRR",
    "site:reddit.com/r/SideProject bootstrap success",
    "site:reddit.com/r/startups indie maker journey",
    "site:reddit.com/r/indiehackers product launch story",
]

# Categories to analyze in the stories
ANALYSIS_CATEGORIES = [
    "Product Type",
    "Strategy Used",
    "Tools and Technologies",
    "Timeline to Success",
    "Revenue/Growth Metrics",
    "Key Learnings",
    "Common Pitfalls",
]


async def create_and_execute_indie_hacker_workflow(
    provider,
    model: str,
):
    """Create and execute an indie hacker analysis workflow using GraphPlanner"""

    # Define the objective for GraphPlanner
    objective = f"""
    Create a workflow to analyze indie hacker success stories from Reddit:
    
    1. Search Reddit for indie hacker stories using these search terms, use google search to find the most relevant posts:
       {', '.join(INDIE_HACKER_SEARCH_TERMS[:4])}  # Limit for brevity
    
    2. For each relevant post found:
       - Extract the post content and top comments
       - Identify key details about the product/service
       - Extract timeline, strategies, tools used, and outcomes
    
    3. Analyze all collected stories to identify patterns and insights across:
       - Product types and categories
       - Common strategies for growth
       - Popular tools and technologies
       - Timeline patterns for reaching milestones
       - Revenue progression patterns
    
    4. Generate a comprehensive markdown report with:
       - Summary table of all stories found
       - Analysis of common patterns and strategies
       - Actionable insights for aspiring indie hackers
       - Recommended tools and approaches based on success stories
    
    The final output should be a well-structured markdown report that provides
    actionable value to someone starting their indie hacker journey.
    """

    # Create GraphPlanner
    graph_planner = GraphPlanner(
        provider=provider,
        model=model,
        objective=objective,
        verbose=True,
        inputs={
            "search_terms": INDIE_HACKER_SEARCH_TERMS,
            "analysis_categories": ANALYSIS_CATEGORIES,
        },
    )

    # Plan the graph
    print(f"🔧 Planning workflow for indie hacker journey analysis...")
    print(f"📦 Target: Reddit indie hacker success stories")
    print(f"🤖 Model: {model}")
    print(f"🔍 Search terms: {len(INDIE_HACKER_SEARCH_TERMS)}")
    print(f"📊 Analysis categories: {len(ANALYSIS_CATEGORIES)}\n")

    context = ProcessingContext(
        user_id="indie_hacker_analyzer", auth_token="local_token"
    )

    # Create the graph through planning phases
    async for update in graph_planner.create_graph(context):
        if isinstance(update, PlanningUpdate):
            print(f"📋 Planning: {update.phase} - {update.status}")
        elif isinstance(update, Chunk):
            print(f"💭 {update.content}", end="", flush=True)

    if not graph_planner.graph:
        raise ValueError("Failed to create workflow graph")

    graph = graph_planner.graph
    print(f"\n✅ Generated workflow with {len(graph.nodes)} nodes")

    # Prepare inputs for execution
    inputs = {
        "search_terms": INDIE_HACKER_SEARCH_TERMS,
        "analysis_categories": ANALYSIS_CATEGORIES,
    }

    # Create execution request
    req = RunJobRequest(
        graph=graph,
        params=inputs,
    )

    print(f"\n🚀 Executing indie hacker analysis workflow...")

    # Execute the workflow
    results = []
    async for msg in run_workflow(req, context=context, use_thread=False):
        print(f"📊 Workflow: {msg}")
        results.append(msg)

    return results


async def test_indie_hacker_journey_graph_planner(provider, model: str):
    """Test indie hacker journey analysis with GraphPlanner."""

    try:
        # Execute the workflow
        start_time = asyncio.get_event_loop().time()

        results = await create_and_execute_indie_hacker_workflow(
            provider=provider,
            model=model,
        )

        end_time = asyncio.get_event_loop().time()
        execution_time = end_time - start_time

        # Display final results
        print(f"\n\n{'='*60}")
        print("📋 FINAL ANALYSIS RESULTS")
        print(f"{'='*60}\n")

        print(f"⏱️  Total execution time: {execution_time:.2f} seconds")
        print(f"🤖 Model used: {model}")
        print(f"🔍 Search terms processed: {len(INDIE_HACKER_SEARCH_TERMS)}")
        print(f"📊 Analysis categories: {len(ANALYSIS_CATEGORIES)}")

        # Print results summary
        if results:
            print(f"\n📈 Workflow generated {len(results)} result items")

            # Look for final report in results
            for result in results:
                if (
                    hasattr(result, "content")
                    and "FINAL" in str(result.content).upper()
                ):
                    print(f"\n📝 Final Report Preview:")
                    print(f"{str(result.content)[:500]}...")
                    break

        print(f"\n💡 Full analysis report saved to workspace: {context.workspace_dir}")

    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n✨ Indie hacker journey analysis complete!")


async def main():
    """Run the indie hacker journey analyzer using GraphPlanner."""

    # Test configurations - you can uncomment different providers
    test_configs = [
        # OpenAI - Good for analysis and reasoning
        {"provider": Provider.OpenAI, "model": "gpt-4o-mini"},
        # Anthropic - Excellent for detailed analysis and structured output
        # {
        #     "provider": Provider.Anthropic,
        #     "model": "claude-3-5-sonnet-20241022"
        # },
        # Gemini - Fast and efficient for large-scale analysis
        # {
        #     "provider": Provider.Gemini,
        #     "model": "gemini-2.0-flash"
        # }
    ]

    for config in test_configs:
        print(f"\n{'#'*80}")
        print(f"# Testing GraphPlanner Indie Hacker Journey Analyzer")
        print(f"# Provider: {config['provider'].value}")
        print(f"# Model: {config['model']}")
        print(f"{'#'*80}")

        await test_indie_hacker_journey_graph_planner(
            provider=get_provider(config["provider"]),
            model=config["model"],
        )

        # Add a delay between providers to avoid rate limits
        if len(test_configs) > 1:
            print("\n⏳ Waiting 10 seconds before next test...")
            await asyncio.sleep(10)


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
