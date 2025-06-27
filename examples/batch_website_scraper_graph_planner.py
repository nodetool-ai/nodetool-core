#!/usr/bin/env python3
"""
Batch Website Scraper using GraphPlanner

This script demonstrates how to use GraphPlanner to create a workflow
for batch processing of multiple websites. It converts the original
SubTaskContext-based implementation to use the GraphPlanner system.

The workflow will:
- Take a list of websites as input
- Create a graph to scrape each website and extract key information
- Process websites in parallel where possible
- Output structured results in JSONL format

Usage:
    python batch_website_scraper_graph_planner.py
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from nodetool.agents.graph_planner import GraphPlanner
from nodetool.chat.providers import get_provider
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.types import Chunk, PlanningUpdate

# List of websites to scrape - demonstrating batch processing
WEBSITES_TO_SCRAPE = [
    "https://techcrunch.com",
    "https://www.theverge.com",
    "https://arstechnica.com",
    "https://www.wired.com",
    "https://hackernews.com",
]


async def create_and_execute_scraping_workflow(
    provider,
    model: str,
):
    """Create and execute a website scraping workflow using GraphPlanner"""

    # Define the objective for GraphPlanner
    objective = f"""
    For each website in the list {WEBSITES_TO_SCRAPE}
    - Extract the content of the website
    - Output the content of the website in a structured format
    """

    # Create GraphPlanner
    graph_planner = GraphPlanner(
        provider=provider,
        model=model,
        objective=objective,
        verbose=True,
    )

    # Plan the graph
    print(f"🔧 Planning workflow for batch website scraping...")
    print(f"📦 Target: {len(WEBSITES_TO_SCRAPE)} websites")
    print(f"🤖 Model: {model}\n")

    context = ProcessingContext(user_id="batch_scraper_user", auth_token="local_token")

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

    # Create execution request
    req = RunJobRequest(graph=graph)

    print(f"\n🚀 Executing website scraping workflow...")

    # Execute the workflow
    results = []
    async for msg in run_workflow(req, context=context, use_thread=False):
        print(f"📊 Workflow: {msg}")
        results.append(msg)

    return results


async def test_batch_website_scraper_with_graph_planner(
    provider, model: str, batch_size: int = 5
):
    """Test batch processing of website scraping with GraphPlanner."""

    try:
        # Execute scraping workflow
        start_time = asyncio.get_event_loop().time()

        results = await create_and_execute_scraping_workflow(
            provider=provider,
            model=model,
        )

        end_time = asyncio.get_event_loop().time()
        execution_time = end_time - start_time

        # Display final results
        print(f"\n\n{'='*60}")
        print("📋 FINAL RESULTS")
        print(f"{'='*60}\n")

        print(f"⏱️  Total execution time: {execution_time:.2f} seconds")
        print(f"🔧 Model used: {model}")
        print(f"📦 Target websites: {len(WEBSITES_TO_SCRAPE)}")

        print(results)

    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n✨ Batch website scraping complete!")


async def main():
    """Run the batch website scraper example using GraphPlanner."""

    # Test configurations - you can uncomment different providers
    test_configs = [
        # OpenAI - Good for general web scraping
        {"provider": Provider.OpenAI, "model": "o4-mini", "batch_size": 4},
        # Anthropic - Excellent context handling
        # {
        #     "provider": Provider.Anthropic,
        #     "model": "claude-3-5-sonnet-20241022",
        #     "batch_size": 5
        # },
        # Gemini - Fast and efficient
        # {
        #     "provider": Provider.Gemini,
        #     "model": "gemini-2.0-flash",
        #     "batch_size": 6
        # }
    ]

    for config in test_configs:
        print(f"\n{'#'*80}")
        print(f"# Testing GraphPlanner Website Scraper")
        print(f"# Provider: {config['provider'].value}")
        print(f"# Model: {config['model']}")
        print(f"# Batch size: {config['batch_size']}")
        print(f"{'#'*80}")

        await test_batch_website_scraper_with_graph_planner(
            provider=get_provider(config["provider"]),
            model=config["model"],
            batch_size=config["batch_size"],
        )

        # Add a delay between providers to avoid rate limits
        if len(test_configs) > 1:
            print("\n⏳ Waiting 10 seconds before next test...")
            await asyncio.sleep(10)


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
