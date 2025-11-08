#!/usr/bin/env python3
"""
Task Planner Batch Processing Test: Multi-Site Content Analyzer

This script demonstrates the TaskPlanner's batch processing capabilities by:
1. Creating a plan to analyze content from multiple websites
2. Automatically detecting when batch processing is needed
3. Generating subtasks with proper batch configuration
4. Processing results efficiently to avoid context overflow

The test objective intentionally involves processing many items to trigger
the batch processing logic in the TaskPlanner.
"""

import asyncio
import json
from pathlib import Path

from nodetool.agents.agent import Agent
from nodetool.providers import get_provider
from nodetool.agents.tools import BrowserTool, GoogleSearchTool
from nodetool.providers.base import BaseProvider
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk, TaskUpdate
from nodetool.runtime.resources import ResourceScope

# Objective designed to trigger batch processing
objective = """
Analyze the top 20 technology news websites to identify emerging AI trends.

Tasks:
1. Create a list of 20 popular technology news websites including:
   - TechCrunch, The Verge, Ars Technica, Wired, Engadget
   - VentureBeat, MIT Technology Review, IEEE Spectrum
   - The Information, Protocol, Hacker News
   - And 9 more similar sites

2. For EACH website (all 20):
   - Visit the homepage
   - Find articles related to AI, machine learning, or LLMs
   - Extract article titles and summaries
   - Note the publication date

3. Analyze the collected data to identify:
   - Most common AI topics across all sites
   - Emerging trends mentioned by multiple sources
   - Key companies or technologies being discussed

4. Create a comprehensive report with:
   - Summary of top 5 AI trends
   - Table of all articles found (title, site, date, topic)
   - Analysis of which sites cover which topics most

Output the results as a structured JSON report with trends analysis and article listings.
"""


async def test_task_planner_batch_processing(
    provider: BaseProvider,
    model: str,
    planning_model: str,
):
    """Test TaskPlanner's ability to handle batch processing scenarios"""

    context = ProcessingContext()

    # Create agent with objective that requires processing many items
    batch_agent = Agent(
        name="Multi-Site AI Trend Analyzer",
        objective=objective,
        provider=provider,
        model=model,
        planning_model=planning_model,
        enable_analysis_phase=True,
        enable_data_contracts_phase=True,
        tools=[
            GoogleSearchTool(),
            BrowserTool(),
        ],
        output_type="json",
        max_subtasks=30,  # Allow more subtasks for batch processing
        max_subtask_iterations=15,  # More iterations for batch work
    )

    print(f"🚀 Starting agent: {batch_agent.name}")
    print("📋 Objective: Analyze 20 tech sites for AI trends")
    print(f"🔧 Planning Model: {planning_model}")
    print(f"⚙️  Execution Model: {model}\n")

    # Track planning details
    subtask_count = 0
    batch_subtasks = []
    plan_generated = False

    print("=" * 60)
    print("PLANNING & EXECUTION LOG")
    print("=" * 60)

    async for item in batch_agent.execute(context):
        if isinstance(item, TaskUpdate):
            # Monitor task updates to see batch processing in action
            if hasattr(item, "task") and item.task and not plan_generated:
                plan_generated = True
                print("\n📊 PLAN GENERATED:")
                print(f"   Total subtasks: {len(item.task.subtasks)}")

                # Check for batch processing
                for i, subtask in enumerate(item.task.subtasks):
                    if (
                        hasattr(subtask, "batch_processing")
                        and subtask.batch_processing
                    ):
                        if subtask.batch_processing.get("enabled", False):
                            batch_subtasks.append(i)
                            print(f"\n   🔄 Subtask {i+1} - BATCH PROCESSING ENABLED:")
                            print(f"      Content: {subtask.content[:100]}...")
                            print(
                                f"      Batch size: {subtask.batch_processing.get('batch_size', 'N/A')}"
                            )
                            print(
                                f"      Items: {subtask.batch_processing.get('start_index', 0)}-{subtask.batch_processing.get('end_index', 'N/A')}"
                            )
                            print(f"      Output: {subtask.output_file}")

                if batch_subtasks:
                    print(
                        f"\n✅ Batch processing detected in {len(batch_subtasks)} subtasks!"
                    )
                else:
                    print(
                        "\n⚠️  No batch processing detected (might process all at once)"
                    )

        elif isinstance(item, Chunk):
            # Show execution progress
            print(item.content, end="", flush=True)

    # Get final results
    final_results = batch_agent.get_results()

    print("\n\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)

    # Analyze the results
    workspace_path = Path(context.workspace_dir)
    json_files = list(workspace_path.glob("*.json"))
    jsonl_files = list(workspace_path.glob("*.jsonl"))

    print(f"\n📁 Workspace: {context.workspace_dir}")
    print(f"📄 JSON files created: {len(json_files)}")
    print(f"📄 JSONL files created: {len(jsonl_files)}")

    # Check if batch processing files were created
    batch_files = [f for f in jsonl_files if "batch" in f.name.lower()]
    if batch_files:
        print("\n🔄 Batch processing files found:")
        for bf in batch_files:
            size = bf.stat().st_size
            print(f"   - {bf.name} ({size:,} bytes)")

            # Count entries in JSONL
            with open(bf, "r") as f:
                entries = sum(1 for _ in f)
            print(f"     Entries: {entries}")

    # Display final results summary
    if final_results:
        print("\n📊 Final Results Preview:")
        # Try to parse as JSON
        try:
            result_data = json.loads(final_results)
            if isinstance(result_data, dict):
                # Show structure
                print(f"   Result keys: {list(result_data.keys())}")
                if "trends" in result_data:
                    print(f"   Trends found: {len(result_data['trends'])}")
                if "articles" in result_data:
                    print(f"   Articles collected: {len(result_data['articles'])}")
        except Exception:  # noqa: S110
            # If not JSON, show text preview
            print(f"   {final_results[:200]}...")

    print("\n✨ Task Planner Batch Processing Test Complete!")

    # Return results for further analysis
    return {
        "workspace": context.workspace_dir,
        "batch_subtasks_count": len(batch_subtasks),
        "total_subtasks": subtask_count,
        "final_results": final_results,
    }


async def run_comparison_test():
    """Run the test with different configurations to compare batch vs non-batch"""

    print("🧪 TASK PLANNER BATCH PROCESSING TEST")
    print("=" * 80)
    print("This test analyzes 20 websites to trigger batch processing logic\n")

    # Test configuration
    provider = await get_provider(Provider.OpenAI)
    model = "gpt-4o-mini"
    planning_model = "gpt-4o-mini"

    # Run the test
    results = await test_task_planner_batch_processing(
        provider=provider, model=model, planning_model=planning_model
    )

    # Summary
    print("\n" + "=" * 80)
    print("🏁 TEST COMPLETE")
    print("=" * 80)

    if results["batch_subtasks_count"] > 0:
        print(
            f"✅ SUCCESS: TaskPlanner created {results['batch_subtasks_count']} batch processing subtasks"
        )
        print(
            "   The planner successfully detected the need to process items in batches!"
        )
    else:
        print("⚠️  WARNING: No batch processing was detected")
        print("   The planner might have created a single task to process all items")

    print(f"\n💡 TIP: Check {results['workspace']} for detailed execution artifacts")


async def main():
    async with ResourceScope():
        await run_comparison_test()
        # await test_task_planner_batch_processing(
        #     provider=await get_provider(Provider.Anthropic),
        #     model="claude-3-5-sonnet-20241022",
        #     planning_model="claude-3-5-sonnet-20241022",
        # )


if __name__ == "__main__":
    # Run the batch processing test
    asyncio.run(main())
