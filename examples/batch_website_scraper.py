#!/usr/bin/env python3
"""
Batch Website Scraper - Testing SubTaskContext Batch Processing

This script demonstrates the batch processing capabilities of SubTaskContext
by scraping multiple websites in batches. It showcases:
- Automatic batch detection and configuration
- Progressive result writing (JSONL format)
- Context window optimization for multiple browser calls
- Memory-efficient processing of large lists

The task scrapes a list of websites to extract key information like titles,
descriptions, and main content, processing them in configurable batches to
minimize context window usage.

Usage:
    python batch_website_scraper.py
"""

import asyncio
import json
from pathlib import Path
from nodetool.chat.providers import get_provider
from nodetool.agents.tools import BrowserTool
from nodetool.chat.providers.base import ChatProvider
from nodetool.metadata.types import Provider, Task, SubTask
from nodetool.agents.sub_task_context import SubTaskContext
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk, TaskUpdate, TaskUpdateEvent


# List of websites to scrape - demonstrating batch processing
WEBSITES_TO_SCRAPE = [
    # Tech news sites
    "https://techcrunch.com",
    "https://www.theverge.com",
    "https://arstechnica.com",
    "https://www.wired.com",
    "https://hackernews.com",
    # AI/ML focused sites
    "https://openai.com/blog",
    "https://www.anthropic.com/news",
    "https://huggingface.co/blog",
    "https://pytorch.org/blog",
    # General tech company blogs
    "https://blog.google",
    "https://engineering.fb.com",
    "https://netflixtechblog.com",
    "https://aws.amazon.com/blogs/aws",
    # Developer resources
    "https://github.blog",
    "https://stackoverflow.blog",
    "https://dev.to",
    "https://medium.com/tag/technology",
]


async def create_website_list_file(workspace_dir: str) -> str:
    """Create a file containing the list of websites to scrape."""
    websites_file = Path(workspace_dir) / "websites_to_scrape.json"

    # Structure the data to make batch processing clear
    website_data = {
        "total_sites": len(WEBSITES_TO_SCRAPE),
        "sites": [
            {"index": i, "url": url, "category": _categorize_site(url)}
            for i, url in enumerate(WEBSITES_TO_SCRAPE)
        ],
    }

    with open(websites_file, "w") as f:
        json.dump(website_data, f, indent=2)

    print(f"Created website list file with {len(WEBSITES_TO_SCRAPE)} sites")
    return "websites_to_scrape.json"


def _categorize_site(url: str) -> str:
    """Simple categorization based on URL."""
    if "blog" in url or "engineering" in url:
        return "blog"
    elif "news" in url or "techcrunch" in url or "verge" in url:
        return "news"
    elif "ai" in url or "anthropic" in url or "openai" in url:
        return "ai"
    else:
        return "general"


async def test_batch_website_scraper(
    provider: ChatProvider, model: str, batch_size: int = 5
):
    """Test batch processing of website scraping with SubTaskContext."""

    # 1. Set up workspace directory
    context = ProcessingContext()
    workspace_dir = context.workspace_dir

    # 2. Create the website list file
    websites_file = await create_website_list_file(workspace_dir)

    # 3. Set up tools - browser for scraping, file tools for I/O
    tools = [
        BrowserTool(),
    ]

    # 4. Create the main task
    task = Task(
        title="Batch Website Information Extraction",
        description="Extract key information from multiple websites using batch processing to minimize context window usage",
        subtasks=[],
    )

    # 5. Create subtasks that will trigger batch processing
    subtasks = []

    # First subtask: Process websites in batches
    # This demonstrates automatic batch detection when processing lists
    batch_subtask = SubTask(
        content=f"""
        Read the website list from '{websites_file}' and extract key information from each site.
        
        For each website:
        1. Navigate to the URL using the browser tool
        2. Extract:
           - Page title
           - Meta description (if available)
           - Main heading (h1)
           - First 200 characters of main content
           - Number of links on the page
           - Any error messages if the site fails to load
        
        Process the sites in batches of {batch_size} to optimize memory usage.
        Write results progressively to the output file as you process each batch.
        
        If a site fails to load or times out, record the error and continue with the next site.
        Include timing information for each site (how long it took to scrape).
        """,
        output_file="website_scraping_results.jsonl",
        input_files=[websites_file],
        output_type="jsonl",
        output_schema=json.dumps(
            {
                "type": "object",
                "properties": {
                    "batch_results": {
                        "type": "array",
                        "description": "Results from scraping websites in this batch",
                        "items": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string"},
                                "index": {"type": "integer"},
                                "category": {"type": "string"},
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "main_heading": {"type": "string"},
                                "content_preview": {"type": "string"},
                                "link_count": {"type": "integer"},
                                "scrape_time_ms": {"type": "integer"},
                                "success": {"type": "boolean"},
                                "error": {"type": "string"},
                                "timestamp": {"type": "string"},
                            },
                            "required": ["url", "index", "success", "timestamp"],
                        },
                    }
                },
            }
        ),
        # Enable batch processing
        batch_processing={
            "enabled": True,
            "batch_size": batch_size,
            "start_index": 0,
            "end_index": len(WEBSITES_TO_SCRAPE),
            "total_items": len(WEBSITES_TO_SCRAPE),
        },
    )
    subtasks.append(batch_subtask)

    # Add subtasks to main task
    task.subtasks = subtasks

    # 6. Execute subtasks with batch processing
    print(f"\n🚀 Starting batch website scraping with {len(WEBSITES_TO_SCRAPE)} sites")
    print(f"📦 Batch size: {batch_size}")
    print(f"🔧 Model: {model}\n")

    for i, subtask in enumerate(subtasks):
        print(f"\n{'='*60}")
        print(f"Executing Subtask {i+1}: {subtask.content[:100]}...")
        print(
            f"Batch processing: {'ENABLED' if subtask.batch_processing.get('enabled', False) else 'DISABLED'}"
        )
        print(f"{'='*60}\n")

        # Create SubTaskContext - it will automatically detect batch processing config
        subtask_context = SubTaskContext(
            task=task,
            subtask=subtask,
            processing_context=ProcessingContext(workspace_dir=workspace_dir),
            tools=tools,
            model=model,
            provider=provider,
            max_iterations=25,  # More iterations for batch processing
        )

        # Track execution metrics
        start_time = asyncio.get_event_loop().time()
        chunk_count = 0
        tool_calls = []

        # Execute the subtask
        async for event in subtask_context.execute():
            if isinstance(event, Chunk):
                chunk_count += 1
                print(event.content, end="", flush=True)
            elif isinstance(event, TaskUpdate):
                if event.event == TaskUpdateEvent.SUBTASK_STARTED:
                    print(f"\n🟢 Subtask started")
                elif event.event == TaskUpdateEvent.ENTERED_CONCLUSION_STAGE:
                    print(f"\n⚠️  Entering conclusion stage (context limit approaching)")
                elif event.event == TaskUpdateEvent.SUBTASK_COMPLETED:
                    print(f"\n✅ Subtask completed")
                elif event.event == TaskUpdateEvent.MAX_ITERATIONS_REACHED:
                    print(f"\n⚠️  Max iterations reached")
            elif hasattr(event, "name"):  # ToolCall
                tool_calls.append(event.name)
                if event.name == "browser":
                    print(f"\n🌐 Browser tool called")

        end_time = asyncio.get_event_loop().time()
        execution_time = end_time - start_time

        # Print execution statistics
        print(f"\n\n📊 Subtask Execution Statistics:")
        print(f"   - Execution time: {execution_time:.2f} seconds")
        print(f"   - Text chunks: {chunk_count}")
        print(f"   - Tool calls: {len(tool_calls)}")
        print(f"   - Browser calls: {tool_calls.count('browser')}")
        print(
            f"   - Iterations used: {subtask_context.iterations}/{subtask_context.max_iterations}"
        )

        # Check output file
        output_path = Path(workspace_dir) / subtask.output_file
        if output_path.exists():
            file_size = output_path.stat().st_size
            print(f"   - Output file size: {file_size:,} bytes")

            # For JSONL, count the lines
            if subtask.output_type == "jsonl":
                with open(output_path, "r") as f:
                    line_count = sum(1 for _ in f)
                print(f"   - JSONL entries: {line_count}")

    # 7. Display final results
    print(f"\n\n{'='*60}")
    print("📋 FINAL RESULTS")
    print(f"{'='*60}\n")

    # Check the scraping results file
    results_path = Path(workspace_dir) / "website_scraping_results.jsonl"
    if results_path.exists():
        file_size = results_path.stat().st_size
        print(f"✅ Scraping completed successfully!")
        print(f"📄 Results file: website_scraping_results.jsonl")
        print(f"📦 File size: {file_size:,} bytes")

        # Count the entries
        with open(results_path, "r") as f:
            line_count = sum(1 for _ in f)
        print(f"📊 Total entries: {line_count}")

        # Show a sample of the data structure
        print(f"\n🔍 Sample entry structure:")
        with open(results_path, "r") as f:
            first_line = f.readline().strip()
            if first_line:
                try:
                    sample_data = json.loads(first_line)
                    if "batch_results" in sample_data and sample_data["batch_results"]:
                        first_result = sample_data["batch_results"][0]
                        print(f"   - URL: {first_result.get('url', 'N/A')}")
                        print(f"   - Success: {first_result.get('success', 'N/A')}")
                        print(f"   - Title: {first_result.get('title', 'N/A')[:50]}...")
                        if first_result.get("error"):
                            print(f"   - Error: {first_result.get('error', 'N/A')}")
                except json.JSONDecodeError:
                    print("   - Unable to parse sample data")
    else:
        print("❌ No results file found. Scraping may have failed.")

    print(f"\n✨ Batch website scraping complete!")
    print(f"📁 Check workspace at: {workspace_dir}")
    print(f"📊 Results available in: website_scraping_results.jsonl")


async def main():
    """Run the batch website scraper example."""

    # You can test with different providers and models
    test_configs = [
        # OpenAI - Good for general web scraping
        {"provider": Provider.OpenAI, "model": "gpt-4o-mini", "batch_size": 4},
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
        print(f"# Testing with {config['provider'].value} - {config['model']}")
        print(f"{'#'*80}")

        await test_batch_website_scraper(
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
