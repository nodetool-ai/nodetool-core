#!/usr/bin/env python3
"""
Batch Website Scraper - Testing SubTaskContext Batch Processing

This script demonstrates the batch processing capabilities of SubTaskContext
by scraping multiple websites in batches. It showcases:
- Automatic batch detection and configuration
- Progressive result writing (JSONL format)
- Context window optimization for multiple browser calls
- Memory-efficient processing of large lists

**NEW: DYNAMIC SUBTASK SUPPORT**
While this example creates subtasks upfront, SubTaskContext now supports dynamic
subtask addition. If you use TaskExecutor instead of directly calling SubTaskContext,
agents can use the add_subtask tool to create new tasks during execution. For example,
if a scraper discovers new URLs that need investigation, it can dynamically add them.

The task scrapes a list of websites to extract key information like titles,
descriptions, and main content, processing them in configurable batches to
minimize context window usage.

Usage:
    python batch_website_scraper.py
"""

import asyncio
import json
from nodetool.providers import get_provider
from nodetool.agents.tools import BrowserTool
from nodetool.providers.base import BaseProvider
from nodetool.metadata.types import Provider, Task, SubTask
from nodetool.agents.sub_task_context import SubTaskContext
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk, TaskUpdate, TaskUpdateEvent


# List of websites to scrape - demonstrating batch processing
WEBSITES_TO_SCRAPE = [
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


async def test_batch_website_scraper(
    provider: BaseProvider,
    model: str,
):
    """Test batch processing of website scraping with SubTaskContext."""

    # 1. Set up workspace directory
    context = ProcessingContext()

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
    for i in range(0, len(WEBSITES_TO_SCRAPE)):
        subtask = SubTask(
            content=f"""
            1. ONLY READ item {i} from the input list "websites"
            2. Navigate to the URL using the browser tool
            3. Extract:
            - Page title
            - Meta description (if available)
            - Main heading (h1)
            - First 200 characters of main content
            - Number of links on the page
            - Any error messages if the site fails to load
            4. Return the results in the output schema
            """,
            max_tool_calls=1,
            output_schema=json.dumps(
                {
                    "type": "object",
                    "properties": {
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
                        "additionalProperties": False,
                        "required": ["url", "index", "success", "timestamp"],
                    },
                }
            ),
            input_tasks=["websites"],
        )
        subtasks.append(subtask)

    # Add subtasks to main task
    task.subtasks = subtasks

    context.set("websites", WEBSITES_TO_SCRAPE)

    for i, subtask in enumerate(subtasks):
        print(f"\n{'='*60}")
        print(f"Executing Subtask {i+1}: {subtask.content[:100]}...")
        print(f"{'='*60}\n")

        # Create SubTaskContext - it will automatically detect batch processing config
        subtask_context = SubTaskContext(
            task=task,
            subtask=subtask,
            processing_context=context,
            tools=tools,
            model=model,
            provider=provider,
        )

        # Track execution metrics
        start_time = asyncio.get_event_loop().time()
        tool_calls = []

        # Execute the subtask
        async for event in subtask_context.execute():
            if isinstance(event, Chunk):
                print(event.content, end="", flush=True)
            elif isinstance(event, TaskUpdate):
                if event.event == TaskUpdateEvent.SUBTASK_STARTED:
                    print("\n🟢 Subtask started")
                elif event.event == TaskUpdateEvent.ENTERED_CONCLUSION_STAGE:
                    print("\n⚠️  Entering conclusion stage (context limit approaching)")
                elif event.event == TaskUpdateEvent.SUBTASK_COMPLETED:
                    print("\n✅ Subtask completed")
                elif event.event == TaskUpdateEvent.MAX_ITERATIONS_REACHED:
                    print("\n⚠️  Max iterations reached")
            elif hasattr(event, "name"):  # ToolCall
                tool_calls.append(event.name)

        end_time = asyncio.get_event_loop().time()
        execution_time = end_time - start_time

        # Print execution statistics
        print("\n\n📊 Subtask Execution Statistics:")
        print(f"   - Execution time: {execution_time:.2f} seconds")
        print(f"   - Tool calls: {len(tool_calls)}")

        print("   - Subtask output:")
        print(context.get(subtask.id))


async def main():
    """Run the batch website scraper example."""

    # You can test with different providers and models
    test_configs = [
        # OpenAI - Good for general web scraping
        {"provider": Provider.HuggingFaceCerebras, "model": "openai/gpt-oss-120b"},
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
        )

        # Add a delay between providers to avoid rate limits
        if len(test_configs) > 1:
            print("\n⏳ Waiting 10 seconds before next test...")
            await asyncio.sleep(10)


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
