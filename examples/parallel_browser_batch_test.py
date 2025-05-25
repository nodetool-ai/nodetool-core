#!/usr/bin/env python3
"""
Parallel Browser Batch Test - Testing Improved Batch Processing

This script specifically tests the improved batch processing that handles
multiple parallel tool calls with large results. It demonstrates:
- How the system handles 4+ browser calls made in parallel
- Automatic chunking of tool calls to prevent context overflow
- Immediate result saving to batch files
- Summary messages in history instead of full content

Usage:
    python parallel_browser_batch_test.py
"""

import asyncio
import json
from typing import List, Dict, Any
from pathlib import Path
from nodetool.chat.providers import get_provider
from nodetool.agents.tools import BrowserTool
from nodetool.chat.providers.base import ChatProvider
from nodetool.metadata.types import Provider, Task, SubTask
from nodetool.agents.sub_task_context import SubTaskContext
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk, TaskUpdate, TaskUpdateEvent
from nodetool.agents.tools.workspace_tools import WriteFileTool, ReadFileTool


# URLs that will trigger parallel browser calls
PARALLEL_TEST_URLS = [
    # Group 1: News sites (likely to be called together)
    ["https://techcrunch.com", "https://www.theverge.com", "https://arstechnica.com", "https://www.wired.com"],
    
    # Group 2: AI/ML sites (another parallel batch)
    ["https://openai.com/blog", "https://www.anthropic.com", "https://huggingface.co", "https://pytorch.org"],
    
    # Group 3: Developer blogs (third parallel batch)
    ["https://github.blog", "https://stackoverflow.blog", "https://dev.to", "https://medium.com/tag/technology"],
]


async def create_test_instructions(workspace_dir: str) -> str:
    """Create test instructions that will trigger parallel browser calls."""
    instructions_file = Path(workspace_dir) / "parallel_test_instructions.json"
    
    instructions = {
        "task": "Compare content across multiple websites in parallel",
        "groups": [
            {
                "name": "Tech News Comparison",
                "instruction": "Navigate to ALL of these sites AT THE SAME TIME and extract their latest headlines",
                "urls": PARALLEL_TEST_URLS[0]
            },
            {
                "name": "AI/ML Blog Analysis", 
                "instruction": "Visit ALL these AI/ML sites SIMULTANEOUSLY and find their most recent posts",
                "urls": PARALLEL_TEST_URLS[1]
            },
            {
                "name": "Developer Resource Check",
                "instruction": "Access ALL developer blogs IN PARALLEL and identify trending topics",
                "urls": PARALLEL_TEST_URLS[2]
            }
        ],
        "important": "For each group, you MUST call the browser tool for ALL URLs in that group AT THE SAME TIME (in parallel) to test the system's ability to handle multiple concurrent requests."
    }
    
    with open(instructions_file, 'w') as f:
        json.dump(instructions, f, indent=2)
    
    print(f"Created parallel test instructions with {sum(len(g) for g in PARALLEL_TEST_URLS)} URLs in {len(PARALLEL_TEST_URLS)} groups")
    return "parallel_test_instructions.json"


async def test_parallel_browser_batch(
    provider: ChatProvider,
    model: str
):
    """Test the improved batch processing with parallel browser calls."""
    
    # 1. Set up workspace
    context = ProcessingContext()
    workspace_dir = context.workspace_dir
    
    # 2. Create test instructions
    instructions_file = await create_test_instructions(workspace_dir)
    
    # 3. Set up tools
    tools = [
        BrowserTool(),
        WriteFileTool(),
        ReadFileTool(),
    ]
    
    # 4. Create task
    task = Task(
        title="Parallel Browser Call Batch Processing Test",
        description="Test handling of multiple parallel browser calls with large results",
        subtasks=[],
    )
    
    # 5. Create subtask that will trigger parallel calls
    parallel_subtask = SubTask(
        content=f"""
        Read the instructions from '{instructions_file}' and follow them EXACTLY.
        
        CRITICAL: For each group of URLs, you MUST:
        1. Call the browser tool for ALL URLs in that group IN A SINGLE RESPONSE
        2. This means making 4 browser tool calls at once for each group
        3. Extract and save key information from each site
        
        After visiting all sites, create a comprehensive comparison report that includes:
        - Which sites loaded successfully
        - Key headlines or posts from each site
        - Common themes across sites in each group
        - Any errors encountered
        
        Save results progressively as you process each group.
        """,
        output_file="parallel_browser_results.jsonl",
        input_files=[instructions_file],
        output_type="jsonl",
        output_schema=json.dumps({
            "type": "object",
            "properties": {
                "group_name": {"type": "string"},
                "urls_processed": {"type": "integer"},
                "parallel_calls": {"type": "integer"},
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "success": {"type": "boolean"},
                            "title": {"type": "string"},
                            "headline": {"type": "string"},
                            "error": {"type": "string"}
                        }
                    }
                },
                "common_themes": {"type": "array", "items": {"type": "string"}},
                "processing_time_ms": {"type": "integer"}
            }
        }),
        batch_processing={
            "enabled": True,
            "batch_size": 4,  # Expecting 4 parallel calls
            "start_index": 0,
            "end_index": 12,  # Total URLs
            "total_items": 12
        }
    )
    
    # 6. Execute with monitoring
    print("\n🚀 Starting Parallel Browser Batch Processing Test")
    print(f"📊 Expecting {len(PARALLEL_TEST_URLS)} groups with 4 parallel calls each")
    print(f"🔧 Model: {model}")
    print(f"⚠️  Watch for: 'Batch processing N tool calls to prevent context overflow'\n")
    
    subtask_context = SubTaskContext(
        task=task,
        subtask=parallel_subtask,
        processing_context=ProcessingContext(workspace_dir=workspace_dir),
        tools=tools,
        model=model,
        provider=provider,
        max_iterations=15,
        max_token_limit=8000,  # Lower limit to trigger optimization sooner
    )
    
    # Track metrics
    tool_call_batches = []
    current_batch = []
    browser_calls_total = 0
    parallel_batches_detected = 0
    
    async for event in subtask_context.execute():
        if isinstance(event, Chunk):
            print(event.content, end="", flush=True)
        elif isinstance(event, TaskUpdate):
            if event.event == TaskUpdateEvent.ENTERED_CONCLUSION_STAGE:
                print(f"\n⚠️  CONTEXT LIMIT: Entering conclusion stage")
        elif hasattr(event, 'name'):  # ToolCall
            current_batch.append(event.name)
            if event.name == "browser":
                browser_calls_total += 1
                print(f"\n🌐 Browser call #{browser_calls_total}: {event.args.get('url', 'unknown')}")
            
            # Check if we're seeing the end of a batch (heuristic)
            if len(current_batch) >= 4 or (current_batch and event.name != "browser"):
                if current_batch.count("browser") >= 3:
                    parallel_batches_detected += 1
                    print(f"\n📦 PARALLEL BATCH DETECTED: {current_batch.count('browser')} browser calls")
                tool_call_batches.append(current_batch)
                current_batch = []
    
    # Add any remaining batch
    if current_batch:
        tool_call_batches.append(current_batch)
    
    # 7. Analyze results
    print(f"\n\n{'='*60}")
    print("📊 PARALLEL PROCESSING ANALYSIS")
    print(f"{'='*60}\n")
    
    print(f"Browser Calls Total: {browser_calls_total}")
    print(f"Parallel Batches Detected: {parallel_batches_detected}")
    print(f"Tool Call Batches: {len(tool_call_batches)}")
    
    # Check if batch processing was triggered
    if any(len(batch) >= 3 for batch in tool_call_batches):
        print("\n✅ SUCCESS: Parallel batch processing was triggered!")
        print("   The system handled multiple browser calls intelligently")
    else:
        print("\n⚠️  WARNING: Parallel batches were not detected as expected")
    
    # Check output file
    output_path = Path(workspace_dir) / parallel_subtask.output_file
    if output_path.exists():
        with open(output_path, 'r') as f:
            lines = f.readlines()
        print(f"\nOutput File Results: {len(lines)} entries")
        
        # Analyze the results
        total_urls = 0
        successful_loads = 0
        for line in lines:
            try:
                entry = json.loads(line)
                total_urls += entry.get("urls_processed", 0)
                successful_loads += sum(1 for r in entry.get("results", []) if r.get("success", False))
            except:
                pass
        
        print(f"Total URLs Processed: {total_urls}")
        print(f"Successful Loads: {successful_loads}")
        print(f"Success Rate: {successful_loads/total_urls*100:.1f}%" if total_urls > 0 else "N/A")
    
    # Check for context optimizations
    print(f"\nContext Management:")
    print(f"  - Max iterations: {subtask_context.max_iterations}")
    print(f"  - Iterations used: {subtask_context.iterations}")
    print(f"  - Token limit: {subtask_context.max_token_limit}")
    print(f"  - Batch processing enabled: {subtask_context.is_batch_processing}")
    
    print(f"\n📁 Workspace: {workspace_dir}")


async def main():
    """Run the parallel browser batch test."""
    
    # Test configurations focusing on parallel processing
    test_configs = [
        # OpenAI - Test with GPT-4
        {
            "provider": Provider.OpenAI,
            "model": "gpt-4o-mini",
            "description": "Testing parallel browser calls with GPT-4 Mini"
        },
        # Can add other providers as needed
    ]
    
    for config in test_configs:
        print(f"\n{'#'*80}")
        print(f"# {config['description']}")
        print(f"# Provider: {config['provider'].value}, Model: {config['model']}")
        print(f"{'#'*80}")
        
        await test_parallel_browser_batch(
            provider=get_provider(config["provider"]),
            model=config["model"]
        )
        
        if len(test_configs) > 1:
            print("\n⏳ Waiting before next test...")
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())