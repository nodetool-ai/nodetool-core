#!/usr/bin/env python3
"""
Test script for the updated browser tools with CSS path generation.
"""

import asyncio
import json
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.tools.browser_tools import DOMExamineTool, DOMSearchTool, DOMExtractTool

async def test_updated_browser_tools():
    context = ProcessingContext()
    
    # Test DOMExamineTool with CSS path generation
    print("=== Testing DOMExamineTool ===")
    examine_tool = DOMExamineTool()
    result = await examine_tool.process(context, {
        "url": "https://news.ycombinator.com",
        "selector": "a.storylink",
        "max_depth": 2
    })
    print(json.dumps(result, indent=2))
    
    # Test DOMSearchTool with CSS path generation  
    print("\n=== Testing DOMSearchTool ===")
    search_tool = DOMSearchTool()
    result = await search_tool.process(context, {
        "url": "https://news.ycombinator.com",
        "search_type": "css",
        "query": "a[href*='item?id=']",
        "limit": 3
    })
    print(json.dumps(result, indent=2))
    
    # Test extraction with proper selector
    print("\n=== Testing DOMExtractTool ===")
    extract_tool = DOMExtractTool()
    result = await extract_tool.process(context, {
        "url": "https://news.ycombinator.com",
        "selector": "a[href*='item?id=']",
        "extract_type": "structured"
    })
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(test_updated_browser_tools())