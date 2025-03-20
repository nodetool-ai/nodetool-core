#!/usr/bin/env python3
"""
Simple test script for a retrieval agent using browser tools.

This script creates a retrieval agent with browser tools and executes a hard-coded
task plan without using the task planner. It demonstrates how to set up an agent
with browser-based retrieval tools and track its execution.
"""

import asyncio

from nodetool.chat.agent import Agent, RETRIEVAL_SYSTEM_PROMPT
from nodetool.chat.providers import get_provider, Chunk
from nodetool.chat.tools.browser import GoogleSearchTool
from nodetool.metadata.types import Provider, Task, TaskPlan, SubTask
from nodetool.chat.workspace_manager import WorkspaceManager
from nodetool.workflows.processing_context import ProcessingContext

GOOGLE_SEARCH_SYSTEM_PROMPT = """
You are a specialized Google search retrieval agent designed to efficiently gather information using available tools.

google_search - Execute precise Google searches with advanced parameters:
   - Use site: filters (e.g., "site:example.com") to narrow results to specific websites
   - Use filetype: filters (e.g., "filetype:pdf") to find specific document types
   - Use exact phrase matching with quotes for precise searches
   - Leverage advanced search parameters like intitle:, inurl:, and intext:
   - Filter by time_period: "past_24h", "past_week", "past_month", "past_year"
   - Specify country and language codes for localized results
   - Control pagination with start parameter
"""


async def main():
    # 1. Set up workspace directory
    workspace_manager = WorkspaceManager()
    workspace_dir = workspace_manager.get_current_directory()
    print(f"Created workspace at: {workspace_dir}")

    # 2. Initialize provider and model
    # provider = get_provider(Provider.Anthropic)
    # model = "claude-3-7-sonnet-20250219"
    provider = get_provider(Provider.OpenAI)
    model = "gpt-4o"

    # 3. Set up tools for retrieval
    tools = [
        GoogleSearchTool(str(workspace_dir)),
    ]

    # 4. Create retrieval agent
    agent = Agent(
        name="Google Search Agent",
        objective="Research information about AI models and save results to files.",
        description="A research agent that retrieves information from Google and saves it to files.",
        provider=provider,
        model=model,
        workspace_dir=str(workspace_dir),
        tools=tools,
        system_prompt=RETRIEVAL_SYSTEM_PROMPT,
        max_steps=50,
        max_subtask_iterations=10,
    )

    # 5. Create a hard-coded task plan with a single task
    task_plan = TaskPlan(
        title="Research AI Models",
        tasks=[
            Task(
                title="Research recent LLM developments",
                agent_name="Research Agent",
                subtasks=[
                    SubTask(
                        content="Search the latest LLM models and their capabilities.",
                        max_tool_calls=10,
                        output_type={
                            "type": "object",
                            "description": "Summary of recent LLM models and their capabilities",
                            "properties": {
                                "models": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "url": {"type": "string"},
                                        },
                                    },
                                    "description": "List of LLM models identified",
                                },
                            },
                        },
                        output_file="/workspace/ai_models_search.yaml",
                        file_dependencies=[],
                    ),
                    # Site-specific search filter
                    SubTask(
                        content="Search for LLM benchmarks on site:huggingface.co and site:paperswithcode.com",
                        max_tool_calls=5,
                        output_type={
                            "type": "array",
                            "description": "Collection of LLM benchmark data from specified sites",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "benchmark_name": {"type": "string"},
                                    "models_evaluated": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "metrics": {"type": "object"},
                                    "source": {"type": "string"},
                                },
                            },
                        },
                        output_file="/workspace/llm_benchmarks_site_filtered.yaml",
                        file_dependencies=[],
                    ),
                    # Filetype filter
                    SubTask(
                        content="Find research papers about 'multimodal AI models' with filetype:pdf",
                        max_tool_calls=5,
                        output_type={
                            "type": "array",
                            "description": "List of research papers on multimodal AI models",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "authors": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "publication_date": {"type": "string"},
                                    "url": {"type": "string"},
                                    "summary": {"type": "string"},
                                },
                            },
                        },
                        output_file="/workspace/multimodal_papers.yaml",
                        file_dependencies=[],
                    ),
                    # Exact phrase matching
                    SubTask(
                        content='Search for "state of AI report 2023" in quotes to find exact matches',
                        max_tool_calls=5,
                        output_type={
                            "type": "object",
                            "description": "Information about the State of AI Report 2023",
                            "properties": {
                                "report_details": {"type": "object"},
                                "url": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                        output_file="/workspace/state_of_ai_exact.yaml",
                        file_dependencies=[],
                    ),
                    # Advanced parameters
                    SubTask(
                        content="Search for intitle:transformer intext:attention papers",
                        max_tool_calls=5,
                        output_type={
                            "type": "array",
                            "description": "Collection of papers about transformers with attention mechanisms",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "authors": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "key_concepts": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "url": {"type": "string"},
                                },
                            },
                        },
                        output_file="/workspace/transformer_advanced_search.yaml",
                        file_dependencies=[],
                    ),
                    # Time period filter
                    SubTask(
                        content="Find news about 'AI regulation' from the past_month",
                        max_tool_calls=5,
                        output_type={
                            "type": "array",
                            "description": "Recent news articles about AI regulation",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "publication": {"type": "string"},
                                    "date": {"type": "string"},
                                    "summary": {"type": "string"},
                                },
                            },
                        },
                        output_file="/workspace/recent_ai_regulation.yaml",
                        file_dependencies=[],
                    ),
                    # Language and country filter
                    SubTask(
                        content="Search for AI conferences in Germany (country:de) in German (lang:de)",
                        max_tool_calls=5,
                        output_type={
                            "type": "array",
                            "description": "List of AI conferences in Germany (in German)",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "location": {"type": "string"},
                                    "date": {"type": "string"},
                                    "topics": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                            },
                        },
                        output_file="/workspace/german_ai_conferences.yaml",
                        file_dependencies=[],
                    ),
                    # Pagination testing
                    SubTask(
                        content="Search for 'open source LLMs' and get results from second page (start:10)",
                        max_tool_calls=5,
                        output_type={
                            "type": "array",
                            "description": "Information about open source LLM projects (from page 2 results)",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "model_name": {"type": "string"},
                                    "repository": {"type": "string"},
                                    "license": {"type": "string"},
                                    "parameters": {"type": "string"},
                                },
                            },
                        },
                        output_file="/workspace/open_source_llms_page2.yaml",
                        file_dependencies=[],
                    ),
                    # Combined filters test
                    SubTask(
                        content="Find academic papers about 'transformer architecture' from the past_year on site:arxiv.org with filetype:pdf",
                        max_tool_calls=5,
                        output_type={
                            "type": "array",
                            "description": "Recent academic papers on transformer architecture from arXiv",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "authors": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "arxiv_id": {"type": "string"},
                                    "publication_date": {"type": "string"},
                                    "abstract": {"type": "string"},
                                },
                            },
                        },
                        output_file="/workspace/transformer_papers_combined.yaml",
                        file_dependencies=[],
                    ),
                ],
            )
        ],
    )

    # 6. Execute the task and print results
    print(f"\nExecuting task plan: {task_plan.title}")

    # Get the first task (we only have one in this example)
    task = task_plan.tasks[0]
    print(f"\nExecuting task: {task.title}")

    # Execute the task
    async for item in agent.execute_task(task, ProcessingContext()):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    print(f"\nWorkspace: {workspace_dir}")


if __name__ == "__main__":
    asyncio.run(main())
