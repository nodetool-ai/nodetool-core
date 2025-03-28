#!/usr/bin/env python3
"""
Script for a pdf research agent using ChromaDB with TaskPlanner.

This script creates a research agent that works with a ChromaDB collection of academic papers.
It uses a TaskPlanner to automatically generate and execute a task plan
based on the research objective, allowing for comprehensive research on the pdf.
"""

import os
import shutil
import asyncio
import chromadb
from nodetool.chat.agent import Agent
from nodetool.chat.multi_agent import MultiAgentCoordinator
from nodetool.chat.providers import get_provider
from nodetool.chat.providers.base import Chunk
from nodetool.metadata.types import Provider
from nodetool.chat.workspace_manager import WorkspaceManager
from nodetool.chat.task_planner import TaskPlanner
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.chat.tools.pdf import ConvertPDFToMarkdownTool
from nodetool.chat.tools.chroma import (
    ChromaHybridSearchTool,
    ChromaMarkdownSplitAndIndexTool,
)
from nodetool.chat.tools.browser import DownloadFileTool, WebFetchTool
from nodetool.common.chroma_client import get_collection


async def main():
    context = ProcessingContext()

    chroma_client = chromadb.PersistentClient(
        os.path.join(context.workspace_dir, "chroma")
    )
    chroma_client.create_collection("papers")

    # 2. Initialize provider and model
    # provider = get_provider(Provider.Ollama)
    # model = "qwen2.5:7b"
    # model = "llama3.2:3b"
    provider = get_provider(Provider.OpenAI)
    model = "gpt-4o"
    # provider = get_provider(Provider.Anthropic)
    # model = "claude-3-7-sonnet-20250219"

    input_files = []

    for paper in os.listdir(os.path.join(os.path.dirname(__file__), "papers")):
        shutil.copy(
            os.path.join(os.path.dirname(__file__), "papers", paper),
            os.path.join(context.workspace_dir, paper),
        )
        input_files.append(os.path.join(context.workspace_dir, paper))

    collection = get_collection("papers")

    download_agent = Agent(
        name="Download Agent",
        objective="""
        1. Fetch the following file: https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Agents-Papers/refs/heads/main/README.md
        2. Identify top 5 papers from the file.
        3. Fetch the websites of the papers.
        4. Download the papers from the websites.
        """,
        provider=provider,
        model=model,
        tools=[
            WebFetchTool(
                workspace_dir=str(context.workspace_dir),
            ),
            DownloadFileTool(
                workspace_dir=str(context.workspace_dir),
            ),
        ],
    )

    # Create ingestion agent
    ingestion_agent = Agent(
        name="Document Ingestion Agent",
        objective="""
        Convert each pdf file into a markdown file.
        Index each markdown document into ChromaDB using chroma_markdown_split_and_index.
        """,
        provider=provider,
        model=model,
        input_files=download_agent.get_results(),
        tools=[
            ConvertPDFToMarkdownTool(
                workspace_dir=str(context.workspace_dir),
            ),
            ChromaMarkdownSplitAndIndexTool(
                workspace_dir=str(context.workspace_dir),
                collection=collection,
            ),
        ],
    )

    # 4. Create research agent
    research_agent = Agent(
        name="Research Agent",
        objective="""
        Research the topics in the given chromadb collection.
        """,
        provider=provider,
        model=model,
        tools=[
            ChromaHybridSearchTool(
                workspace_dir=str(workspace_dir),
                collection=collection,
            ),
        ],
    )

    processing_context = ProcessingContext()

    async for item in download_agent.execute(processing_context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    # async for item in ingestion_agent.execute(processing_context):
    #     if isinstance(item, Chunk):
    #         print(item.content, end="", flush=True)

    # async for item in research_agent.execute(processing_context):
    #     if isinstance(item, Chunk):
    #         print(item.content, end="", flush=True)

    print(f"\nWorkspace: {context.workspace_dir}")


if __name__ == "__main__":
    asyncio.run(main())
