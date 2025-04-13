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
from nodetool.agents.agent import Agent
from nodetool.chat.providers import get_provider
from nodetool.chat.providers.base import ChatProvider
from nodetool.workflows.types import Chunk
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.tools.pdf import ConvertPDFToMarkdownTool
from nodetool.agents.tools.chroma import (
    ChromaHybridSearchTool,
    ChromaMarkdownSplitAndIndexTool,
)
from nodetool.common.chroma_client import get_chroma_client, get_collection


async def test_chromadb_research_agent(provider: ChatProvider, model: str):
    context = ProcessingContext()
    input_files = []

    for paper in os.listdir(os.path.join(os.path.dirname(__file__), "papers")):
        if paper.endswith(".pdf"):
            input_files.append(os.path.join(os.path.dirname(__file__), "papers", paper))

    chroma_client = get_chroma_client()

    try:
        collection = chroma_client.delete_collection("test-papers")
    except Exception:
        pass

    collection = chroma_client.create_collection("test-papers")

    ingestion_agent = Agent(
        name="Document Ingestion Agent",
        objective="""
        Convert each pdf file into a markdown file and index it into ChromaDB using chroma_markdown_split_and_index.
        """,
        provider=provider,
        model=model,
        input_files=input_files,
        enable_analysis_phase=False,
        enable_data_contracts_phase=False,
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

    research_agent = Agent(
        name="Research Agent",
        objective="""
        Explain attention in LLMs. How does it work? What are the different types of attention?
        """,
        provider=provider,
        model=model,
        enable_analysis_phase=False,
        enable_data_contracts_phase=False,
        tools=[
            ChromaHybridSearchTool(
                workspace_dir=str(context.workspace_dir),
                collection=collection,
            ),
        ],
    )

    print("Ingesting documents...")
    async for item in ingestion_agent.execute(context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    result = collection.query(
        query_texts=["attention"],
        n_results=5,
    )
    print("Query result:")
    print(result)

    print("Researching...")
    async for item in research_agent.execute(context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    # print(f"\nWorkspace: {context.workspace_dir}")


if __name__ == "__main__":

    asyncio.run(
        test_chromadb_research_agent(
            provider=get_provider(Provider.OpenAI),
            model="gpt-4o",
        )
    )

    # asyncio.run(
    #     test_chromadb_research_agent(
    #         provider=get_provider(Provider.Gemini),
    #         model="gemini-2.0-flash",
    #     )
    # )
