#!/usr/bin/env python3
"""
Script for a pdf research agent using ChromaDB with TaskPlanner.

This script creates a research agent that works with a ChromaDB collection of academic papers.
It uses a TaskPlanner to automatically generate and execute a task plan
based on the research objective, allowing for comprehensive research on the pdf.

"""

import asyncio
import os

import pymupdf
import pymupdf4llm
from nodetool.workflows.types import Chunk

from nodetool.agents.agent import Agent
from nodetool.agents.tools.chroma_tools import (
    ChromaHybridSearchTool,
    ChromaIndexTool,
    ChromaMarkdownSplitAndIndexTool,
)
from nodetool.agents.tools.pdf_tools import ConvertPDFToMarkdownTool
from nodetool.integrations.vectorstores.chroma.async_chroma_client import (
    get_async_chroma_client,
)
from nodetool.metadata.types import Provider
from nodetool.providers import get_provider
from nodetool.providers.base import BaseProvider
from nodetool.runtime.resources import ResourceScope
from nodetool.workflows.processing_context import ProcessingContext


async def test_chromadb_research_agent(provider: BaseProvider, model: str):
    context = ProcessingContext()
    input_files = []

    for paper in os.listdir(os.path.join(os.path.dirname(__file__), "papers")):
        if paper.endswith(".pdf"):
            doc = pymupdf.open(os.path.join(os.path.dirname(__file__), "papers", paper))
            md_text = pymupdf4llm.to_markdown(doc)
            input_files.append(md_text)
    input_data = "\n".join(input_files)

    chroma_client = await get_async_chroma_client()

    try:
        await chroma_client.delete_collection("test-papers")
    except Exception:
        pass

    collection = await chroma_client.create_collection("test-papers")
    objective = f"""
    Index the following markdown content into ChromaDB using chroma_index.
    ----------------------------------------
    {input_data}
    """

    ingestion_agent = Agent(
        name="Document Ingestion Agent",
        objective=objective,
        provider=provider,
        model=model,
        tools=[
            ChromaIndexTool(collection=collection),
        ],
    )

    research_agent = Agent(
        name="Research Agent",
        objective="""
        Explain attention in LLMs.
        How does it work?
        What are the different types of attention?
        """,
        provider=provider,
        model=model,
        tools=[
            ChromaHybridSearchTool(collection=collection),
        ],
    )

    print("Ingesting documents...")
    async for item in ingestion_agent.execute(context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    if ingestion_agent.results:
        print(ingestion_agent.results)

    print("Researching...")
    async for item in research_agent.execute(context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    if research_agent.results:
        print(research_agent.results)

    print(f"\nWorkspace: {context.workspace_dir}")


async def main():
    async with ResourceScope():
        await test_chromadb_research_agent(
            provider=await get_provider(Provider.HuggingFaceCerebras),
            model="openai/gpt-oss-120b",
        )


if __name__ == "__main__":
    asyncio.run(main())
