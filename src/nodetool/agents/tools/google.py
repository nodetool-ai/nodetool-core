from typing import Dict, Any, List
from google.genai import Client
from google.genai.client import AsyncClient
from google.genai.types import (
    Tool as GenAITool,
    GenerateContentConfig,
    GoogleSearch,
)
from nodetool.agents.tools.base import Tool
from nodetool.common.environment import Environment
from nodetool.workflows.processing_context import ProcessingContext


def get_genai_client() -> AsyncClient:
    env = Environment.get_environment()
    api_key = env.get("GEMINI_API_KEY")
    assert api_key, "GEMINI_API_KEY is not set"
    return Client(api_key=api_key).aio


class GoogleGroundedSearchTool(Tool):
    """
    🔍 Google Grounded Search Tool - Searches the web using Gemini API's grounding capabilities

    This tool uses Google's Gemini API to perform web searches and return structured results
    with source information. Requires a Gemini API key.
    """

    name = "google_grounded_search"
    description = "Search the web using Google's Gemini API with grounding capabilities"

    def __init__(self, workspace_dir: str):
        super().__init__(workspace_dir)
        self.client = get_genai_client()
        self.input_schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to execute",
                }
            },
            "required": ["query"],
        }

    async def process(
        self, context: ProcessingContext, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a web search using Gemini API with grounding.

        Args:
            context: The processing context
            params: The search parameters including the query

        Returns:
            Dict containing the search results and sources
        """
        query = params.get("query")
        if not query:
            raise ValueError("Search query is required")

        # Configure Google Search as a tool
        google_search_tool = GenAITool(google_search=GoogleSearch())

        # Generate content with search grounding
        response = await self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=query,
            config=GenerateContentConfig(
                tools=[google_search_tool],
                response_modalities=["TEXT"],
            ),
        )

        # Extract search results and source information
        results = []
        sources = []

        # Check if we have a valid response with candidates
        if not response or not response.candidates:
            return {
                "query": query,
                "results": "No results found",
                "sources": [],
                "status": "error",
                "error": "No response received from Gemini API",
            }

        candidate = response.candidates[0]
        if not candidate or not candidate.content:
            return {
                "query": query,
                "results": "No results found",
                "sources": [],
                "status": "error",
                "error": "Invalid response format from Gemini API",
            }

        # Get the main response text
        if candidate.content.parts:
            for part in candidate.content.parts:
                if part.text:
                    results.append(part.text)

        # Extract source information if available
        if (
            candidate.grounding_metadata
            and candidate.grounding_metadata.grounding_chunks
        ):
            # Extract sources from grounding chunks
            chunks = candidate.grounding_metadata.grounding_chunks
            for chunk in chunks:
                if hasattr(chunk, "web") and chunk.web:
                    source = {
                        "title": (
                            chunk.web.title
                            if hasattr(chunk.web, "title")
                            else "Unknown Source"
                        ),
                        "url": chunk.web.uri if hasattr(chunk.web, "uri") else None,
                    }
                    if source not in sources and source["url"]:
                        sources.append(source)

        # Extract grounding supports if available
        grounding_supports = []
        if (
            candidate.grounding_metadata
            and candidate.grounding_metadata.grounding_supports
        ):
            supports = candidate.grounding_metadata.grounding_supports
            for support in supports:
                if support.segment:
                    support_info = {
                        "text": support.segment.text,
                        "start_index": support.segment.start_index,
                        "end_index": support.segment.end_index,
                        "chunk_indices": support.grounding_chunk_indices,
                        "confidence_scores": support.confidence_scores,
                    }
                    grounding_supports.append(support_info)

        # Format the results
        formatted_results = {
            "query": query,
            "results": results,
            "sources": sources,
            "grounding_supports": grounding_supports,
            "status": "success",
        }

        return formatted_results
