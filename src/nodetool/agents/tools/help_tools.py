"""
Help Tools for Nodetool Documentation Search

This module contains tool classes for searching and retrieving Nodetool documentation,
examples, and node properties. These tools support both semantic and keyword-based
searches and are designed to work with the agent system.

Tools included:
- SemanticSearchDocumentationTool: Semantic search using embeddings
- KeywordSearchDocumentationTool: Keyword-based search
- SearchExamplesTool: Search workflow examples
- NodePropertiesTool: Get node properties and metadata
"""

from typing import Any

from nodetool.agents.tools.base import Tool
from nodetool.config.environment import Environment
import logging
from nodetool.workflows.base_node import get_node_class
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.type_metadata import ALL_TYPES

log = logging.getLogger(__name__)


class SearchNodesTool(Tool):
    name: str = "search_nodes"
    description: str = (
        """
        Performs keyword search on Nodetool documentation. 
        Use for finding specific node types or features by exact word matches.
        Supply a list of words to search for, including synonyms and related words.
        Returns a list of node metadata that match the search query.
        """
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            },
            "n_results": {
                "type": "integer",
                "description": "The number of results to return.",
                "default": 10,
            },
            "input_type": {
                "type": "string",
                "description": "Optional. The type of input to filter by. Use to refine search if a broad search is ambiguous or returns too many irrelevant results.",
                "enum": ALL_TYPES,
            },
            "output_type": {
                "type": "string",
                "description": "Optional. The type of output to filter by. Use to refine search if a broad search is ambiguous or returns too many irrelevant results.",
                "enum": ALL_TYPES,
            },
            "exclude_namespaces": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            },
        },
    }

    def __init__(self, *args, **kwargs):
        self.exclude_namespaces = kwargs.pop("exclude_namespaces", [])
        super().__init__(*args, **kwargs)

    async def process(self, context: ProcessingContext, params: dict[str, Any]):
        assert "query" in params, "query is required"
        query = params["query"]
        input_type = params.get("input_type", None)
        output_type = params.get("output_type", None)
        n_results = params.get("n_results", 10)
        exclude_namespaces = params.get("exclude_namespaces", self.exclude_namespaces)
        if input_type and input_type not in ALL_TYPES:
            raise ValueError(f"Invalid input type: {input_type}")
        if output_type and output_type not in ALL_TYPES:
            raise ValueError(f"Invalid output type: {output_type}")

        # Import here to avoid circular imports
        from nodetool.chat.search_nodes import search_nodes

        results = search_nodes(
            query=query,
            input_type=input_type,
            output_type=output_type,
            n_results=n_results,
            exclude_namespaces=exclude_namespaces,
        )
        for result in results:
            # Import node class to import necessary types
            get_node_class(result.node_type)

        return [
            {
                "type": result.node_type,
                "title": result.title,
                "description": result.description,
                "properties": {
                    prop.name: prop.type.get_json_schema() for prop in result.properties
                },
                "outputs": {
                    out.name: out.type.get_json_schema() for out in result.outputs
                },
                "is_dynamic": result.is_dynamic,
                "is_streaming": result.is_streaming,
            }
            for result in results
        ]


class SearchExamplesTool(Tool):
    name: str = "search_examples"
    description: str = (
        "Searches for relevant Nodetool workflow examples. Use for finding example workflows and use cases."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            },
        },
    }

    async def process(self, context: ProcessingContext, params: dict[str, Any]):
        assert "query" in params, "query is required"
        query = params["query"]
        log.info(f"Executing SearchExamplesTool with query: {query}")
        # Import here to avoid circular imports
        from nodetool.chat.search_examples import search_examples

        return search_examples(query=query)
