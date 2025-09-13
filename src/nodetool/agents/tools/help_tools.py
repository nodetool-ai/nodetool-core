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
from nodetool.config.logging_config import get_logger
from nodetool.workflows.base_node import BaseNode, get_node_class
from nodetool.workflows.processing_context import ProcessingContext

log = get_logger(__name__)

TYPES = [
    "str",
    "int",
    "float",
    "bool",
    "list",
    "dict",
    "tuple",
    "union",
    "enum",
    "any",
    "bytes",
    "audio",
    "image",
    "video",
    "document",
    "dataframe",
]


class SearchNodesTool(Tool):
    name: str = "search_nodes"
    description: str = (
        """
        Performs keyword search on nodetool nodes. 
        Use for finding specific node types or features by exact word matches.
        Supply a list of words to search for as array, including synonyms and related words.
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
                "enum": TYPES,
            },
            "output_type": {
                "type": "string",
                "description": "Optional. The type of output to filter by. Use to refine search if a broad search is ambiguous or returns too many irrelevant results.",
                "enum": TYPES,
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
        if input_type and input_type not in TYPES:
            raise ValueError(f"Invalid input type: {input_type}")
        if output_type and output_type not in TYPES:
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
        node_types: list[type[BaseNode]] = []
        for result in results:
            # Import node class to import necessary types
            node_type = get_node_class(result.node_type)
            assert node_type is not None, f"Node type {result.node_type} not found"
            node_types.append(node_type)

        return [
            {
                "type": node_type.get_node_type(),
                "title": node_type.get_title(),
                "description": node_type.get_description(),
                "properties": {
                    prop.name: prop.type.get_json_schema()
                    for prop in node_type.properties()
                },
                "outputs": {
                    out.name: out.type.get_json_schema() for out in node_type.outputs()
                },
                "is_dynamic": node_type.is_dynamic(),
                "is_streaming_output": node_type.is_streaming_output(),
                "is_streaming_input": node_type.is_streaming_input(),
            }
            for node_type in node_types
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

        results = []
        for q in query:
            log.info(f"Searching for examples with query: {q}")
            results.extend(search_examples(query=q))
        return results
