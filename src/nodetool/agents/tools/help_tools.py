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

from typing import Any, ClassVar

from nodetool.agents.tools.base import Tool
from nodetool.config.logging_config import get_logger
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
    description: str = """
        Performs keyword search on nodetool nodes.
        Use for finding node types by exact word matches.
        Supply a list of words to search for as array, including synonyms and related words.
        By default returns only matching node_type strings (token-saving).
        Set include_description/include_properties to fetch more detail only when needed.
        """
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            },
            "include_description": {
                "type": "boolean",
                "description": "Include title/description fields in results (more tokens).",
                "default": False,
            },
            "include_properties": {
                "type": "boolean",
                "description": "Include property schemas in results (more tokens).",
                "default": False,
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
        include_description = bool(params.get("include_description", False))
        include_properties = bool(params.get("include_properties", False))
        input_type = params.get("input_type")
        output_type = params.get("output_type")
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

        # Default: return only node_type strings (saves tokens).
        if not include_description and not include_properties:
            return [node_type.node_type for node_type in results]

        enriched: list[dict[str, Any]] = []
        for node_type in results:
            item: dict[str, Any] = {"node_type": node_type.node_type}
            if include_description:
                item["title"] = node_type.title
                item["description"] = node_type.description
            if include_properties:
                item["properties"] = [
                    {
                        "name": prop.name,
                        "type": prop.type.type,
                        "description": getattr(prop, "description", None),
                        "default": getattr(prop, "default", None),
                        "required": bool(getattr(prop, "required", False)),
                    }
                    for prop in node_type.properties
                ]
                item["outputs"] = [
                    {"name": out.name, "type": out.type.type if out.type else "any"}
                    for out in node_type.outputs
                ]
                item["is_dynamic"] = node_type.is_dynamic
            enriched.append(item)

        return enriched


class SearchExamplesTool(Tool):
    name: str = "search_examples"
    description: str = "Searches for relevant Nodetool workflow examples. Use for finding example workflows and use cases."
    input_schema: ClassVar[dict[str, Any]] = {
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
