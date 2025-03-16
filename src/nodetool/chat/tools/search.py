"""
Search and database tools module.

This module provides tools for semantic and keyword searching:
- ChromaTextSearchTool: Semantic search in ChromaDB
- ChromaHybridSearchTool: Combined semantic/keyword search
- SemanticDocSearchTool: Search documentation semantically
- KeywordDocSearchTool: Search documentation by keywords
"""

from typing import Any, Dict

from chromadb.api.types import IncludeEnum
from nodetool.workflows.processing_context import ProcessingContext
from .base import Tool


class ChromaTextSearchTool(Tool):
    name = "chroma_text_search"
    description = (
        "Search all ChromaDB collections for similar text using semantic search"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to search for",
            },
            "n_results": {
                "type": "integer",
                "description": "Number of results to return",
                "default": 10,
            },
        },
        "required": ["text"],
    }

    async def process(self, context: ProcessingContext, params: dict) -> dict[str, str]:
        try:
            from nodetool.common.chroma_client import (
                get_all_collections,
            )

            collections = get_all_collections()
            combined = []

            for collection in collections:
                result = collection.query(
                    query_texts=[params["text"]],
                    n_results=params.get("n_results", 5),
                )

                # Sort results by ID for consistency
                if result["documents"] is None:
                    continue
                combined.extend(
                    list(
                        zip(
                            result["ids"][0],
                            result["documents"][0],
                        )
                    )
                )

            combined.sort(key=lambda x: str(x[0]))

            return dict(combined)

        except Exception as e:
            import traceback

            traceback.print_exc()
            return {"error": str(e)}


class ChromaHybridSearchTool(Tool):
    name = "chroma_hybrid_search"
    description = (
        "Search all ChromaDB collections using both semantic and keyword-based search"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to search for",
            },
            "n_results": {
                "type": "integer",
                "description": "Number of results to return per collection",
                "default": 5,
            },
            "k_constant": {
                "type": "number",
                "description": "Constant for reciprocal rank fusion",
                "default": 60.0,
            },
            "min_keyword_length": {
                "type": "integer",
                "description": "Minimum length for keyword tokens",
                "default": 3,
            },
        },
        "required": ["text"],
    }

    def _get_keyword_query(self, text: str, min_length: int) -> dict:
        import re

        pattern = r"[ ,.!?\-_=|]+"
        query_tokens = [
            token.strip()
            for token in re.split(pattern, text.lower())
            if len(token.strip()) >= min_length
        ]

        if not query_tokens:
            return {}

        if len(query_tokens) > 1:
            return {"$or": [{"$contains": token} for token in query_tokens]}
        return {"$contains": query_tokens[0]}

    async def process(self, context: ProcessingContext, params: dict) -> dict[str, str]:
        try:
            from nodetool.common.chroma_client import get_all_collections

            if not params["text"].strip():
                return {"error": "Search text cannot be empty"}

            collections = get_all_collections()
            n_results = params.get("n_results", 5)
            k_constant = params.get("k_constant", 60.0)
            min_keyword_length = params.get("min_keyword_length", 3)

            all_results = []

            for collection in collections:
                # Semantic search
                semantic_results = collection.query(
                    query_texts=[params["text"]],
                    n_results=n_results * 2,
                    include=[IncludeEnum.documents],
                )

                # Keyword search
                keyword_query = self._get_keyword_query(
                    params["text"], min_keyword_length
                )
                if keyword_query:
                    keyword_results = collection.query(
                        query_texts=[params["text"]],
                        n_results=n_results * 2,
                        where_document=keyword_query,
                        include=[IncludeEnum.documents],
                    )
                else:
                    keyword_results = semantic_results

                # Combine results using reciprocal rank fusion
                combined_scores = {}

                if semantic_results["documents"]:
                    # Process semantic results
                    for rank, (id_, doc) in enumerate(
                        zip(
                            semantic_results["ids"][0],
                            semantic_results["documents"][0],
                        )
                    ):
                        score = 1 / (rank + k_constant)
                        combined_scores[id_] = {"doc": doc, "score": score}

                if keyword_results["documents"]:
                    # Process keyword results
                    for rank, (id_, doc) in enumerate(
                        zip(
                            keyword_results["ids"][0],
                            keyword_results["documents"][0],
                        )
                    ):
                        score = 1 / (rank + k_constant)
                        if id_ in combined_scores:
                            combined_scores[id_]["score"] += score
                        else:
                            combined_scores[id_] = {"doc": doc, "score": score}

                # Add top results from this collection
                sorted_results = sorted(
                    combined_scores.items(), key=lambda x: x[1]["score"], reverse=True
                )[:n_results]
                all_results.extend(sorted_results)

            # Sort all results and take top n
            final_results = sorted(
                all_results, key=lambda x: x[1]["score"], reverse=True
            )[:n_results]

            # Convert to simple id->document dictionary
            return {str(id_): item["doc"] for id_, item in final_results}

        except Exception as e:
            return {"error": str(e)}


class SemanticDocSearchTool(Tool):
    name = "semantic_doc_search"
    description = "Search documentation using semantic similarity"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The text to search for in the documentation",
            },
        },
        "required": ["query"],
    }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            from nodetool.chat.help import semantic_search_documentation

            results = semantic_search_documentation(params["query"])
            return {
                "results": [
                    {
                        "id": result.id,
                        "content": result.content,
                        "metadata": result.metadata,
                    }
                    for result in results
                ]
            }
        except Exception as e:
            return {"error": str(e)}


class KeywordDocSearchTool(Tool):
    name = "keyword_doc_search"
    description = "Search documentation using keyword matching"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The text to search for in the documentation",
            },
        },
        "required": ["query"],
    }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            from nodetool.chat.help import keyword_search_documentation

            results = keyword_search_documentation(params["query"])
            return {
                "results": [
                    {
                        "id": result.id,
                        "content": result.content,
                        "metadata": result.metadata,
                    }
                    for result in results
                ]
            }
        except Exception as e:
            return {"error": str(e)}
