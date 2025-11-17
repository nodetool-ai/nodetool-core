from typing import Any

from nodetool.packages.registry import Registry


def search_examples(query: str, n_results: int = 5) -> list[dict[str, Any]]:
    """
    Search the examples for the given query string.

    Args:
        query: The query to search for.
        n_results: The number of results to return.

    Returns:
        A tuple of the ids and documents that match the query.
    """
    registry = Registry()
    examples = registry.list_examples()
    search_results = []
    for example in examples:
        if query in example.name or query in example.description:
            search_results.append(example.model_dump())
    if len(search_results) > n_results:
        return search_results[:n_results]
    return search_results
