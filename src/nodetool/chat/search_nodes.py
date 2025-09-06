from nodetool.metadata.node_metadata import NodeMetadata
from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.packages.registry import Registry


registry = Registry()


def search_nodes(
    query: list[str],
    input_type: str | None = None,
    output_type: str | None = None,
    exclude_namespaces: list[str] = [],
    n_results: int = 10,
) -> list[NodeMetadata]:
    """
    Iterates over all nodes and checks if the query is in the node description.
    If it is, it adds the node to the search results.

    Args:
        query: The query to search for.
    input_type: The type of input to search for.
        output_type: The type of output to search for.
        n_results: The number of results to return.

    Returns:
        A list of search results from keyword matching.
    """
    node_metadata_list = registry.get_all_installed_nodes()
    query_lower = [q.lower() for q in query]

    def type_matches(type_metadata: TypeMetadata, type_str: str) -> bool:
        if type_metadata.type == "any":
            return True
        if type_metadata.type == "union":
            return any(type_matches(arg, type_str) for arg in type_metadata.type_args)
        if type_metadata.type == "enum":
            return type_str in type_metadata.values if type_metadata.values else False
        return type_metadata.type == type_str

    scored_results = []
    for node_metadata in node_metadata_list:
        title = node_metadata.title.lower()
        name = node_metadata.node_type.lower()
        desc = node_metadata.description.lower()
        input_types = [prop.type for prop in node_metadata.properties if prop.type]
        output_types = [out.type for out in node_metadata.outputs if out.type]
        if node_metadata.namespace in exclude_namespaces:
            continue
        if input_type:
            if not any(type_matches(t, input_type) for t in input_types):
                continue
        if output_type:
            if not any(type_matches(t, output_type) for t in output_types):
                continue
        score = 0
        if any(q in title for q in query_lower):
            score += 10
        if any(q in name for q in query_lower):
            score += 5
        if any(q in desc for q in query_lower):
            score += 1

        if score > 0:
            scored_results.append({"score": score, "node": node_metadata})

    # Sort results by score in descending order
    scored_results.sort(key=lambda x: x["score"], reverse=True)

    # Extract just the node data for the final list
    search_results = [result["node"] for result in scored_results]

    if len(search_results) > n_results:
        return search_results[:n_results]
    return search_results
