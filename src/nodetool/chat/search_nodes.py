import re
from itertools import combinations, permutations

from nodetool.metadata.node_metadata import NodeMetadata
from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.packages.registry import Registry

registry = Registry()

_CAMEL_SPLIT_1 = re.compile(r"([a-z0-9])([A-Z])")
_CAMEL_SPLIT_2 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_NON_ALNUM = re.compile(r"[^a-zA-Z0-9]+")
_WS = re.compile(r"\s+")


def _normalize_text(value: str) -> str:
    value = _CAMEL_SPLIT_2.sub(r"\1 \2", value)
    value = _CAMEL_SPLIT_1.sub(r"\1 \2", value)
    value = value.replace(".", " ").replace("_", " ").replace("-", " ").replace("/", " ")
    value = _NON_ALNUM.sub(" ", value)
    value = value.lower()
    return _WS.sub(" ", value).strip()


def _normalize_query_tokens(query: list[str]) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in query:
        normalized = _normalize_text(raw)
        for part in normalized.split():
            if not part or part in seen:
                continue
            seen.add(part)
            tokens.append(part)
    return tokens


def _compile_phrase_regexes(tokens: list[str]) -> list[tuple[int, re.Pattern[str]]]:
    """
    Builds a bounded set of regex patterns that match word sequences with gaps, e.g.:
      text.*to.*image
    """
    if len(tokens) < 2:
        return []

    # Bound complexity: keep the first N tokens from user query.
    tokens = tokens[:7]

    max_patterns = 256
    max_k = min(4, len(tokens))
    patterns: list[tuple[int, re.Pattern[str]]] = []

    def add_pattern(k: int, ordered_terms: tuple[str, ...]) -> bool:
        expr = r"\b" + r"\b.*\b".join(re.escape(t) for t in ordered_terms) + r"\b"
        patterns.append((k, re.compile(expr, re.IGNORECASE)))
        return len(patterns) < max_patterns

    # Strong signal: the user's order as provided.
    add_pattern(len(tokens), tuple(tokens))

    # Add shorter phrase patterns; prioritize longer phrases first.
    for k in range(max_k, 1, -1):
        for comb in combinations(tokens, k):
            for perm in permutations(comb, k):
                if not add_pattern(k, perm):
                    return patterns

    return patterns


def search_nodes(
    query: list[str],
    input_type: str | None = None,
    output_type: str | None = None,
    exclude_namespaces: list[str] | None = None,
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
    query_tokens = _normalize_query_tokens(query)
    query_lower = [q.lower() for q in query_tokens]
    phrase_regexes = _compile_phrase_regexes(query_tokens)
    exclude_namespaces = exclude_namespaces or []

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
        title = _normalize_text(node_metadata.title or "")
        name = _normalize_text(node_metadata.node_type or "")
        desc = _normalize_text(node_metadata.description or "")
        title_and_name = f"{title} {name}".strip()
        input_types = [prop.type for prop in node_metadata.properties if prop.type]
        output_types = [out.type for out in node_metadata.outputs if out.type]
        if node_metadata.namespace in exclude_namespaces:
            continue
        if input_type and not any(type_matches(t, input_type) for t in input_types):
            continue
        if output_type and not any(type_matches(t, output_type) for t in output_types):
            continue
        score = 0
        # Phrase match boost: reward sequences like "text.*to.*image" across title/name.
        max_phrase_len = 0
        for k, rx in phrase_regexes:
            if rx.search(title_and_name):
                if k > max_phrase_len:
                    max_phrase_len = k
        if max_phrase_len:
            score += 60 + (max_phrase_len * 20)

        # Term-level scoring with coverage.
        title_hits = sum(1 for q in query_lower if q in title)
        name_hits = sum(1 for q in query_lower if q in name)
        desc_hits = sum(1 for q in query_lower if q in desc)
        score += title_hits * 10
        score += name_hits * 6
        score += desc_hits * 1

        # Bonus when most query tokens appear in title/name.
        if query_lower:
            covered = sum(1 for q in query_lower if q in title_and_name)
            coverage_ratio = covered / len(query_lower)
            if coverage_ratio >= 0.75:
                score += 25
            elif coverage_ratio >= 0.5:
                score += 10

        if score > 0:
            scored_results.append({"score": score, "node": node_metadata})

    # Sort results by score in descending order
    scored_results.sort(key=lambda x: x["score"], reverse=True)

    # Extract just the node data for the final list
    search_results = [result["node"] for result in scored_results]

    if len(search_results) > n_results:
        return search_results[:n_results]
    return search_results
