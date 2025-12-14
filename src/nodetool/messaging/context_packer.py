"""
Context packer for HelpMessageProcessor.

Provides utilities to create compact graph representations for LLM context,
minimizing token usage while preserving essential information.
"""

from typing import Any

from nodetool.types.graph import Edge, Graph, Node


def estimate_tokens(content: str) -> int:
    """Rough token estimate (chars / 4).
    
    This is a simple heuristic that works reasonably well for most LLMs.
    For more accurate counts, use tiktoken or the model's tokenizer.
    """
    return len(content) // 4


def create_compact_graph_context(
    graph: Graph,
    selection: dict[str, Any] | None = None,
    max_tokens: int = 8000,
) -> dict[str, Any]:
    """Create a compact graph representation for the LLM.
    
    Strips UI properties and verbose metadata, keeping only essential
    information for the model to understand and modify the graph.
    
    Args:
        graph: The full graph object
        selection: Optional selection context (e.g., {"node_id": "n2"})
        max_tokens: Maximum token budget for the context
        
    Returns:
        Compact dict representation:
        {
            "nodes": [{"id": "n1", "type": "...", "name": "..."}, ...],
            "edges": [{"src": "n1.output", "dst": "n2.text"}, ...],
            "selection": {"node_id": "n2"} | None,
            "node_count": 5,
            "edge_count": 4
        }
    """
    compact_nodes = []
    for node in graph.nodes:
        compact_node: dict[str, Any] = {
            "id": node.id,
            "type": node.type,
        }
        # Include name if present in data
        if node.data and isinstance(node.data, dict):
            if "name" in node.data:
                compact_node["name"] = node.data["name"]
            # Include essential non-default data values (skip UI-only fields)
            essential_data = {
                k: v for k, v in node.data.items()
                if k not in ("name",) and v is not None and v != "" and v != {}
            }
            if essential_data:
                compact_node["data"] = essential_data
                
        compact_nodes.append(compact_node)
    
    # Compact edge format: "source.handle" -> "target.handle"
    compact_edges = [
        {
            "src": f"{edge.source}.{edge.sourceHandle}",
            "dst": f"{edge.target}.{edge.targetHandle}",
        }
        for edge in graph.edges
    ]
    
    result: dict[str, Any] = {
        "nodes": compact_nodes,
        "edges": compact_edges,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
    }
    
    if selection:
        result["selection"] = selection
        
    return result


def get_node_neighborhood(
    graph: Graph,
    node_id: str,
    hops: int = 2,
) -> Graph:
    """Extract subgraph within N hops of a node.
    
    Useful when the user asks about a specific node - we only need
    to include nearby nodes, not the entire graph.
    
    Args:
        graph: The full graph
        node_id: Center node ID
        hops: Number of edge hops to include (default 2)
        
    Returns:
        Subgraph containing only nodes within N hops
    """
    if hops < 0:
        raise ValueError("hops must be non-negative")
        
    # Build adjacency (both directions since graph is directed but we want neighborhood)
    adjacency: dict[str, set[str]] = {}
    for node in graph.nodes:
        adjacency[node.id] = set()
    
    for edge in graph.edges:
        if edge.source in adjacency:
            adjacency[edge.source].add(edge.target)
        if edge.target in adjacency:
            adjacency[edge.target].add(edge.source)
    
    # BFS to find nodes within N hops
    visited: set[str] = {node_id}
    frontier: set[str] = {node_id}
    
    for _ in range(hops):
        next_frontier: set[str] = set()
        for nid in frontier:
            for neighbor in adjacency.get(nid, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break
    
    # Filter nodes and edges to the neighborhood
    neighborhood_nodes = [n for n in graph.nodes if n.id in visited]
    neighborhood_edges = [
        e for e in graph.edges
        if e.source in visited and e.target in visited
    ]
    
    return Graph(nodes=neighborhood_nodes, edges=neighborhood_edges)
