from nodetool.types.graph import Edge
from nodetool.workflows.base_node import BaseNode


def convert_to_comfy_workflow(edges: list[Edge], nodes: list[BaseNode]):
    """
    Converts a Nodetool workflow to a ComfyUI-compatible workflow JSON format.

    This function transforms a graph of nodes and edges from the Nodetool format
    into the JSON structure expected by ComfyUI. The conversion process:
    1. Creates a node entry for each BaseNode with its class type and inputs
    2. Resolves connections between nodes by converting edges to the ComfyUI format
       where inputs reference the source node ID and output index

    Parameters:
    -----------
    edges : list[Edge]
        List of Edge objects representing connections between nodes in the workflow.
        Each edge contains source/target node IDs and the corresponding handles.

    nodes : list[BaseNode]
        List of BaseNode objects representing the nodes in the workflow.
        Each node has properties, inputs, and outputs.

    Returns:
    --------
    dict
        A dictionary representing the ComfyUI workflow in JSON format, where:
        - Keys are node IDs
        - Values are dictionaries containing:
          - 'class_type': The class name of the node
          - 'inputs': A dictionary of input parameters, where connected inputs
                     are represented as [source_node_id, source_output_index]

    Raises:
    -------
    Exception
        If a referenced node ID cannot be found or if a referenced output property
        does not exist on the source node.
    """
    json = {}
    for node in nodes:
        inputs = node.node_properties()
        inputs = {k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in inputs.items() if v is not None}
        json[node._id] = {
            "class_type": node.__class__.__name__,
            "inputs": inputs,
        }

    def find_node(id):
        for node in nodes:
            if node._id == id:
                return node
        raise Exception(f"Could not find node with id {id}")

    def find_output(node: BaseNode, name: str):
        for index, output in enumerate(node.outputs()):
            if output.name == name:
                return index
        raise Exception(f"Could not find property {name} in node {node}")

    for edge in edges:
        source_node = find_node(edge.source)
        source_index = find_output(source_node, edge.sourceHandle)
        json[edge.target]["inputs"][edge.targetHandle] = [edge.source, source_index]

    return json
