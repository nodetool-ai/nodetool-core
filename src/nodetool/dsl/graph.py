from nodetool.workflows.types import Error, OutputUpdate
from pydantic import BaseModel, Field
import uuid

from nodetool.types.graph import Graph, Node
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.processing_context import (
    AssetOutputMode,
    ProcessingContext,
)


class GraphNode(BaseModel):
    """
    Represents a node in a graph DSL.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def set_node_type(cls, node_type: str):
        cls.__annotations__["node_type"] = node_type

    @classmethod
    def get_node_type(cls) -> str:
        return cls.__annotations__["node_type"]


def graph(*nodes):
    """
    Create a graph representation based on the given nodes.

    Args:
      *nodes: Variable number of nodes to be included in the graph.

    Returns:
      Graph: A graph object containing the nodes and edges.

    """
    from nodetool.dsl.graph_node_converter import GraphNodeConverter

    g = GraphNodeConverter()
    for node in nodes:
        g.add(node)
    nodes = [
        Node(
            id=n._id,
            type=n.get_node_type(),
            data=n.model_dump(),
        )
        for n in g.nodes.values()
    ]
    return Graph(nodes=nodes, edges=g.edges)


async def run_graph(
    graph: Graph,
    user_id: str = "1",
    auth_token: str = "token",
    asset_output_mode: AssetOutputMode | None = None,
):
    """
    Run the workflow with the given graph.

    Args:
      graph (Graph): The graph object representing the workflow.
      asset_output_mode (AssetOutputMode | None): Optional asset output mode applied to the run.

    Returns:
      Any: The result of the workflow execution.
    """
    req = RunJobRequest(user_id=user_id, auth_token=auth_token, graph=graph)

    context = None
    if asset_output_mode is not None:
        context = ProcessingContext(
            user_id=user_id,
            auth_token=auth_token,
            asset_output_mode=asset_output_mode,
        )

    res = {}
    async for msg in run_workflow(req, context=context):
        if isinstance(msg, OutputUpdate):
            res[msg.node_name] = msg.value
        elif isinstance(msg, Error):
            raise Exception(msg.error)
    return res


async def graph_result(
    example,
    asset_output_mode: AssetOutputMode | None = None,
):
    """
    Helper function to run a graph and return its result.

    Args:
        example: The graph example to run
        asset_output_mode: Optional asset output mode applied to execution.

    Returns:
        The result of running the graph
    """
    result = await run_graph(
        graph(example),
        asset_output_mode=asset_output_mode,
    )
    assert result is not None, "Result is None"
    return result
