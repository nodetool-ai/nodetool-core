from typing import Any, ClassVar

from pydantic import Field

from nodetool.types.graph import Graph as APIGraph
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.graph import Graph
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.read_graph import read_graph
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.types import (
    Chunk,
    Error,
    LogUpdate,
    NodeProgress,
    NodeUpdate,
    OutputUpdate,
)


class WorkflowNode(BaseNode):
    """
    Executes a sub-workflow as a reusable component within a larger workflow.

    Use this node to embed and run complete workflows inside other workflows,
    enabling modular and composable workflow design. Input nodes in the sub-workflow
    become dynamic properties, and output nodes provide the results.
    """

    _dynamic = True
    _supports_dynamic_outputs: ClassVar[bool] = True

    workflow_json: dict[str, Any] = Field(
        default_factory=dict, description="The workflow definition as a JSON dictionary."
    )

    @classmethod
    def is_streaming_output(cls):
        return True

    def load_graph(self):
        edges, nodes = read_graph(self.workflow_json)
        return Graph.from_dict(
            {
                "nodes": [node.model_dump() for node in nodes],
                "edges": [edge.model_dump() for edge in edges],
            }
        )

    def get_api_graph(self) -> APIGraph:
        edges, nodes = read_graph(self.workflow_json)
        return APIGraph(edges=edges, nodes=nodes)

    async def gen_process(self, context: ProcessingContext):
        req = RunJobRequest(
            user_id=context.user_id,
            auth_token=context.auth_token,
            graph=self.get_api_graph(),
            params=self._dynamic_properties,
        )
        async for msg in run_workflow(req):
            if isinstance(msg, Error):
                raise Exception(msg.error)
            if isinstance(msg, OutputUpdate):
                yield msg.output_name, msg.value
            if isinstance(msg, NodeProgress):
                context.post_message(
                    NodeProgress(
                        node_id=self._id,
                        progress=msg.progress,
                        total=msg.total,
                    )
                )
            if isinstance(msg, Chunk):
                context.post_message(Chunk(content=msg.content))
            if isinstance(msg, LogUpdate):
                context.post_message(
                    LogUpdate(
                        node_id=self._id,
                        node_name=msg.node_name,
                        content=msg.content,
                        severity=msg.severity,
                    )
                )
            if isinstance(msg, NodeUpdate) and msg.status == "completed":
                context.post_message(
                    LogUpdate(
                        node_id=self._id,
                        node_name=msg.node_name,
                        content=f"{msg.node_name} {msg.status}",
                        severity="error",
                    )
                )
