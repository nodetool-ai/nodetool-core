from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.read_graph import read_graph
from nodetool.types.graph import Graph as APIGraph
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.graph import Graph
from typing import Any

from nodetool.workflows.types import Chunk, NodeProgress


class WorkflowNode(BaseNode):
    """
    A WorkflowNode is a node that can execute a sub-workflow.

    - Load and manage workflow definitions from JSON, including validation of the structure.
    - Generate properties based on input nodes in the workflow, allowing for dynamic input handling.
    - Execute sub-workflows within a larger workflow context, enabling modular workflow design.
    - Handle progress updates, error reporting, and logging during workflow execution to facilitate debugging and monitoring.
    """

    _dynamic = True

    workflow_json: dict = {}

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

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        req = RunJobRequest(
            user_id=context.user_id,
            auth_token=context.auth_token,
            graph=self.get_api_graph(),
            params=self._dynamic_properties,
        )
        output = {}
        async for msg in run_workflow(req):
            # Convert Pydantic model to dict for uniform access
            if hasattr(msg, "model_dump"):
                msg_dict = msg.model_dump()
            else:
                msg_dict = msg

            assert "type" in msg_dict
            if msg_dict["type"] == "error":
                raise Exception(msg_dict["error"])
            if msg_dict["type"] == "job_update":
                if msg_dict["status"] == "completed":
                    # Prefer the final value per output; reduce lists to last element
                    result = msg_dict.get("result") or {}
                    reduced: dict[str, Any] = {}
                    for k, v in result.items():
                        if isinstance(v, list) and len(v) > 0:
                            reduced[k] = [v[-1]]
                        else:
                            reduced[k] = v
                    output = reduced
            if msg_dict["type"] == "node_progress":
                context.post_message(
                    NodeProgress(
                        node_id=self._id,
                        progress=msg_dict["progress"],
                        total=msg_dict["total"],
                    )
                )
            if msg_dict["type"] == "chunk":
                context.post_message(Chunk(content=msg_dict["content"]))
            if msg_dict["type"] == "node_update":
                if msg_dict["status"] == "completed":
                    context.post_message(
                        NodeProgress(
                            node_id=self._id,
                            progress=0,
                            total=0,
                            chunk=f"{msg_dict['node_name']} {msg_dict['status']}",
                        )
                    )
        return output
