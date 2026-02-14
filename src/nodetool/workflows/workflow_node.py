from typing import Any, ClassVar

from pydantic import Field

from nodetool.types.api_graph import Graph as APIGraph
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
    A WorkflowNode is a node that can execute a sub-workflow.

    - Load and manage workflow definitions from JSON, including validation of the structure.
    - Generate properties based on input nodes in the workflow, allowing for dynamic input handling.
    - Execute sub-workflows within a larger workflow context, enabling modular workflow design.
    - Handle progress updates, error reporting, and logging during workflow execution to facilitate debugging and monitoring.
    """

    _is_dynamic: ClassVar[bool] = True
    _supports_dynamic_outputs: ClassVar[bool] = True

    workflow_id: str = Field(default="", description="The ID of the workflow to execute")

    @classmethod
    def is_streaming_output(cls):
        return True

    
    async def get_workflow_json(self, context: ProcessingContext) -> Any:
        workflow = await context.get_workflow(self.workflow_id)
        if workflow is None:
            raise ValueError(f"Workflow with ID {self.workflow_id} not found")
        # read_graph expects the workflow graph payload, not the workflow model object
        if isinstance(workflow, dict):
            return workflow.get("graph", workflow)
        graph = getattr(workflow, "graph", None)
        if graph is None:
            raise ValueError(f"Workflow {self.workflow_id} has no graph")
        return graph

    def _to_api_graph(self, workflow_graph: Any) -> APIGraph:
        # Current persisted workflows use API graph shape: {"nodes": [...], "edges": [...]}
        if (
            isinstance(workflow_graph, dict)
            and "nodes" in workflow_graph
            and "edges" in workflow_graph
            and isinstance(workflow_graph["nodes"], list)
        ):
            return APIGraph.model_validate(workflow_graph)

        # Backward compatibility for keyed/comfy-like dict graphs
        edges, nodes = read_graph(workflow_graph)
        return APIGraph(edges=edges, nodes=nodes)

    async def load_graph(self, context: ProcessingContext):
        workflow_graph = await self.get_workflow_json(context)
        api_graph = self._to_api_graph(workflow_graph)
        return Graph.from_dict(api_graph.model_dump())

    async def get_api_graph(self, context: ProcessingContext) -> APIGraph:
        workflow_graph = await self.get_workflow_json(context)
        return self._to_api_graph(workflow_graph)

    async def gen_process(self, context: ProcessingContext):
        req = RunJobRequest(
            user_id=context.user_id,
            auth_token=context.auth_token,
            graph=await self.get_api_graph(context),
            params=self._dynamic_properties,
        )
        async for msg in run_workflow(req):
            if isinstance(msg, Error):
                raise Exception(msg.message)
            if isinstance(msg, OutputUpdate):
                yield {msg.output_name: msg.value}
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
