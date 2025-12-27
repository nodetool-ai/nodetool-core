import asyncio
import queue
import pytest
from nodetool.types.graph import Graph as APIGraph
from nodetool.types.graph import Node as APINode
from nodetool.types.graph import Edge as APIEdge
from nodetool.workflows.base_node import InputNode, OutputNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.workflow_runner import WorkflowRunner
from nodetool.workflows.run_job_request import RunJobRequest

class ParamInput(InputNode):
    value: str = "default"

    async def process(self, context: ProcessingContext) -> str:
        return self.value

class ParamOutput(OutputNode):
    value: str = ""

    async def process(self, context: ProcessingContext) -> str:
        return self.value

@pytest.mark.asyncio
async def test_workflow_runner_applies_input_params():
    # Define nodes: Input -> Output
    nodes = [
        APINode(
            id="input_node", 
            type=ParamInput.get_node_type(), 
            data={"name": "my_param", "value": "default_from_graph"}
        ),
        APINode(
            id="output_node", 
            type=ParamOutput.get_node_type(), 
            data={"name": "result"}
        )
    ]
    
    edges = [
        APIEdge(
            id="e1", 
            source="input_node", 
            sourceHandle="output", 
            target="output_node", 
            targetHandle="value"
        )
    ]
    
    api_graph = APIGraph(nodes=nodes, edges=edges)
    
    # Pass a param "my_param" which should override the default
    req = RunJobRequest(
        graph=api_graph,
        params={"my_param": "active_value"}
    )
    
    ctx = ProcessingContext(message_queue=queue.Queue())
    runner = WorkflowRunner(job_id="test-params-1")
    
    await runner.run(req, ctx)
    
    assert runner.status == "completed"
    # Verification: output should reflect the param value
    assert runner.outputs.get("result") == ["active_value"]

@pytest.mark.asyncio
async def test_workflow_runner_raises_on_invalid_param():
    # Define a graph with no inputs
    api_graph = APIGraph(nodes=[], edges=[])
    
    req = RunJobRequest(
        graph=api_graph,
        params={"non_existent_param": "some_value"}
    )
    
    ctx = ProcessingContext(message_queue=queue.Queue())
    runner = WorkflowRunner(job_id="test-params-error")
    
    with pytest.raises(ValueError, match="No input node found for param: non_existent_param"):
        await runner.run(req, ctx)
