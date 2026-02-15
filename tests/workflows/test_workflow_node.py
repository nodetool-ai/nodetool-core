import pytest

from nodetool.models.workflow import Workflow
from nodetool.workflows.base_node import BaseNode, InputNode, OutputNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.workflow_node import WorkflowNode


class IntegerInput(InputNode):
    value: int = 0

    async def process(self, context: ProcessingContext) -> int:
        return self.value


class IntegerOutput(OutputNode):
    value: int = 0

    async def process(self, context: ProcessingContext) -> int:
        return self.value


class Add(BaseNode):
    a: int = 0
    b: int = 0

    async def process(self, context: ProcessingContext) -> int:
        return self.a + self.b


math_json = {
    "number1": {
        "inputs": {"name": "number1", "value": 1},
        "type": IntegerInput.get_node_type(),
    },
    "number2": {
        "inputs": {"name": "number2", "value": 2},
        "type": IntegerInput.get_node_type(),
    },
    "addition": {
        "inputs": {"a": ["number1", "output"], "b": ["number2", "output"]},
        "type": Add.get_node_type(),
    },
    "output": {
        "inputs": {"name": "output", "value": ["addition", "output"]},
        "type": IntegerOutput.get_node_type(),
    },
}


def test_workflow_node_is_dynamic():
    assert WorkflowNode.is_dynamic() is True


@pytest.mark.asyncio
async def test_read_workflow(context: ProcessingContext):
    """Test that load_graph can load a workflow by ID from the context."""
    # Create a workflow in the database
    workflow = await Workflow.create(
        user_id=context.user_id,
        name="test_math_workflow",
        graph=math_json,
    )

    workflow_node = WorkflowNode(workflow_id=workflow.id)
    graph = await workflow_node.load_graph(context)
    assert graph is not None
    assert len(graph.nodes) == 4


@pytest.mark.asyncio
async def test_process(context: ProcessingContext):
    # Create a workflow in the database
    workflow = await Workflow.create(
        user_id=context.user_id,
        name="test_math_workflow_process",
        graph=math_json,
    )

    workflow_node = WorkflowNode(
        workflow_id=workflow.id,
        dynamic_properties={"number1": 1, "number2": 2},  # type: ignore
    )

    output = {}
    async for result in workflow_node.gen_process(context):
        output.update(result)

    assert output == {"output": 3}
