import pytest
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


def test_read_workflow():
    workflow_json = WorkflowNode(workflow_json=math_json)
    graph = workflow_json.load_graph()
    assert graph


@pytest.mark.asyncio
async def test_process(context: ProcessingContext):
    workflow_node = WorkflowNode(
        workflow_json=math_json,
        dynamic_properties={"number1": 1, "number2": 2},  # type: ignore
    )

    output = {}
    async for handle, value in workflow_node.gen_process(context):
        output[handle] = value

    assert output == {"output": 3}
