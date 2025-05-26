import re
import asyncio
from unittest.mock import Mock, AsyncMock
from nodetool.models.workflow import Workflow
from nodetool.types.graph import Node, Edge
from nodetool.workflows.export_dsl import workflow_to_dsl, _edge_map, _dsl_import_path
from nodetool.workflows.base_node import InputNode, OutputNode, BaseNode


# Define simple test nodes for our tests
class TestAdd(BaseNode):
    """Simple addition node for testing."""

    a: float = 0.0
    b: float = 0.0

    async def process(self):
        return self.a + self.b

    @classmethod
    def get_node_type(cls):
        return "test.math.Add"


class TestMultiply(BaseNode):
    """Simple multiplication node for testing."""

    a: float = 0.0
    b: float = 0.0

    async def process(self):
        return self.a * self.b

    @classmethod
    def get_node_type(cls):
        return "test.math.Multiply"


class TestSquare(BaseNode):
    """Simple square node for testing."""

    input: float = 0.0

    async def process(self):
        return self.input**2

    @classmethod
    def get_node_type(cls):
        return "test.math.Square"


class TestNumber(BaseNode):
    """Simple constant number node for testing."""

    value: float = 0.0

    async def process(self):
        return self.value

    @classmethod
    def get_node_type(cls):
        return "test.constant.Number"


class TestChatGPT(BaseNode):
    """Mock ChatGPT node for testing."""

    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 100
    system_prompt: str = ""
    messages: list = []

    async def process(self):
        return "Mock response"

    @classmethod
    def get_node_type(cls):
        return "test.llm.ChatGPT"


def test_edge_map():
    """Test the edge mapping utility function."""
    edges = [
        Edge(
            id="e1",
            source="node1",
            sourceHandle="output",
            target="node2",
            targetHandle="input",
        ),
        Edge(
            id="e2",
            source="node2",
            sourceHandle="result",
            target="node3",
            targetHandle="value",
        ),
    ]

    mapping = _edge_map(edges)

    assert len(mapping) == 2
    assert ("node2", "input") in mapping
    assert ("node3", "value") in mapping
    assert mapping[("node2", "input")].source == "node1"
    assert mapping[("node3", "value")].sourceHandle == "result"


def test_dsl_import_path():
    """Test the DSL import path generation."""
    module, class_name = _dsl_import_path("test.input.TextInput")
    assert module == "nodetool.dsl.test.input"
    assert class_name == "TextInput"

    module, class_name = _dsl_import_path("custom.processing.DataProcessor")
    assert module == "nodetool.dsl.custom.processing"
    assert class_name == "DataProcessor"


def test_workflow_to_dsl_basic():
    """Test basic workflow with single input and output."""
    nodes = [
        Node(
            id="1",
            type=InputNode.get_node_type(),
            data={"name": "text_input", "value": "Hello"},
        ),
        Node(
            id="2",
            type=OutputNode.get_node_type(),
            data={"name": "result"},
        ),
    ]
    edges = [
        Edge(
            id="e1",
            source="1",
            sourceHandle="output",
            target="2",
            targetHandle="value",
        )
    ]
    wf = Workflow(
        id="w1",
        user_id="u1",
        name="Simple Workflow",
        description="A simple test workflow",
        graph={
            "nodes": [n.model_dump() for n in nodes],
            "edges": [e.model_dump() for e in edges],
        },
    )

    code = workflow_to_dsl(wf)

    # Check function definition
    assert "async def simple_workflow(text_input):" in code
    assert "A simple test workflow" in code

    # Check node creation
    assert re.search(r"n0 = .*Input\(.*value=text_input", code)
    assert re.search(r"n1 = .*Output\(.*value=\(n0, 'output'\)", code)

    # Check graph creation and execution
    assert "g = graph(n0, n1)" in code
    assert "result = await run_graph(g)" in code

    # Check return statement
    assert "outputs = {}" in code
    assert "outputs['result'] = result[n1]" in code
    assert "return outputs" in code


def test_workflow_to_dsl_multiple_inputs_outputs():
    """Test workflow with multiple inputs and outputs."""
    nodes = [
        Node(
            id="1",
            type=InputNode.get_node_type(),
            data={"name": "num1", "value": 10},
        ),
        Node(
            id="2",
            type=InputNode.get_node_type(),
            data={"name": "num2", "value": 20},
        ),
        Node(
            id="3",
            type="test.math.Add",
            data={},
        ),
        Node(
            id="4",
            type=OutputNode.get_node_type(),
            data={"name": "sum"},
        ),
        Node(
            id="5",
            type=OutputNode.get_node_type(),
            data={"name": "first_num"},
        ),
    ]
    edges = [
        Edge(
            id="e1",
            source="1",
            sourceHandle="output",
            target="3",
            targetHandle="a",
        ),
        Edge(
            id="e2",
            source="2",
            sourceHandle="output",
            target="3",
            targetHandle="b",
        ),
        Edge(
            id="e3",
            source="3",
            sourceHandle="result",
            target="4",
            targetHandle="value",
        ),
        Edge(
            id="e4",
            source="1",
            sourceHandle="output",
            target="5",
            targetHandle="value",
        ),
    ]
    wf = Workflow(
        id="w2",
        user_id="u1",
        name="Math Operations",
        graph={
            "nodes": [n.model_dump() for n in nodes],
            "edges": [e.model_dump() for e in edges],
        },
    )

    code = workflow_to_dsl(wf)

    # Check function signature with multiple inputs
    assert "async def math_operations(num1, num2):" in code

    # Check imports
    assert "from nodetool.dsl.test.math import Add" in code

    # Check node definitions with proper connections
    assert re.search(r"n0 = .*Input\(.*value=num1", code)
    assert re.search(r"n1 = .*Input\(.*value=num2", code)
    assert "n2 = Add(a=(n0, 'output'), b=(n1, 'output'))" in code

    # Check multiple outputs
    assert "outputs['sum'] = result[n3]" in code
    assert "outputs['first_num'] = result[n4]" in code


def test_workflow_to_dsl_no_inputs_outputs():
    """Test workflow with no explicit input/output nodes."""
    nodes = [
        Node(
            id="1",
            type="test.constant.Number",
            data={"value": 42},
        ),
        Node(
            id="2",
            type="test.math.Square",
            data={},
        ),
    ]
    edges = [
        Edge(
            id="e1",
            source="1",
            sourceHandle="output",
            target="2",
            targetHandle="input",
        )
    ]
    wf = Workflow(
        id="w3",
        user_id="u1",
        name="No IO Workflow",
        graph={
            "nodes": [n.model_dump() for n in nodes],
            "edges": [e.model_dump() for e in edges],
        },
    )

    code = workflow_to_dsl(wf)

    # Function should have no parameters
    assert "async def no_io_workflow():" in code

    # Should return the raw result
    assert "return result" in code
    assert "outputs = {}" not in code


def test_workflow_to_dsl_complex_properties():
    """Test workflow with nodes having complex properties."""
    nodes = [
        Node(
            id="1",
            type="test.llm.ChatGPT",
            data={
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 100,
                "system_prompt": "You are a helpful assistant",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        ),
        Node(
            id="2",
            type=OutputNode.get_node_type(),
            data={"name": "response"},
        ),
    ]
    edges = [
        Edge(
            id="e1",
            source="1",
            sourceHandle="output",
            target="2",
            targetHandle="value",
        )
    ]
    wf = Workflow(
        id="w4",
        user_id="u1",
        name="LLM Workflow",
        graph={
            "nodes": [n.model_dump() for n in nodes],
            "edges": [e.model_dump() for e in edges],
        },
    )

    code = workflow_to_dsl(wf)

    # Check complex property serialization
    assert "model='gpt-4'" in code
    assert "temperature=0.7" in code
    assert "max_tokens=100" in code
    assert "system_prompt='You are a helpful assistant'" in code
    assert "messages=[{'role': 'user', 'content': 'Hello'}]" in code


def test_workflow_to_dsl_special_characters_in_name():
    """Test workflow name sanitization for function names."""
    wf = Workflow(
        id="w5",
        user_id="u1",
        name="My-Special Workflow! (v2.0)",
        graph={
            "nodes": [],
            "edges": [],
        },
    )

    code = workflow_to_dsl(wf)

    # Name should be sanitized to valid Python identifier
    assert "async def my_special_workflow!_(v2.0)():" not in code
    assert (
        "async def my_special_workflow__v2_0_():" in code
        or "my_special" in code.lower()
    )


def test_workflow_to_dsl_unnamed_inputs():
    """Test workflow with input nodes that don't have explicit names."""
    nodes = [
        Node(
            id="1",
            type=InputNode.get_node_type(),
            data={"value": "default"},  # No name property
        ),
        Node(
            id="2",
            type=OutputNode.get_node_type(),
            data={},  # No name property
        ),
    ]
    edges = [
        Edge(
            id="e1",
            source="1",
            sourceHandle="output",
            target="2",
            targetHandle="value",
        )
    ]
    wf = Workflow(
        id="w6",
        user_id="u1",
        name="Unnamed IO",
        graph={
            "nodes": [n.model_dump() for n in nodes],
            "edges": [e.model_dump() for e in edges],
        },
    )

    code = workflow_to_dsl(wf)

    # Should generate default parameter names
    assert "async def unnamed_io(input_n0):" in code
    assert "value=input_n0" in code
    assert "outputs['output_n1']" in code


def test_workflow_to_dsl_empty_workflow():
    """Test empty workflow generation."""
    wf = Workflow(
        id="w7",
        user_id="u1",
        name="Empty",
        graph={
            "nodes": [],
            "edges": [],
        },
    )

    code = workflow_to_dsl(wf)

    # Should still generate valid function
    assert "async def empty():" in code
    assert "g = graph()" in code
    assert "return result" in code


def test_workflow_to_dsl_imports_deduplication():
    """Test that imports are properly deduplicated."""
    nodes = [
        Node(
            id="1",
            type="test.math.Add",
            data={},
        ),
        Node(
            id="2",
            type="test.math.Add",
            data={},
        ),
        Node(
            id="3",
            type="test.math.Multiply",
            data={},
        ),
    ]
    wf = Workflow(
        id="w8",
        user_id="u1",
        name="Math Ops",
        graph={
            "nodes": [n.model_dump() for n in nodes],
            "edges": [],
        },
    )

    code = workflow_to_dsl(wf)

    # Should have single import with both classes
    assert code.count("from nodetool.dsl.test.math import") == 1
    assert "Add, Multiply" in code or "Multiply, Add" in code


def test_workflow_to_dsl_executable():
    """Test that the generated code can be executed."""
    # Create a simple workflow with inputs and outputs
    nodes = [
        Node(
            id="1",
            type=InputNode.get_node_type(),
            data={"name": "x", "value": 5},
        ),
        Node(
            id="2",
            type=InputNode.get_node_type(),
            data={"name": "y", "value": 3},
        ),
        Node(
            id="3",
            type=OutputNode.get_node_type(),
            data={"name": "sum"},
        ),
        Node(
            id="4",
            type=OutputNode.get_node_type(),
            data={"name": "product"},
        ),
    ]
    edges = [
        Edge(
            id="e1",
            source="1",
            sourceHandle="output",
            target="3",
            targetHandle="value",
        ),
        Edge(
            id="e2",
            source="2",
            sourceHandle="output",
            target="4",
            targetHandle="value",
        ),
    ]
    wf = Workflow(
        id="w9",
        user_id="u1",
        name="Test Executable",
        description="Test workflow for execution",
        graph={
            "nodes": [n.model_dump() for n in nodes],
            "edges": [e.model_dump() for e in edges],
        },
    )

    code = workflow_to_dsl(wf)

    # Mock the required imports
    mock_graph = Mock()
    mock_run_graph = AsyncMock()

    # Create mock node classes
    mock_input_class = Mock()
    mock_output_class = Mock()

    # Create mock node instances
    mock_n0 = Mock()
    mock_n1 = Mock()
    mock_n2 = Mock()
    mock_n3 = Mock()

    # Set up the mock classes to return our mock instances
    mock_input_class.side_effect = [mock_n0, mock_n1]
    mock_output_class.side_effect = [mock_n2, mock_n3]

    # Set up mock_graph to return a graph object
    mock_graph_instance = Mock()
    mock_graph.return_value = mock_graph_instance

    # Set up mock_run_graph to return results
    mock_result = {mock_n2: 8, mock_n3: 15}  # sum result  # product result
    mock_run_graph.return_value = mock_result

    # Create a namespace for exec
    namespace = {
        "graph": mock_graph,
        "run_graph": mock_run_graph,
        "Input": mock_input_class,
        "Output": mock_output_class,
    }

    # Remove import lines from code before execution
    code_lines = code.split("\n")
    code_without_imports = "\n".join(
        line
        for line in code_lines
        if not line.startswith("from ") and not line.startswith("import ")
    )

    # Execute the generated code
    exec(code_without_imports, namespace)

    # Get the generated function
    test_executable = namespace["test_executable"]

    # Call the function
    result = asyncio.run(test_executable(10, 20))

    # Verify the function was called correctly
    assert mock_input_class.call_count == 2
    assert mock_output_class.call_count == 2

    # Verify the inputs were set correctly
    mock_input_class.assert_any_call(value=10, name="x")
    mock_input_class.assert_any_call(value=20, name="y")

    # Verify outputs were connected correctly
    mock_output_class.assert_any_call(value=(mock_n0, "output"), name="sum")
    mock_output_class.assert_any_call(value=(mock_n1, "output"), name="product")

    # Verify graph was created with all nodes
    mock_graph.assert_called_once_with(mock_n0, mock_n1, mock_n2, mock_n3)

    # Verify run_graph was called
    mock_run_graph.assert_called_once_with(mock_graph_instance)

    # Verify the result
    assert result == {"sum": 8, "product": 15}


def test_workflow_to_dsl_executable_with_processing():
    """Test generated code with actual processing nodes."""
    nodes = [
        Node(
            id="1",
            type=InputNode.get_node_type(),
            data={"name": "a", "value": 10},
        ),
        Node(
            id="2",
            type=InputNode.get_node_type(),
            data={"name": "b", "value": 5},
        ),
        Node(
            id="3",
            type="test.math.Add",
            data={},
        ),
        Node(
            id="4",
            type=OutputNode.get_node_type(),
            data={"name": "result"},
        ),
    ]
    edges = [
        Edge(
            id="e1",
            source="1",
            sourceHandle="output",
            target="3",
            targetHandle="a",
        ),
        Edge(
            id="e2",
            source="2",
            sourceHandle="output",
            target="3",
            targetHandle="b",
        ),
        Edge(
            id="e3",
            source="3",
            sourceHandle="output",
            target="4",
            targetHandle="value",
        ),
    ]
    wf = Workflow(
        id="w10",
        user_id="u1",
        name="Add Function",
        graph={
            "nodes": [n.model_dump() for n in nodes],
            "edges": [e.model_dump() for e in edges],
        },
    )

    code = workflow_to_dsl(wf)

    # Create mocks
    mock_graph = Mock()
    mock_run_graph = AsyncMock()
    mock_input_class = Mock()
    mock_output_class = Mock()
    mock_add_class = Mock()

    # Create node instances
    mock_n0 = Mock()
    mock_n1 = Mock()
    mock_n2 = Mock()
    mock_n3 = Mock()

    mock_input_class.side_effect = [mock_n0, mock_n1]
    mock_add_class.return_value = mock_n2
    mock_output_class.return_value = mock_n3

    mock_graph_instance = Mock()
    mock_graph.return_value = mock_graph_instance

    # Simulate the add operation result
    mock_result = {mock_n3: 15}
    mock_run_graph.return_value = mock_result

    # Create namespace
    namespace = {
        "graph": mock_graph,
        "run_graph": mock_run_graph,
        "Input": mock_input_class,
        "Output": mock_output_class,
        "Add": mock_add_class,
    }

    # Remove import lines from code before execution
    code_lines = code.split("\n")
    code_without_imports = "\n".join(
        line
        for line in code_lines
        if not line.startswith("from ") and not line.startswith("import ")
    )

    # Execute the code
    exec(code_without_imports, namespace)

    # Get and run the function
    add_function = namespace["add_function"]
    result = asyncio.run(add_function(20, 30))

    # Verify Add node was created with correct connections
    mock_add_class.assert_called_once_with(a=(mock_n0, "output"), b=(mock_n1, "output"))

    # Verify the result
    assert result == {"result": 15}
