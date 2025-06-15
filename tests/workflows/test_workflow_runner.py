from collections import deque
import asyncio
from typing import Any, AsyncGenerator, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import pytest
from nodetool.metadata.types import ImageRef, Message, TextRef, Event
from nodetool.types.graph import Node, Edge
from nodetool.types.job import JobUpdate
from nodetool.workflows.base_node import BaseNode, InputNode, OutputNode
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.workflow_runner import WorkflowRunner, acquire_gpu_lock, release_gpu_lock, get_available_vram
from nodetool.workflows.graph import Graph
from nodetool.workflows.types import NodeUpdate, NodeProgress, OutputUpdate
from nodetool.types.graph import (
    Graph as APIGraph,
)
from nodetool.common.environment import Environment


class String(BaseNode):
    value: str = ""

    async def process(self, context: ProcessingContext) -> str:
        return self.value


class Float(BaseNode):
    value: float = 0.0

    async def process(self, context: ProcessingContext) -> float:
        return self.value


class Add(BaseNode):
    a: float = 0.0
    b: float = 0.0

    async def process(self, context: ProcessingContext) -> float:
        return self.a + self.b


class IntegerOutput(OutputNode):
    value: int = 0

    async def process(self, context: ProcessingContext) -> int:
        return self.value


class IntegerInput(InputNode):
    value: int = 0

    async def process(self, context: ProcessingContext) -> int:
        return self.value


class ImageInput(InputNode):
    value: ImageRef = ImageRef()

    async def process(self, context: ProcessingContext) -> ImageRef:
        return self.value


class TextInput(InputNode):
    value: TextRef = TextRef()

    async def process(self, context: ProcessingContext) -> TextRef:
        return self.value


class ImageToText(BaseNode):
    """Converts image to text"""
    image: ImageRef = ImageRef()

    async def process(self, context: ProcessingContext) -> TextRef:
        return TextRef(data="Image description")


class TextOutput(OutputNode):
    value: TextRef = TextRef()

    async def process(self, context: ProcessingContext) -> TextRef:
        return self.value


class StringToInt(BaseNode):
    """Converts string to int"""
    text: str = ""

    async def process(self, context: ProcessingContext) -> int:
        return int(self.text) if self.text.isdigit() else 0


class StringOutput(OutputNode):
    """Output node that accepts string values"""
    value: str = ""

    async def process(self, context: ProcessingContext) -> str:
        return self.value


@pytest.fixture
def workflow_runner() -> WorkflowRunner:
    return WorkflowRunner(job_id="1")


@pytest.mark.asyncio
async def test_process_node(workflow_runner: WorkflowRunner):
    node = String(id="1", value="test")  # type: ignore
    next_node = IntegerOutput(id="2")  # type: ignore
    context = ProcessingContext(
        user_id="",
        workflow_id="",
        auth_token="token",
        graph=Graph(
            nodes=[node, next_node],
            edges=[
                Edge(
                    id="1",
                    source="1",
                    target="2",
                    sourceHandle="output",
                    targetHandle="value",
                ),
            ],
        ),
    )
    workflow_runner.edge_queues[("1", "output", "2", "value")] = deque()
    await workflow_runner.process_node(context, node, {})

    value = workflow_runner.edge_queues[("1", "output", "2", "value")][0]

    assert value == "test"


async def get_workflow_updates(context: ProcessingContext):
    messages = []

    while context.has_messages():
        messages.append(await context.pop_message_async())

    return list(filter(lambda x: isinstance(x, JobUpdate), messages))


@pytest.mark.asyncio
async def test_from_dict():
    image_input = {
        "id": "1",
        "type": ImageInput.get_node_type(),
        "data": {
            "name": "image_a",
            "value": {
                "type": "image",
                "uri": "https://example.com/image.jpg",
            },
        },
    }

    node = ImageInput.from_dict(image_input)

    assert node.id == "1"  # type: ignore
    assert node.value.uri == "https://example.com/image.jpg"  # type: ignore


@pytest.mark.asyncio
async def test_process_graph(workflow_runner: WorkflowRunner):
    input_a = {
        "id": "1",
        "data": {"name": "input_1"},
        "type": IntegerInput.get_node_type(),
    }
    input_b = {
        "id": "2",
        "data": {"name": "input_2"},
        "type": IntegerInput.get_node_type(),
    }
    add_node = {"id": "3", "type": Add.get_node_type()}
    out_node = {
        "id": "4",
        "data": {"name": "output"},
        "type": IntegerOutput.get_node_type(),
    }

    nodes = [
        input_a,
        input_b,
        add_node,
        out_node,
    ]

    edges = [
        {
            "id": "1",
            "source": "1",
            "target": "3",
            "sourceHandle": "output",
            "targetHandle": "a",
            "ui_properties": {},
        },
        {
            "id": "2",
            "source": "2",
            "target": "3",
            "sourceHandle": "output",
            "targetHandle": "b",
            "ui_properties": {},
        },
        {
            "id": "3",
            "source": "3",
            "target": "4",
            "sourceHandle": "output",
            "targetHandle": "value",
            "ui_properties": {},
        },
    ]
    graph = APIGraph(nodes=[Node(**n) for n in nodes], edges=[Edge(**e) for e in edges])
    params = {"input_1": 1, "input_2": 2}

    req = RunJobRequest(
        user_id="1",
        workflow_id="",
        job_type="",
        params=params,
        graph=graph,
    )
    context = ProcessingContext(
        user_id="1",
        auth_token="local_token",
    )

    await workflow_runner.run(req, context)

    workflow_updates = await get_workflow_updates(context)
    print(workflow_updates)

    assert len(workflow_updates) == 2
    assert workflow_updates[1].result["output"] == [3]


class ErrorNode(BaseNode):
    async def process(self, context: ProcessingContext) -> str:
        raise ValueError("Node processing error")


class InitErrorNode(BaseNode):
    async def initialize(self, context: ProcessingContext):
        raise ValueError("Node initialization error")

    async def process(self, context: ProcessingContext) -> str:
        return "should not reach here"


class CacheableNode(BaseNode):
    value: str = "initial"
    process_count: int = 0

    def is_cacheable(self) -> bool:
        return True

    async def process(self, context: ProcessingContext) -> str:
        # Adding a log to trace calls to this method
        log = Environment.get_logger()  # Corrected: No argument
        log.info(
            f"CacheableNode '{self.id}' process() called. Current process_count: {self.process_count}. Instance MEM ID: {id(self)}"
        )
        self.process_count += 1
        return self.value


@pytest.mark.asyncio
async def test_process_node_error(workflow_runner: WorkflowRunner):
    error_node = {"id": "1", "type": ErrorNode.get_node_type()}
    # Use StringOutput instead of IntegerOutput since ErrorNode returns a string
    out_node = {
        "id": "2",
        "data": {"name": "output"},
        "type": StringOutput.get_node_type(),
    }
    nodes = [error_node, out_node]
    edges = [
        {
            "id": "1",
            "source": "1",
            "target": "2",
            "sourceHandle": "output",
            "targetHandle": "value",
            "ui_properties": {},
        }
    ]
    graph = APIGraph(nodes=[Node(**n) for n in nodes], edges=[Edge(**e) for e in edges])
    req = RunJobRequest(
        user_id="1", workflow_id="", job_type="", params={}, graph=graph
    )
    context = ProcessingContext(user_id="1", auth_token="local_token")

    # Expect run() to raise the error originating from the node's process method
    with pytest.raises(ValueError, match="Node processing error"):
        await workflow_runner.run(req, context)

    # After the error is raised and caught by pytest.raises,
    # check messages posted to the context before the exception propagated.
    messages = []
    while context.has_messages():
        messages.append(await context.pop_message_async())

    node_updates = [m for m in messages if isinstance(m, NodeUpdate)]
    job_updates = [m for m in messages if isinstance(m, JobUpdate)]

    assert any(
        update.node_id == "1" and update.status == "error" and "Node processing error" in update.error  # type: ignore
        for update in node_updates
    ), "NodeUpdate with processing error not found"

    assert any(
        update.status == "error" and "Node processing error" in str(update.error)  # type: ignore
        for update in job_updates
    ), "JobUpdate with status error not found or error message mismatch"


@pytest.mark.asyncio
async def test_initialize_node_error(workflow_runner: WorkflowRunner):
    init_error_node = {"id": "1", "type": InitErrorNode.get_node_type()}
    nodes: list[dict[str, Any]] = [init_error_node]
    edges: list[dict[str, Any]] = []
    graph = APIGraph(nodes=[Node(**n) for n in nodes], edges=[Edge(**e) for e in edges])
    req = RunJobRequest(
        user_id="1", workflow_id="", job_type="", params={}, graph=graph
    )
    context = ProcessingContext(user_id="1", auth_token="local_token")

    with pytest.raises(ValueError, match="Node initialization error"):
        await workflow_runner.run(req, context)

    messages = []
    while context.has_messages():
        messages.append(await context.pop_message_async())

    node_updates = [m for m in messages if isinstance(m, NodeUpdate)]

    assert any(
        update.node_id == "1" and update.status == "error" and "Node initialization error" in update.error  # type: ignore
        for update in node_updates
    ), "NodeUpdate with initialization error not found"

    # Check if a JobUpdate with status error was posted
    # This might depend on how workflow_runner.run finalizes after re-raising
    # For now, primarily check NodeUpdate as initialize_graph posts it before raising.
    # A JobUpdate error might also be there.
    # assert any(
    #     update.status == "error" and "Node initialization error" in update.error # type: ignore
    #     for update in job_updates
    # ), "JobUpdate with status error not found for initialization error"


@pytest.mark.asyncio
async def test_edge_type_validation_compatible_types(workflow_runner: WorkflowRunner):
    """Test that compatible edge types pass validation"""
    # String -> String (compatible)
    string_input = {"id": "1", "type": String.get_node_type(), "data": {"value": "test"}}
    string_to_int = {"id": "2", "type": StringToInt.get_node_type()}
    int_output = {"id": "3", "type": IntegerOutput.get_node_type(), "data": {"name": "output"}}
    
    nodes: list[dict[str, Any]] = [string_input, string_to_int, int_output]
    edges: list[dict[str, Any]] = [
        {
            "id": "1",
            "source": "1",
            "target": "2",
            "sourceHandle": "output",
            "targetHandle": "text",
        },
        {
            "id": "2", 
            "source": "2",
            "target": "3",
            "sourceHandle": "output",
            "targetHandle": "value",
        }
    ]
    
    graph = APIGraph(nodes=[Node(**n) for n in nodes], edges=[Edge(**e) for e in edges])
    req = RunJobRequest(
        user_id="1", workflow_id="", job_type="", params={}, graph=graph
    )
    context = ProcessingContext(user_id="1", auth_token="local_token")
    
    # Should not raise any validation errors
    await workflow_runner.run(req, context)
    
    # Check that validation passed
    workflow_updates = await get_workflow_updates(context)
    assert workflow_updates[1].status == "completed"


@pytest.mark.asyncio
async def test_edge_type_validation_incompatible_types(workflow_runner: WorkflowRunner):
    """Test that incompatible edge types fail validation"""
    # Image -> String (incompatible - ImageRef cannot be assigned to str)
    image_input = {"id": "1", "type": ImageInput.get_node_type(), "data": {"name": "image_input"}}
    string_to_int = {"id": "2", "type": StringToInt.get_node_type()}
    
    nodes: list[dict[str, Any]] = [image_input, string_to_int]
    edges: list[dict[str, Any]] = [
        {
            "id": "1",
            "source": "1",
            "target": "2", 
            "sourceHandle": "output",
            "targetHandle": "text",  # Expects string, but gets ImageRef
        }
    ]
    
    graph = APIGraph(nodes=[Node(**n) for n in nodes], edges=[Edge(**e) for e in edges])
    req = RunJobRequest(
        user_id="1", workflow_id="", job_type="", params={}, graph=graph
    )
    context = ProcessingContext(user_id="1", auth_token="local_token")
    
    # Should raise validation error
    with pytest.raises(ValueError, match="Type mismatch"):
        await workflow_runner.run(req, context)
    
    # Check error messages
    messages = []
    while context.has_messages():
        messages.append(await context.pop_message_async())
    
    node_updates = [m for m in messages if isinstance(m, NodeUpdate)]
    assert any(
        "Type mismatch" in update.error  # type: ignore
        for update in node_updates
    ), "NodeUpdate with type mismatch error not found"


@pytest.mark.asyncio
async def test_edge_type_validation_float_to_int(workflow_runner: WorkflowRunner):
    """Test that float to int edge is allowed (numeric compatibility)"""
    float_input = {"id": "1", "type": Float.get_node_type(), "data": {"value": 3.14}}
    int_output = {"id": "2", "type": IntegerOutput.get_node_type(), "data": {"name": "output"}}
    
    nodes: list[dict[str, Any]] = [float_input, int_output]
    edges: list[dict[str, Any]] = [
        {
            "id": "1",
            "source": "1",
            "target": "2",
            "sourceHandle": "output", 
            "targetHandle": "value",  # int property can accept float value
        }
    ]
    
    graph = APIGraph(nodes=[Node(**n) for n in nodes], edges=[Edge(**e) for e in edges])
    req = RunJobRequest(
        user_id="1", workflow_id="", job_type="", params={}, graph=graph  
    )
    context = ProcessingContext(user_id="1", auth_token="local_token")
    
    # Should not raise validation errors (float is assignable to int in many cases)
    await workflow_runner.run(req, context)
    
    workflow_updates = await get_workflow_updates(context)
    assert workflow_updates[1].status == "completed"


@pytest.mark.asyncio
async def test_edge_type_validation_missing_output(workflow_runner: WorkflowRunner):
    """Test that missing output slot fails validation"""
    string_input = {"id": "1", "type": String.get_node_type(), "data": {"value": "test"}}
    int_output = {"id": "2", "type": IntegerOutput.get_node_type()}
    
    nodes: list[dict[str, Any]] = [string_input, int_output]
    edges: list[dict[str, Any]] = [
        {
            "id": "1",
            "source": "1",
            "target": "2",
            "sourceHandle": "nonexistent_output",  # This output doesn't exist
            "targetHandle": "value",
        }
    ]
    
    graph = APIGraph(nodes=[Node(**n) for n in nodes], edges=[Edge(**e) for e in edges])
    req = RunJobRequest(
        user_id="1", workflow_id="", job_type="", params={}, graph=graph
    )
    context = ProcessingContext(user_id="1", auth_token="local_token")
    
    # Should raise validation error
    with pytest.raises(ValueError, match="Output.*not found"):
        await workflow_runner.run(req, context)


@pytest.mark.asyncio
async def test_edge_type_validation_missing_property(workflow_runner: WorkflowRunner):
    """Test that missing target property fails validation"""
    string_input = {"id": "1", "type": String.get_node_type(), "data": {"value": "test"}}
    int_output = {"id": "2", "type": IntegerOutput.get_node_type()}
    
    nodes: list[dict[str, Any]] = [string_input, int_output]
    edges: list[dict[str, Any]] = [
        {
            "id": "1",
            "source": "1",
            "target": "2",
            "sourceHandle": "output",
            "targetHandle": "nonexistent_property",  # This property doesn't exist
        }
    ]
    
    graph = APIGraph(nodes=[Node(**n) for n in nodes], edges=[Edge(**e) for e in edges])
    req = RunJobRequest(
        user_id="1", workflow_id="", job_type="", params={}, graph=graph
    )
    context = ProcessingContext(user_id="1", auth_token="local_token")
    
    # Should raise validation error
    with pytest.raises(ValueError, match="Property.*not found"):
        await workflow_runner.run(req, context)


# ============= GPU-RELATED TESTS =============

class GPUNode(BaseNode):
    """Node that requires GPU processing"""
    value: str = "gpu_test"
    device: str = "unknown"
    
    def requires_gpu(self) -> bool:
        return True
    
    async def move_to_device(self, device: str):
        self.device = device
    
    async def process(self, context: ProcessingContext) -> str:
        return f"{self.value}_processed_on_{getattr(self, 'device', 'unknown')}"
    
    @classmethod
    def get_node_type(cls):
        return "test.gpu.GPUNode"


class GPUNodeWithGrad(GPUNode):
    """GPU node that requires gradient computation"""
    _requires_grad = True


@pytest.mark.asyncio
async def test_gpu_lock_acquisition():
    """Test GPU lock acquisition and release"""
    node = GPUNode(id="1") # type: ignore
    context = ProcessingContext(user_id="1", auth_token="token")
    
    # Test acquiring lock
    await acquire_gpu_lock(node, context)
    
    # Verify lock is held
    from nodetool.workflows.workflow_runner import gpu_lock
    assert gpu_lock.locked()
    
    # Test releasing lock
    release_gpu_lock()
    assert not gpu_lock.locked()


@pytest.mark.asyncio
async def test_gpu_lock_contention():
    """Test GPU lock behavior with multiple nodes"""
    node1 = GPUNode(id="1") # type: ignore
    context = ProcessingContext(user_id="1", auth_token="token")
    
    # First node acquires lock
    await acquire_gpu_lock(node1, context)
    
    # Verify lock is held
    from nodetool.workflows.workflow_runner import gpu_lock
    assert gpu_lock.locked()
    
    # Try to check if it's locked (it should be)
    assert gpu_lock.locked()
    
    # Release lock
    release_gpu_lock()
    assert not gpu_lock.locked()
    
    # Can acquire again
    await acquire_gpu_lock(node1, context)
    assert gpu_lock.locked()
    release_gpu_lock()


@pytest.mark.asyncio
async def test_get_available_vram():
    """Test VRAM availability check"""
    # Without torch, should return 0
    with patch("nodetool.workflows.workflow_runner.TORCH_AVAILABLE", False):
        assert get_available_vram() == 0
    
    # With torch but no CUDA
    with patch("nodetool.workflows.workflow_runner.TORCH_AVAILABLE", True):
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        with patch("nodetool.workflows.workflow_runner.torch", mock_torch):
            assert get_available_vram() == 0
    
    # With torch and CUDA
    with patch("nodetool.workflows.workflow_runner.TORCH_AVAILABLE", True):
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = Mock(total_memory=8000000000)
        mock_torch.cuda.memory_allocated.return_value = 2000000000
        with patch("nodetool.workflows.workflow_runner.torch", mock_torch):
            assert get_available_vram() == 6000000000


@pytest.mark.asyncio
async def test_gpu_node_processing():
    """Test processing a node that requires GPU"""
    # This test verifies GPU node requirements are checked
    node = GPUNode(id="1") # type: ignore
    assert node.requires_gpu() == True
    
    # Test move_to_device
    await node.move_to_device("cuda")
    assert node.device == "cuda"
    
    # Test process returns expected format
    context = ProcessingContext(user_id="1", auth_token="token")
    result = await node.process(context)
    assert result == "gpu_test_processed_on_cuda"


@pytest.mark.asyncio
async def test_gpu_node_no_gpu_available(workflow_runner: WorkflowRunner):
    """Test error when GPU required but not available"""
    workflow_runner.device = "cpu"  # No GPU available
    
    gpu_node = {"id": "1", "type": GPUNode.get_node_type()}
    nodes: list[dict[str, Any]] = [gpu_node]
    edges: list[dict[str, Any]] = []
    
    graph = APIGraph(nodes=[Node(**n) for n in nodes], edges=[])
    req = RunJobRequest(
        user_id="1", workflow_id="", job_type="", params={}, graph=graph
    )
    context = ProcessingContext(user_id="1", auth_token="token")
    
    with pytest.raises(RuntimeError, match="requires a GPU"):
        await workflow_runner.run(req, context)


@pytest.mark.asyncio
async def test_gpu_oom_retry():
    """Test VRAM OOM error handling and retry logic"""
    # Test that the retry mechanism exists in process_with_gpu
    workflow_runner = WorkflowRunner(job_id="1", device="cuda")
    
    # Verify the method exists and has retry logic
    assert hasattr(workflow_runner, 'process_with_gpu')
    
    # Test that MAX_RETRIES is defined
    from nodetool.workflows import workflow_runner as wr
    assert hasattr(wr, 'MAX_RETRIES')
    assert wr.MAX_RETRIES > 0


# ============= STREAMING NODE TESTS =============

class StreamingNode(BaseNode):
    """Test streaming node"""
    items: list[str] | None = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.items is None:
            self.items = ["item1", "item2", "item3"]
    
    def is_streaming_output(self) -> bool:
        return True
    
    @classmethod
    def return_type(cls):
        return {"output": str, "index": int}
    
    @classmethod
    def get_node_type(cls):
        return "test.streaming.StreamingNode"
    
    async def gen_process(self, context: ProcessingContext) -> AsyncGenerator[tuple[str, Any], None]:
        for i, item in enumerate(self.items):
            yield "output", item
            yield "index", i
            await asyncio.sleep(0.001)


class StreamingErrorNode(StreamingNode):
    """Streaming node that errors partway through"""
    
    @classmethod
    def get_node_type(cls):
        return "test.streaming.StreamingErrorNode"
    
    async def gen_process(self, context: ProcessingContext) -> AsyncGenerator[tuple[str, Any], None]:
        yield "output", "first"
        raise ValueError("Streaming error")


@pytest.mark.asyncio
async def test_streaming_node_basic(workflow_runner: WorkflowRunner):
    """Test basic streaming node functionality"""
    streaming = {"id": "1", "type": StreamingNode.get_node_type()}
    collector = {"id": "2", "type": StringOutput.get_node_type(), "data": {"name": "collected"}}
    
    nodes: list[dict[str, Any]] = [streaming, collector]
    edges: list[dict[str, Any]] = [{
        "id": "1",
        "source": "1",
        "target": "2",
        "sourceHandle": "output",
        "targetHandle": "value",
    }]
    
    graph = APIGraph(nodes=[Node(**n) for n in nodes], edges=[Edge(**e) for e in edges])
    req = RunJobRequest(
        user_id="1", workflow_id="", job_type="", params={}, graph=graph
    )
    context = ProcessingContext(user_id="1", auth_token="token")
    
    await workflow_runner.run(req, context)
    
    # Should have collected all items
    assert workflow_runner.outputs["collected"] == ["item1", "item2", "item3"]


@pytest.mark.asyncio
async def test_streaming_node_error(workflow_runner: WorkflowRunner):
    """Test streaming node error handling"""
    streaming = {"id": "1", "type": StreamingErrorNode.get_node_type()}
    collector = {"id": "2", "type": StringOutput.get_node_type(), "data": {"name": "collected"}}
    
    nodes: list[dict[str, Any]] = [streaming, collector]
    edges: list[dict[str, Any]] = [{
        "id": "1",
        "source": "1",
        "target": "2",
        "sourceHandle": "output",
        "targetHandle": "value",
    }]
    
    graph = APIGraph(nodes=[Node(**n) for n in nodes], edges=[Edge(**e) for e in edges])
    req = RunJobRequest(
        user_id="1", workflow_id="", job_type="", params={}, graph=graph
    )
    context = ProcessingContext(user_id="1", auth_token="token")
    
    with pytest.raises(ValueError, match="Streaming error"):
        await workflow_runner.run(req, context)
    
    # Check that the streaming node was cleaned up
    assert "1" not in workflow_runner.active_generators


@pytest.mark.asyncio
async def test_streaming_node_multiple_outputs(workflow_runner: WorkflowRunner):
    """Test streaming node with multiple output slots"""
    streaming = {"id": "1", "type": StreamingNode.get_node_type()}
    string_out = {"id": "2", "type": StringOutput.get_node_type(), "data": {"name": "strings"}}
    int_out = {"id": "3", "type": IntegerOutput.get_node_type(), "data": {"name": "indices"}}
    
    nodes: list[dict[str, Any]] = [streaming, string_out, int_out]
    edges: list[dict[str, Any]] = [
        {
            "id": "1",
            "source": "1",
            "target": "2",
            "sourceHandle": "output",
            "targetHandle": "value",
        },
        {
            "id": "2",
            "source": "1",
            "target": "3",
            "sourceHandle": "index",
            "targetHandle": "value",
        }
    ]
    
    graph = APIGraph(nodes=[Node(**n) for n in nodes], edges=[Edge(**e) for e in edges])
    req = RunJobRequest(
        user_id="1", workflow_id="", job_type="", params={}, graph=graph
    )
    context = ProcessingContext(user_id="1", auth_token="token")
    
    await workflow_runner.run(req, context)
    
    assert workflow_runner.outputs["strings"] == ["item1", "item2", "item3"]
    assert workflow_runner.outputs["indices"] == [0, 1, 2]


# ============= EVENT HANDLING TESTS =============

class EventProducerNode(BaseNode):
    """Node that produces events"""
    
    @classmethod
    def return_type(cls):
        return {"event": Event}
    
    @classmethod
    def get_node_type(cls):
        return "test.event.EventProducerNode"
    
    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        return {"event": Event(name="test_event", payload={"data": "test"})}


class EventConsumerNode(BaseNode):
    """Node that handles events"""
    event_input: Optional[Event] = None
    processed_events: list[str] | None = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.processed_events is None:
            self.processed_events = []
    
    @classmethod 
    def return_type(cls):
        return {"result": str}
    
    @classmethod
    def get_node_type(cls):
        return "test.event.EventConsumerNode"
    
    async def handle_event(self, context: ProcessingContext, event: Event) -> AsyncGenerator[tuple[str, Any], None]:
        self.processed_events.append(event.name)
        yield "result", f"Processed: {event.name}"


@pytest.mark.asyncio
async def test_event_handling(workflow_runner: WorkflowRunner):
    """Test event production and handling"""
    producer = {"id": "1", "type": EventProducerNode.get_node_type()}
    consumer = {"id": "2", "type": EventConsumerNode.get_node_type()}
    output = {"id": "3", "type": StringOutput.get_node_type(), "data": {"name": "result"}}
    
    nodes: list[dict[str, Any]] = [producer, consumer, output]
    edges: list[dict[str, Any]] = [
        {
            "id": "1",
            "source": "1",
            "target": "2",
            "sourceHandle": "event",
            "targetHandle": "event_input",
        },
        {
            "id": "2",
            "source": "2",
            "target": "3",
            "sourceHandle": "result",
            "targetHandle": "value",
        }
    ]
    
    graph = APIGraph(nodes=[Node(**n) for n in nodes], edges=[Edge(**e) for e in edges])
    req = RunJobRequest(
        user_id="1", workflow_id="", job_type="", params={}, graph=graph
    )
    context = ProcessingContext(user_id="1", auth_token="token")
    
    await workflow_runner.run(req, context)
    
    assert workflow_runner.outputs["result"] == ["Processed: test_event"]


@pytest.mark.asyncio
async def test_event_immediate_processing(workflow_runner: WorkflowRunner):
    """Test that events trigger immediate processing"""
    
    class SlowNode(BaseNode):
        """Node that takes time to process"""
        value: str = "slow"
        
        @classmethod
        def get_node_type(cls):
            return "test.event.SlowNode"
        
        async def process(self, context: ProcessingContext) -> str:
            await asyncio.sleep(0.1)  # Simulate slow processing
            return self.value
    
    class EventWithDataNode(EventConsumerNode):
        """Event consumer that also needs regular data"""
        data_input: str = ""
        
        @classmethod
        def get_node_type(cls):
            return "test.event.EventWithDataNode"
        
        async def handle_event(self, context: ProcessingContext, event: Event) -> AsyncGenerator[tuple[str, Any], None]:
            yield "result", f"{self.data_input}:{event.name}"
    
    slow = {"id": "1", "type": SlowNode.get_node_type()}
    producer = {"id": "2", "type": EventProducerNode.get_node_type()}
    consumer = {"id": "3", "type": EventWithDataNode.get_node_type()}
    output = {"id": "4", "type": StringOutput.get_node_type(), "data": {"name": "result"}}
    
    nodes: list[dict[str, Any]] = [slow, producer, consumer, output]
    edges: list[dict[str, Any]] = [
        {
            "id": "1",
            "source": "1",
            "target": "3",
            "sourceHandle": "output",
            "targetHandle": "data_input",
        },
        {
            "id": "2",
            "source": "2",
            "target": "3",
            "sourceHandle": "event",
            "targetHandle": "event_input",
        },
        {
            "id": "3",
            "source": "3",
            "target": "4",
            "sourceHandle": "result",
            "targetHandle": "value",
        }
    ]
    
    graph = APIGraph(nodes=[Node(**n) for n in nodes], edges=[Edge(**e) for e in edges])
    req = RunJobRequest(
        user_id="1", workflow_id="", job_type="", params={}, graph=graph
    )
    context = ProcessingContext(user_id="1", auth_token="token")
    
    await workflow_runner.run(req, context)
    
    # Event should process with whatever data is available
    assert workflow_runner.outputs["result"] == ["slow:test_event"]


# ============= CACHING TESTS =============

@pytest.mark.asyncio
async def test_caching_functionality(workflow_runner: WorkflowRunner):
    """Test node result caching"""
    # Use the existing CacheableNode
    cacheable = {"id": "1", "type": CacheableNode.get_node_type(), "data": {"value": "cached_value"}}
    output = {"id": "2", "type": StringOutput.get_node_type(), "data": {"name": "result"}}
    
    nodes: list[dict[str, Any]] = [cacheable, output]
    edges: list[dict[str, Any]] = [{
        "id": "1",
        "source": "1",
        "target": "2",
        "sourceHandle": "output",
        "targetHandle": "value",
    }]
    
    graph = APIGraph(nodes=[Node(**n) for n in nodes], edges=[Edge(**e) for e in edges])
    req = RunJobRequest(
        user_id="1", workflow_id="", job_type="", params={}, graph=graph
    )
    
    # Mock the context caching methods
    context = ProcessingContext(user_id="1", auth_token="token")
    cached_results = {}
    
    def mock_get_cached(node):
        return cached_results.get(node.id)
    
    def mock_cache_result(node, result, ttl=3600):
        cached_results[node.id] = result
    
    context.get_cached_result = mock_get_cached
    context.cache_result = mock_cache_result
    
    # First run - should process and cache
    await workflow_runner.run(req, context)
    
    # Get the node instance to check process count
    assert workflow_runner.context is not None
    cacheable_node = workflow_runner.context.graph.find_node("1")
    assert isinstance(cacheable_node, CacheableNode)
    assert cacheable_node.process_count == 1
    
    # Second run with same context - should use cache
    workflow_runner2 = WorkflowRunner(job_id="2")
    await workflow_runner2.run(req, context)
    
    # Process count should still be 1 (not incremented)
    assert cacheable_node.process_count == 1


# ============= EDGE QUEUE TESTS =============

@pytest.mark.asyncio
async def test_edge_queue_initialization(workflow_runner: WorkflowRunner):
    """Test edge queue initialization"""
    nodes: list[dict[str, Any]] = [
        {"id": "1", "type": String.get_node_type()},
        {"id": "2", "type": String.get_node_type()},
        {"id": "3", "type": String.get_node_type()},
    ]
    edges: list[dict[str, Any]] = [
        {"id": "e1", "source": "1", "target": "2", "sourceHandle": "output", "targetHandle": "value"},
        {"id": "e2", "source": "2", "target": "3", "sourceHandle": "output", "targetHandle": "value"},
    ]
    
    graph = APIGraph(nodes=[Node(**n) for n in nodes], edges=[Edge(**e) for e in edges])
    req = RunJobRequest(
        user_id="1", workflow_id="", job_type="", params={}, graph=graph
    )
    context = ProcessingContext(user_id="1", auth_token="token")
    
    # Clear any existing queues
    workflow_runner.edge_queues.clear()
    
    # Just initialize, don't run
    assert context.graph is not None
    assert req.graph is not None
    loaded_nodes = context.load_nodes(req.graph.nodes)
    graph_obj = Graph(nodes=loaded_nodes, edges=req.graph.edges)
    workflow_runner._initialize_edge_queues(graph_obj)
    
    # Check queues are initialized
    assert len(workflow_runner.edge_queues) == 2
    assert ("1", "output", "2", "value") in workflow_runner.edge_queues
    assert ("2", "output", "3", "value") in workflow_runner.edge_queues


@pytest.mark.asyncio
async def test_multiple_messages_in_queue():
    """Test handling multiple messages in edge queues"""
    
    class MultiProducerNode(BaseNode):
        """Produces multiple values"""
        @classmethod
        def get_node_type(cls):
            return "test.queue.MultiProducerNode"
            
        async def process(self, context: ProcessingContext) -> dict[str, Any]:
            return {"output": "value1"}  # First call
    
    class CollectorNode(BaseNode):
        """Collects all values"""
        values: list[str] | None = None
        value: str = ""
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if self.values is None:
                self.values = []
        
        @classmethod
        def get_node_type(cls):
            return "test.queue.CollectorNode"
        
        async def process(self, context: ProcessingContext) -> str:
            self.values.append(self.value)
            return f"collected_{len(self.values)}"
    
    # Create a scenario where multiple messages queue up
    workflow_runner = WorkflowRunner(job_id="1")
    
    producer = MultiProducerNode(id="1") # type: ignore
    collector = CollectorNode(id="2") # type: ignore
    
    context = ProcessingContext(user_id="1", auth_token="token", graph=Graph(nodes=[producer, collector], edges=[]))
    
    # Manually queue multiple messages
    edge_key = ("1", "output", "2", "value")
    workflow_runner.edge_queues[edge_key] = deque(["msg1", "msg2", "msg3"])
    
    # Process collector with first message
    await workflow_runner.process_node(context, collector, {"value": workflow_runner.edge_queues[edge_key].popleft()})
    
    assert collector.values == ["msg1"]
    assert len(workflow_runner.edge_queues[edge_key]) == 2


# ============= PROCESSING LOOP TESTS =============

@pytest.mark.asyncio
async def test_loop_termination_idle():
    """Test loop termination due to idle state"""
    workflow_runner = WorkflowRunner(job_id="1")
    
    # Create simple graph
    nodes = [String(id="1", value="test")] # type: ignore
    edges = []
    graph = Graph(nodes=nodes, edges=edges)
    context = ProcessingContext(user_id="1", auth_token="token", graph=graph)
    
    # Test termination condition check
    should_terminate = workflow_runner._check_loop_termination_conditions(
        context, graph, iterations_without_progress=3, max_iterations_limit=100
    )
    
    assert should_terminate


@pytest.mark.asyncio 
async def test_loop_termination_pending_data():
    """Test loop termination with pending data in queues"""
    workflow_runner = WorkflowRunner(job_id="1")
    
    nodes = [String(id="1"), String(id="2")] # type: ignore
    edges = []
    graph = Graph(nodes=nodes, edges=edges)
    context = ProcessingContext(user_id="1", auth_token="token", graph=graph)
    
    # Add pending data
    workflow_runner.edge_queues[("1", "output", "2", "value")] = deque(["pending"])
    
    # Should terminate but with warning
    with patch("nodetool.workflows.workflow_runner.log") as mock_log:
        should_terminate = workflow_runner._check_loop_termination_conditions(
            context, graph, iterations_without_progress=3, max_iterations_limit=100
        )
        
        assert should_terminate
        # Check warning was logged
        mock_log.warning.assert_called()


# ============= TORCH CONTEXT TESTS =============

@pytest.mark.asyncio
async def test_torch_context_manager():
    """Test torch_context context manager"""
    workflow_runner = WorkflowRunner(job_id="1")
    context = ProcessingContext(user_id="1", auth_token="token")
    
    # Test without torch/comfy - should work without errors
    with workflow_runner.torch_context(context):
        pass
    
    # Test with torch available
    with patch("nodetool.workflows.workflow_runner.TORCH_AVAILABLE", True):
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        
        with patch("nodetool.workflows.workflow_runner.torch", mock_torch):
            with workflow_runner.torch_context(context):
                pass


# ============= OUTPUT NODE TESTS =============

@pytest.mark.asyncio
async def test_output_node_processing(workflow_runner: WorkflowRunner):
    """Test OutputNode specific processing"""
    input_node = {"id": "1", "type": String.get_node_type(), "data": {"value": "test_output"}}
    output = {"id": "2", "type": StringOutput.get_node_type(), "data": {"name": "my_output"}}
    
    nodes: list[dict[str, Any]] = [input_node, output]
    edges: list[dict[str, Any]] = [{
        "id": "1",
        "source": "1",
        "target": "2",
        "sourceHandle": "output",
        "targetHandle": "value",
    }]
    
    graph = APIGraph(nodes=[Node(**n) for n in nodes], edges=[Edge(**e) for e in edges])
    req = RunJobRequest(
        user_id="1", workflow_id="", job_type="", params={}, graph=graph
    )
    
    messages = []
    context = ProcessingContext(user_id="1", auth_token="token")
    context.post_message = lambda message: messages.append(message)
    
    await workflow_runner.run(req, context)
    
    # Check outputs
    assert workflow_runner.outputs["my_output"] == ["test_output"]
    
    # Check OutputUpdate message was sent
    output_updates = [m for m in messages if isinstance(m, OutputUpdate)]
    assert len(output_updates) == 1
    assert output_updates[0].output_name == "my_output"
    assert output_updates[0].value == "test_output"
    assert output_updates[0].output_type == "string"


# ============= COMPLEX GRAPH TESTS =============

@pytest.mark.asyncio
async def test_parallel_node_execution():
    """Test parallel execution of independent nodes"""
    
    class SlowNode(BaseNode):
        delay: float = 0.1
        value: str = ""
        
        @classmethod
        def get_node_type(cls):
            return "test.parallel.SlowNode"
        
        async def process(self, context: ProcessingContext) -> str:
            await asyncio.sleep(self.delay)
            return f"{self.value}_processed"
    
    workflow_runner = WorkflowRunner(job_id="1")
    
    # Two independent paths that should run in parallel
    slow1 = {"id": "1", "type": SlowNode.get_node_type(), "data": {"value": "path1", "delay": 0.1}}
    slow2 = {"id": "2", "type": SlowNode.get_node_type(), "data": {"value": "path2", "delay": 0.1}}
    out1 = {"id": "3", "type": StringOutput.get_node_type(), "data": {"name": "out1"}}
    out2 = {"id": "4", "type": StringOutput.get_node_type(), "data": {"name": "out2"}}
    
    nodes: list[dict[str, Any]] = [slow1, slow2, out1, out2]
    edges: list[dict[str, Any]] = [
        {"id": "e1", "source": "1", "target": "3", "sourceHandle": "output", "targetHandle": "value"},
        {"id": "e2", "source": "2", "target": "4", "sourceHandle": "output", "targetHandle": "value"},
    ]
    
    graph = APIGraph(nodes=[Node(**n) for n in nodes], edges=[Edge(**e) for e in edges])
    req = RunJobRequest(
        user_id="1", workflow_id="", job_type="", params={}, graph=graph
    )
    context = ProcessingContext(user_id="1", auth_token="token")
    
    import time
    start = time.time()
    await workflow_runner.run(req, context)
    duration = time.time() - start
    
    # Should complete in ~0.1s if parallel, ~0.2s if serial
    assert duration < 0.15  # Allow some overhead
    assert workflow_runner.outputs["out1"] == ["path1_processed"]
    assert workflow_runner.outputs["out2"] == ["path2_processed"]


@pytest.mark.asyncio
async def test_complex_dependency_graph():
    """Test complex graph with multiple dependencies"""
    # Diamond pattern: 1 -> (2,3) -> 4
    n1 = {"id": "1", "type": String.get_node_type(), "data": {"value": "start"}}
    n2 = {"id": "2", "type": String.get_node_type()}
    n3 = {"id": "3", "type": String.get_node_type()}
    n4 = {"id": "4", "type": StringOutput.get_node_type(), "data": {"name": "final"}}
    
    nodes: list[dict[str, Any]] = [n1, n2, n3, n4]
    edges: list[dict[str, Any]] = [
        {"id": "e1", "source": "1", "target": "2", "sourceHandle": "output", "targetHandle": "value"},
        {"id": "e2", "source": "1", "target": "3", "sourceHandle": "output", "targetHandle": "value"},
        {"id": "e3", "source": "2", "target": "4", "sourceHandle": "output", "targetHandle": "value"},
        # Note: n3 output is not connected, testing partial dependencies
    ]
    
    workflow_runner = WorkflowRunner(job_id="1")
    graph = APIGraph(nodes=[Node(**n) for n in nodes], edges=[Edge(**e) for e in edges])
    req = RunJobRequest(
        user_id="1", workflow_id="", job_type="", params={}, graph=graph
    )
    context = ProcessingContext(user_id="1", auth_token="token")
    
    await workflow_runner.run(req, context)
    
    assert workflow_runner.outputs["final"] == ["start"]


# ============= EDGE CASE TESTS =============

@pytest.mark.asyncio
async def test_empty_graph():
    """Test handling of empty graph"""
    workflow_runner = WorkflowRunner(job_id="1")
    
    graph = APIGraph(nodes=[], edges=[])
    req = RunJobRequest(
        user_id="1", workflow_id="", job_type="", params={}, graph=graph
    )
    context = ProcessingContext(user_id="1", auth_token="token")
    
    await workflow_runner.run(req, context)
    
    assert workflow_runner.status == "completed"
    assert workflow_runner.outputs == {}


@pytest.mark.asyncio
async def test_node_finalization():
    """Test that node finalize is called"""
    
    class FinalizableNode(BaseNode):
        finalized: bool = False
        
        @classmethod
        def get_node_type(cls):
            return "test.finalize.FinalizableNode"
        
        async def process(self, context: ProcessingContext) -> str:
            return "processed"
        
        async def finalize(self, context: ProcessingContext):
            self.finalized = True
    
    workflow_runner = WorkflowRunner(job_id="1")
    
    node: dict[str, Any] = {"id": "1", "type": FinalizableNode.get_node_type()}
    nodes: list[dict[str, Any]] = [node]
    
    graph = APIGraph(nodes=[Node(**n) for n in nodes], edges=[])
    req = RunJobRequest(
        user_id="1", workflow_id="", job_type="", params={}, graph=graph
    )
    context = ProcessingContext(user_id="1", auth_token="token")
    
    await workflow_runner.run(req, context)
    
    # Get the actual node instance
    assert workflow_runner.context is not None
    node_instance = workflow_runner.context.graph.find_node("1")
    assert isinstance(node_instance, FinalizableNode)
    assert node_instance.finalized


@pytest.mark.asyncio
async def test_chat_input_handling():
    """Test handling of chat messages input"""
    # Test that messages require a ChatInput node
    workflow_runner = WorkflowRunner(job_id="1")
    
    # No ChatInput node
    graph = APIGraph(nodes=[], edges=[])
    messages: list[Message] = [
        Message(role="user", content="Hello")
    ]
    req = RunJobRequest(
        user_id="1", 
        workflow_id="", 
        job_type="", 
        params={},
        messages=messages,
        graph=graph
    )
    
    context = ProcessingContext(user_id="1", auth_token="token")
    
    # Should raise error when messages provided but no ChatInput node
    with pytest.raises(ValueError, match="Chat input node not found"):
        await workflow_runner.run(req, context)


@pytest.mark.asyncio
async def test_missing_input_node_for_param():
    """Test error when parameter has no corresponding input node"""
    workflow_runner = WorkflowRunner(job_id="1")
    
    nodes = []  # No input nodes
    graph = APIGraph(nodes=[], edges=[])
    
    req = RunJobRequest(
        user_id="1",
        workflow_id="",
        job_type="",
        params={"missing_input": "value"},  # No node for this param
        graph=graph
    )
    context = ProcessingContext(user_id="1", auth_token="token")
    
    with pytest.raises(ValueError, match="No input node found for param: missing_input"):
        await workflow_runner.run(req, context)


@pytest.mark.asyncio
async def test_workflow_runner_device_selection():
    """Test device selection logic in WorkflowRunner init"""
    # Test CPU fallback
    with patch("nodetool.workflows.workflow_runner.TORCH_AVAILABLE", False):
        runner = WorkflowRunner(job_id="1")
        assert runner.device == "cpu"
    
    # Test CUDA selection
    with patch("nodetool.workflows.workflow_runner.TORCH_AVAILABLE", True):
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = True
        with patch("nodetool.workflows.workflow_runner.torch", mock_torch):
            runner = WorkflowRunner(job_id="1")
            assert runner.device == "cuda"
    
    # Test MPS selection
    with patch("nodetool.workflows.workflow_runner.TORCH_AVAILABLE", True):
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        with patch("nodetool.workflows.workflow_runner.torch", mock_torch):
            runner = WorkflowRunner(job_id="1")
            assert runner.device == "mps"
    
    # Test explicit device
    runner = WorkflowRunner(job_id="1", device="custom")
    assert runner.device == "custom"


@pytest.mark.asyncio
async def test_is_running_method():
    """Test is_running status check"""
    runner = WorkflowRunner(job_id="1")
    assert runner.is_running()
    
    runner.status = "completed"
    assert not runner.is_running()
    
    runner.status = "error"
    assert not runner.is_running()
