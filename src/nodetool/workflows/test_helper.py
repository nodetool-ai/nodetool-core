import asyncio
from typing import Any, AsyncGenerator, TypedDict

from pydantic import Field

from nodetool.workflows.base_node import BaseNode, InputNode, OutputNode


class StringInput(InputNode):
    """Test-only string input node for workflow testing."""

    value: str = ""

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.StringInput"

    async def process(self, context: Any) -> str:
        return self.value


class FloatInput(InputNode):
    """Test-only float input node for workflow testing."""

    default: float = 0.0

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.FloatInput"

    async def process(self, context: Any) -> float:
        return self.default


class IntInput(InputNode):
    """Test-only int input node for workflow testing."""

    default: int = 0

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.IntInput"

    async def process(self, context: Any) -> int:
        return self.default


class StringOutput(OutputNode):
    """Test-only string output node for workflow testing."""

    value: str = ""

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.StringOutput"


class IntOutput(OutputNode):
    """Test-only int output node for workflow testing."""

    value: int = 0

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.IntOutput"


class FormatText(BaseNode):
    """Test-only format text node for workflow testing."""

    template: str = Field(default="Hello, {{ text }}", description="The template string with placeholders")
    text: str = Field(default="", description="The text value to insert into the template")

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.FormatText"

    async def process(self, context: Any) -> str:
        result = self.template
        result = result.replace("{{ text }}", str(self.text))
        return result


class ThresholdProcessor(BaseNode):
    """Test-only node with configurable threshold for control edge testing.

    This node processes a value and returns whether it exceeds the threshold.
    Designed to be controlled via control edges.
    """

    value: float = Field(default=0.0, description="The input value to process")
    threshold: float = Field(default=0.5, description="The threshold for processing")
    mode: str = Field(default="normal", description="Processing mode: 'normal' or 'strict'")

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.ThresholdProcessor"

    async def process(self, context: Any) -> str:
        exceeds = self.value > self.threshold if self.mode == "strict" else self.value >= self.threshold
        return f"value={self.value}, threshold={self.threshold}, mode={self.mode}, exceeds={exceeds}"


class SimpleController(BaseNode):
    """Test-only controller node that yields RunEvents.

    This node emits control events to trigger controlled node execution.
    It can be configured with control properties to send to controlled nodes.
    """

    control_threshold: float = Field(default=0.8, description="Threshold to set on controlled node")
    control_mode: str = Field(default="strict", description="Mode to set on controlled node")
    trigger_on_init: bool = Field(default=True, description="Whether to trigger control on initialization")
    include_properties: bool = Field(
        default=True,
        description="Whether to include control properties in the emitted RunEvent",
    )

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.SimpleController"

    class OutputType(TypedDict):
        result: str

    @classmethod
    def get_return_type(cls):
        return cls.OutputType

    async def gen_process(self, context: Any) -> AsyncGenerator[dict, None]:
        """Yield control events as an async generator."""
        from nodetool.workflows.control_events import RunEvent

        if self.trigger_on_init:
            properties: dict[str, Any] = {}
            if self.include_properties:
                properties = {
                    "threshold": self.control_threshold,
                    "mode": self.control_mode,
                }

            # Emit control event with properties for controlled node
            yield {"__control__": RunEvent(properties=properties)}

        # Also yield a normal output
        yield {"result": f"Controller configured with threshold={self.control_threshold}, mode={self.control_mode}"}


class MultiTriggerController(BaseNode):
    """Controller that emits multiple RunEvents for testing fan-out scenarios."""

    event_properties: list[dict[str, Any]] = Field(
        default_factory=lambda: [
            {"threshold": 0.3},
            {"threshold": 0.6},
            {"threshold": 0.9},
        ],
        description="Sequence of property overrides to send with each RunEvent",
    )
    emit_final_result: bool = Field(
        default=False,
        description="Whether to emit a final data output after control events",
    )
    final_message: str = Field(
        default="MultiTrigger controller completed",
        description="Message emitted when emit_final_result is True",
    )

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.MultiTriggerController"

    class OutputType(TypedDict):
        result: str

    @classmethod
    def get_return_type(cls):
        return cls.OutputType

    async def gen_process(self, context: Any) -> AsyncGenerator[dict, None]:
        from nodetool.workflows.control_events import RunEvent

        events = self.event_properties or [{}]
        for props in events:
            yield {"__control__": RunEvent(properties=dict(props or {}))}

        if self.emit_final_result:
            yield {"result": self.final_message}


class ErrorProcessor(BaseNode):
    """Node that always raises an error when processed (for error propagation tests)."""

    message: str = Field(default="Controlled node failure", description="Error message to raise")

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.ErrorProcessor"

    async def process(self, context: Any) -> Any:
        raise RuntimeError(self.message)


class IntAccumulator(BaseNode):
    """Test-only node that accumulates integer values.

    Tracks how many times it has been executed, useful for testing
    multiple control events.
    """

    value: int = Field(default=0, description="Input value")

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.IntAccumulator"

    def __init__(self, **data):
        super().__init__(**data)
        self._execution_count = 0
        self._accumulated_values: list[int] = []

    async def process(self, context: Any) -> dict[str, Any]:
        self._execution_count += 1
        self._accumulated_values.append(self.value)
        return {
            "output": f"execution #{self._execution_count}, value={self.value}",
            "count": self._execution_count,
            "values": list(self._accumulated_values),
        }


class StreamingInputProcessor(BaseNode):
    """Test-only node with is_streaming_input=True.

    Consumes input stream manually and yields results.
    """

    prefix: str = Field(default="item", description="Prefix for output items")
    value: Any = Field(default=None, description="Input stream to consume")

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.StreamingInputProcessor"

    class OutputType(TypedDict):
        result: str

    @classmethod
    def get_return_type(cls):
        return cls.OutputType

    @classmethod
    def is_streaming_input(cls) -> bool:
        return True

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    async def gen_process(self, context: Any) -> AsyncGenerator[dict, None]:
        # Use self.iter_input instead of inputs argument
        count = 0
        async for item in self.iter_input("value"):
            count += 1
            yield {"result": f"{self.prefix}: {item} (#{count})"}


class StreamingOutputProcessor(BaseNode):
    """Test-only node with is_streaming_output=True (but buffered input).

    Takes a single input (e.g., a count) and yields multiple outputs.
    """

    count: int = Field(default=3, description="Number of items to yield")
    base_value: str = Field(default="data", description="Base string for output")

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.StreamingOutputProcessor"

    class OutputType(TypedDict):
        result: str

    @classmethod
    def get_return_type(cls):
        return cls.OutputType

    async def gen_process(self, context: Any) -> AsyncGenerator[dict, None]:
        for i in range(self.count):
            yield {"result": f"{self.base_value}_{i + 1}"}


class StreamingInputBufferedOutputNode(BaseNode):
    """Test-only node with is_streaming_input=True but buffered output.

    Consumes entire input stream then returns one result.
    """

    value: Any = Field(default=None, description="Input stream to consume")

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.StreamingInputBufferedOutputNode"

    class OutputType(TypedDict):
        result: str

    @classmethod
    def get_return_type(cls):
        return cls.OutputType

    @classmethod
    def is_streaming_input(cls) -> bool:
        return True

    @classmethod
    def is_streaming_output(cls) -> bool:
        return False

    async def process(self, context: Any) -> dict[str, Any]:
        items = []
        async for item in self.iter_input("value"):
            items.append(item)
        return {"result": str(items)}


class ListSumProcessor(BaseNode):
    """Node that aggregates a list of integers and sums them.

    Useful for testing list aggregation (multi-edge or single-edge to list).
    """

    values: list[int] = Field(default=[], description="List of integers to sum")

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.ListSumProcessor"

    class OutputType(TypedDict):
        sum: int
        count: int
        items: list[int]

    @classmethod
    def get_return_type(cls):
        return cls.OutputType

    async def process(self, context: Any) -> dict[str, Any]:
        return {
            "sum": sum(self.values),
            "count": len(self.values),
            "items": self.values,
        }


class ListMultiplierProcessor(BaseNode):
    """Node that takes a list of integers and a multiplier factor.

    Useful for testing mixed list and non-list inputs (ACTOR-013).
    """

    values: list[int] = Field(default=[], description="List of integers to multiply")
    factor: int = Field(default=1, description="Multiplier factor")

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.ListMultiplierProcessor"

    class OutputType(TypedDict):
        result: list[int]
        sum: int

    @classmethod
    def get_return_type(cls):
        return cls.OutputType

    async def process(self, context: Any) -> dict[str, Any]:
        result = [v * self.factor for v in self.values]
        return {
            "result": result,
            "sum": sum(result),
        }


class SilentNode(BaseNode):
    """Node that suppresses output routing (for ACTOR-016 testing).

    This node should not route its outputs to downstream nodes.
    """

    value: Any = Field(default=None, description="Input value")

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.SilentNode"

    def should_route_output(self, output_name: str) -> bool:
        return False

    async def process(self, context: Any) -> dict[str, Any]:
        return {"output": f"silent:{self.value}"}


class PassThrough(BaseNode):
    """Simple pass-through node for chained control testing (CTRL-023).

    Passes input to output and can be controlled via control edges.
    """

    value: Any = Field(default=None, description="Input value to pass through")
    prefix: str = Field(default="", description="Optional prefix to add")

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.PassThrough"

    async def process(self, context: Any) -> str:
        if self.prefix:
            return f"{self.prefix}:{self.value}"
        return str(self.value)


class ConditionalErrorProcessor(BaseNode):
    """Node that raises error conditionally based on message content (CTRL-022).

    Useful for testing error recovery scenarios.
    """

    message: str = Field(default="", description="Message to check for error condition")
    error_on: str = Field(default="boom", description="Substring that triggers error")

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.ConditionalErrorProcessor"

    async def process(self, context: Any) -> str:
        if self.error_on in self.message:
            raise RuntimeError(f"Triggered error: {self.message}")
        return f"Safe: {self.message}"


class ConditionalErrorController(BaseNode):
    """Controller that yields events then optionally errors (CTRL-018).

    Useful for testing controller error handling mid-stream.
    """

    fail_after_events: int = Field(default=1, description="Number of events to emit before failing")
    event_properties: list[dict[str, Any]] = Field(
        default_factory=lambda: [{"threshold": 0.5}],
        description="Properties to emit for each event",
    )
    error_message: str = Field(default="Controller failed mid-stream", description="Error message")

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.ConditionalErrorController"

    async def gen_process(self, context: Any) -> AsyncGenerator[dict, None]:
        from nodetool.workflows.control_events import RunEvent

        events = self.event_properties or [{}]
        for i, props in enumerate(events):
            if i >= self.fail_after_events:
                raise RuntimeError(self.error_message)
            yield {"__control__": RunEvent(properties=dict(props or {}))}


class StopEventController(BaseNode):
    """Controller that emits StopEvent (CTRL-028).

    Emits a control StopEvent to test graceful stopping.
    """

    emit_stop: bool = Field(default=True, description="Whether to emit StopEvent")
    emit_run_first: bool = Field(default=True, description="Emit RunEvent before StopEvent")

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.StopEventController"

    async def gen_process(self, context: Any) -> AsyncGenerator[dict, None]:
        from nodetool.workflows.control_events import RunEvent, StopEvent

        if self.emit_run_first:
            yield {"__control__": RunEvent(properties={})}
        if self.emit_stop:
            yield {"__control__": StopEvent()}


class LegacyControlController(BaseNode):
    """Controller that emits legacy __control_output__ format (CTRL-030, LEGACY-001).

    Uses the old dict-based control output format for backward compatibility testing.
    """

    threshold: float = Field(default=0.5, description="Threshold to set")
    mode: str = Field(default="normal", description="Mode to set")

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.LegacyControlController"

    async def process(self, context: Any) -> dict[str, Any]:
        # Legacy format: return __control_output__ dict
        return {
            "__control_output__": {
                "threshold": self.threshold,
                "mode": self.mode,
            },
            "result": "legacy controller done",
        }


class FullStreamingNode(BaseNode):
    """Full streaming node with both streaming input and output (ACTOR-004).

    Consumes input stream and produces output stream.
    """

    input_stream: Any = Field(default=None, description="Input stream to process")
    transform: str = Field(default="upper", description="Transformation: upper, lower, reverse")

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.FullStreamingNode"

    @classmethod
    def is_streaming_input(cls) -> bool:
        return True

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    async def gen_process(self, context: Any) -> AsyncGenerator[dict, None]:
        async for item in self.iter_input("input_stream"):
            if self.transform == "upper":
                result = str(item).upper()
            elif self.transform == "lower":
                result = str(item).lower()
            elif self.transform == "reverse":
                result = str(item)[::-1]
            else:
                result = str(item)
            yield {"output": result}


class IntStreamingOutputProcessor(BaseNode):
    """Streaming output node that yields integers.

    Useful for feeding list[int] inputs in tests.
    """

    count: int = Field(default=3, description="Number of integers to yield")
    start: int = Field(default=1, description="Starting integer value")
    step: int = Field(default=1, description="Step between integers")

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.IntStreamingOutputProcessor"

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    class OutputType(TypedDict):
        result: int

    @classmethod
    def get_return_type(cls):
        return cls.OutputType

    async def gen_process(self, context: Any) -> AsyncGenerator[dict, None]:
        for i in range(self.count):
            yield {"result": self.start + i * self.step}


class SlowNode(BaseNode):
    """A node that takes a specified duration to execute.

    Useful for testing cancellation and timeout scenarios.
    """

    duration: float = Field(default=1.0, description="Duration to sleep in seconds")

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.SlowNode"

    async def process(self, context: Any) -> str:
        await asyncio.sleep(self.duration)
        return "completed"
