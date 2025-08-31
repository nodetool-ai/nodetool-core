from pydantic import BaseModel
from enum import Enum

from typing import Any, Literal
from nodetool.metadata.types import BaseType, Task, SubTask
from nodetool.types.job import JobUpdate
from nodetool.types.prediction import Prediction


class TaskUpdateEvent(str, Enum):
    """Enum for different task update event types."""

    TASK_CREATED = "task_created"
    SUBTASK_STARTED = "subtask_started"
    ENTERED_CONCLUSION_STAGE = "entered_conclusion_stage"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    MAX_TOOL_CALLS_REACHED = "max_tool_calls_reached"
    SUBTASK_COMPLETED = "subtask_completed"
    SUBTASK_FAILED = "subtask_failed"
    TASK_COMPLETED = "task_completed"


class SubTaskResult(BaseModel):
    """
    A message representing a result from a subtask.
    """

    type: Literal["subtask_result"] = "subtask_result"
    subtask: SubTask
    result: Any
    error: str | None = None
    is_task_result: bool = False


class PlanningUpdate(BaseModel):
    """
    A message representing a planning update from a node.

    Used for communicating planning stage information to clients, especially
    for nodes that involve multi-step planning processes.
    """

    type: Literal["planning_update"] = "planning_update"
    node_id: str | None = None
    phase: str
    status: str
    content: str | None = None


class PreviewUpdate(BaseModel):
    """
    A message representing a preview update from a node.
    """

    type: Literal["preview_update"] = "preview_update"
    node_id: str
    value: Any


class ToolResultUpdate(BaseModel):
    """
    A message representing a tool result from a node.
    """

    type: Literal["tool_result_update"] = "tool_result_update"
    node_id: str
    result: dict[str, Any]


class TaskUpdate(BaseModel):
    """
    A message representing an update to a task's status.

    Used for communicating progress and status changes for complex
    task-based operations, such as agent workflows.
    """

    type: Literal["task_update"] = "task_update"
    node_id: str | None = None
    task: Task
    subtask: SubTask | None = None
    event: TaskUpdateEvent


class NodeUpdate(BaseModel):
    """
    A message representing a general update from a node.

    This is the primary way nodes communicate their status, results,
    and errors to the workflow runner and clients.
    """

    type: Literal["node_update"] = "node_update"
    node_id: str
    node_name: str
    node_type: str
    status: str
    error: str | None = None
    logs: str | None = None
    result: dict[str, Any] | None = None
    properties: dict[str, Any] | None = None


class EdgeUpdate(BaseModel):
    """
    A message representing an update to an edge.
    """

    type: Literal["edge_update"] = "edge_update"
    edge_id: str
    status: str


class ToolCallUpdate(BaseModel):
    """
    A message representing a tool call from a provider.

    Used to communicate when an AI provider executes a tool call,
    particularly useful in agent-based workflows.
    """

    type: Literal["tool_call_update"] = "tool_call_update"
    node_id: str | None = None
    name: str
    args: dict[str, Any]
    message: str | None = None


class BinaryUpdate(BaseModel):
    """
    A message containing binary data such as images, audio, or other non-text content.

    Used for passing binary data between nodes or to clients, with metadata
    to identify the source and purpose of the data.
    """

    type: Literal["binary_update"] = "binary_update"
    node_id: str
    output_name: str
    binary: bytes

    def encode(self) -> bytes:
        """
        Create an encoded message containing two null-terminated strings and binary data.

        Returns:
            bytes: Encoded message with node_id and output_name as null-terminated strings,
                  followed by the binary content.
        """
        # Encode the strings as UTF-8 and add null terminators
        encoded_node_id = self.node_id.encode("utf-8") + b"\x00"
        encoded_output_name = self.output_name.encode("utf-8") + b"\x00"

        # Combine all parts of the message
        message = encoded_node_id + encoded_output_name + self.binary

        return message


class NodeProgress(BaseModel):
    """
    A message representing progress of a node's execution.

    Used for communicating completion percentage and partial results
    from long-running operations to clients.
    """

    type: Literal["node_progress"] = "node_progress"
    node_id: str
    progress: int
    total: int
    chunk: str = ""


class Chunk(BaseType):
    """
    A message representing a chunk of streamed content from a provider.

    Used for streaming partial results in text generation, audio processing,
    or other operations where results are produced incrementally.
    """

    type: Literal["chunk"] = "chunk"
    node_id: str | None = None
    content_type: Literal["text", "audio", "image", "video", "document"] = "text"
    content: str
    done: bool = False


class Error(BaseModel):
    """
    A message representing a general error that occurred during workflow execution.

    Used for communicating errors that aren't specific to a particular node
    or when the node context is unavailable.
    """

    type: Literal["error"] = "error"
    error: str


class OutputUpdate(BaseModel):
    """
    A message representing output from an output node.

    This message type allows for direct streaming of output values from workflow nodes
    to consumers that may need immediate access to outputs before workflow completion.
    It provides structured metadata about the output, including its type and source.
    """

    type: Literal["output_update"] = "output_update"
    node_id: str
    node_name: str
    output_name: str
    value: Any
    output_type: str
    metadata: dict[str, Any] = {}


ProcessingMessage = (
    NodeUpdate
    | NodeProgress
    | EdgeUpdate
    | JobUpdate
    | Error
    | Chunk
    | Prediction
    | PreviewUpdate
    | TaskUpdate
    | ToolCallUpdate
    | ToolResultUpdate
    | PlanningUpdate
    | OutputUpdate
    | SubTaskResult
)
