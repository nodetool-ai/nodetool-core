from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from nodetool.metadata.types import Chunk, Step, Task
from nodetool.types.job import JobUpdate
from nodetool.types.prediction import Prediction


def sanitize_memory_uris_for_client(value: Any) -> Any:
    """
    Recursively sanitize memory:// URIs from values before sending to clients.

    This function ensures that AssetRef objects with memory:// URIs are converted
    to a client-safe format:
    - If data is available inline, set uri to "" and keep data
    - If asset_id exists, set uri to "asset://<asset_id>"
    - Otherwise, set uri to ""

    Args:
        value: Any Python value that might contain AssetRef objects with memory:// URIs

    Returns:
        The value with all memory:// URIs sanitized
    """
    if isinstance(value, dict):
        # Check if this dict looks like a serialized AssetRef
        if "type" in value and "uri" in value:
            uri = value.get("uri")
            if isinstance(uri, str) and uri.startswith("memory://"):
                # Sanitize the memory:// URI
                sanitized = dict(value)
                if value.get("data") is not None:
                    # Data is available, clear the memory:// URI
                    sanitized["uri"] = ""
                elif value.get("asset_id"):
                    # Asset ID exists, use asset:// URI
                    sanitized["uri"] = f"asset://{value['asset_id']}"
                else:
                    # No data or asset_id, clear the URI
                    sanitized["uri"] = ""
                # Recursively sanitize other fields
                return {k: sanitize_memory_uris_for_client(v) if k not in ("uri", "data", "asset_id") else v
                        for k, v in sanitized.items()}
            else:
                # Recursively sanitize nested values
                return {k: sanitize_memory_uris_for_client(v) for k, v in value.items()}
        else:
            # Regular dict, recursively sanitize all values
            return {k: sanitize_memory_uris_for_client(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [sanitize_memory_uris_for_client(item) for item in value]
    elif isinstance(value, tuple):
        return tuple(sanitize_memory_uris_for_client(item) for item in value)
    elif hasattr(value, "model_copy") and hasattr(value, "uri"):
        # Handle Pydantic model AssetRef objects
        uri = getattr(value, "uri", None)
        if isinstance(uri, str) and uri.startswith("memory://"):
            data = getattr(value, "data", None)
            asset_id = getattr(value, "asset_id", None)
            if data is not None:
                return value.model_copy(update={"uri": ""})
            elif asset_id:
                return value.model_copy(update={"uri": f"asset://{asset_id}"})
            else:
                return value.model_copy(update={"uri": ""})
        return value
    else:
        return value


class TaskUpdateEvent(str, Enum):
    """Enum for different task update event types."""

    TASK_CREATED = "task_created"
    STEP_STARTED = "step_started"
    ENTERED_CONCLUSION_STAGE = "entered_conclusion_stage"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    TASK_COMPLETED = "task_completed"


class StepResult(BaseModel):
    """
    A message representing a result from a step.
    """

    type: Literal["step_result"] = "step_result"
    step: Step
    result: Any
    error: str | None = None
    is_task_result: bool = False
    thread_id: str | None = None
    workflow_id: str | None = None


class PlanningUpdate(BaseModel):
    """
    A message representing a planning update from a node.

    Used for communicating planning stage information to clients, especially
    for nodes that involve multi-step planning processes.
    """

    type: Literal["planning_update"] = "planning_update"
    node_id: str | None = None
    thread_id: str | None = None
    workflow_id: str | None = None
    phase: str
    status: str
    content: str | None = None


class LogUpdate(BaseModel):
    """
    A message representing a log update from a node.
    """

    type: Literal["log_update"] = "log_update"
    node_id: str
    node_name: str
    content: str
    severity: Literal["info", "warning", "error"]
    workflow_id: str | None = None


class Notification(BaseModel):
    """
    A message representing a notification from a node.
    """

    type: Literal["notification"] = "notification"
    node_id: str
    content: str
    severity: Literal["info", "warning", "error"]
    workflow_id: str | None = None


class PreviewUpdate(BaseModel):
    """
    A message representing a preview update from a node.
    """

    type: Literal["preview_update"] = "preview_update"
    node_id: str
    value: Any

    @model_validator(mode="after")
    def _sanitize_memory_uris(self) -> "PreviewUpdate":
        """Ensure memory:// URIs are never sent to clients."""
        self.value = sanitize_memory_uris_for_client(self.value)
        return self


class SaveUpdate(BaseModel):
    """
    A message representing a save update from a node.
    """

    type: Literal["save_update"] = "save_update"
    node_id: str
    name: str
    value: Any
    output_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResultUpdate(BaseModel):
    """
    A message representing a tool result from a node.
    """

    type: Literal["tool_result_update"] = "tool_result_update"
    node_id: str
    thread_id: str | None = None
    workflow_id: str | None = None
    result: dict[str, Any]

    @model_validator(mode="after")
    def _sanitize_memory_uris(self) -> "ToolResultUpdate":
        """Ensure memory:// URIs are never sent to clients."""
        self.result = sanitize_memory_uris_for_client(self.result)
        return self


class TaskUpdate(BaseModel):
    """
    A message representing an update to a task's status.

    Used for communicating progress and status changes for complex
    task-based operations, such as agent workflows.
    """

    type: Literal["task_update"] = "task_update"
    node_id: str | None = None
    thread_id: str | None = None
    workflow_id: str | None = None
    task: Task
    step: Step | None = None
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
    result: dict[str, Any] | None = None
    properties: dict[str, Any] | None = None
    workflow_id: str | None = None


class EdgeUpdate(BaseModel):
    """
    A message representing an update to an edge.
    """

    type: Literal["edge_update"] = "edge_update"
    workflow_id: str
    edge_id: str
    status: str
    counter: int | None = None


class ToolCallUpdate(BaseModel):
    """
    A message representing a tool call from a provider.

    Used to communicate when an AI provider executes a tool call,
    particularly useful in agent-based workflows.
    """

    type: Literal["tool_call_update"] = "tool_call_update"
    node_id: str | None = None
    thread_id: str | None = None
    workflow_id: str | None = None
    tool_call_id: str | None = None
    name: str
    args: dict[str, Any]
    message: str | None = None
    step_id: str | None = None
    agent_execution_id: str | None = None


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
    workflow_id: str | None = None


class Error(BaseModel):
    """
    A message representing a general error that occurred during workflow execution.

    Used for communicating errors that aren't specific to a particular node
    or when the node context is unavailable.
    """

    type: Literal["error"] = "error"
    message: str
    thread_id: str | None = None
    workflow_id: str | None = None


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
    metadata: dict[str, Any] = Field(default_factory=dict)
    workflow_id: str | None = None

    @model_validator(mode="after")
    def _sanitize_memory_uris(self) -> "OutputUpdate":
        """Ensure memory:// URIs are never sent to clients."""
        self.value = sanitize_memory_uris_for_client(self.value)
        return self


ProcessingMessage = (
    NodeUpdate
    | NodeProgress
    | EdgeUpdate
    | JobUpdate
    | Error
    | Chunk
    | Notification
    | Prediction
    | PreviewUpdate
    | SaveUpdate
    | LogUpdate
    | TaskUpdate
    | ToolCallUpdate
    | ToolResultUpdate
    | PlanningUpdate
    | OutputUpdate
    | StepResult
)
