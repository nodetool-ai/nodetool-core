from pydantic import BaseModel
from enum import Enum

from typing import Any, Callable, Literal
from nodetool.metadata.types import Task, SubTask
from nodetool.types.job import JobUpdate
from nodetool.types.prediction import Prediction


class TaskUpdateEvent(str, Enum):
    """Enum for different task update event types."""

    TASK_CREATED = "task_created"
    SUBTASK_STARTED = "subtask_started"
    ENTERED_CONCLUSION_STAGE = "entered_conclusion_stage"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    SUBTASK_COMPLETED = "subtask_completed"


class TaskUpdate(BaseModel):
    """A task update from a provider."""

    type: Literal["task_update"] = "task_update"
    node_id: str | None = None
    task: Task
    subtask: SubTask | None = None
    event: TaskUpdateEvent


class NodeUpdate(BaseModel):
    type: Literal["node_update"] = "node_update"
    node_id: str
    node_name: str
    status: str
    error: str | None = None
    logs: str | None = None
    result: dict[str, Any] | None = None
    properties: dict[str, Any] | None = None


class BinaryUpdate(BaseModel):
    type: Literal["binary_update"] = "binary_update"
    node_id: str
    output_name: str
    binary: bytes

    def encode(self) -> bytes:
        """
        Create an encoded message containing two null-terminated strings and PNG data.
        """
        # Encode the strings as UTF-8 and add null terminators
        encoded_node_id = self.node_id.encode("utf-8") + b"\x00"
        encoded_output_name = self.output_name.encode("utf-8") + b"\x00"

        # Combine all parts of the message
        message = encoded_node_id + encoded_output_name + self.binary

        return message


class NodeProgress(BaseModel):
    type: Literal["node_progress"] = "node_progress"
    node_id: str
    progress: int
    total: int
    chunk: str = ""


class Error(BaseModel):
    type: Literal["error"] = "error"
    error: str


class RunFunction(BaseModel):
    """
    A message to run a function in the main thread.
    """

    type: Literal["run_function"] = "run_function"
    function: Callable
    args: list[Any] = []
    kwargs: dict[str, Any] = {}


ProcessingMessage = (
    NodeUpdate
    | NodeProgress
    | JobUpdate
    | Error
    | Prediction
    | RunFunction
    | TaskUpdate
)
