from nodetool.metadata.types import Message
from nodetool.types.graph import Graph


from pydantic import BaseModel


from typing import Any, Literal
from enum import Enum


class ResourceLimits(BaseModel):
    """
    Resource limits for subprocess execution.

    Attributes:
        cpu_percent: CPU usage limit as percentage (1-100), requires cpulimit
        memory_mb: Memory limit in megabytes, uses ulimit -v
        time_seconds: CPU time limit in seconds, uses ulimit -t
        file_size_mb: File size limit in megabytes, uses ulimit -f
        open_files: Maximum number of open file descriptors, uses ulimit -n
        max_processes: Maximum number of processes, uses ulimit -u
    """

    cpu_percent: int | None = None
    memory_mb: int | None = None
    time_seconds: int | None = None
    file_size_mb: int | None = None
    open_files: int | None = None
    max_processes: int | None = None


class ExecutionStrategy(str, Enum):
    """Execution strategy for workflow jobs."""

    THREADED = "threaded"
    SUBPROCESS = "subprocess"
    DOCKER = "docker"


class RunJobRequest(BaseModel):
    """
    A request model for running a workflow.

    Attributes:
        type: The type of request, always "run_job_request".
        job_type: The type of job to run, defaults to "workflow".
        execution_strategy: Strategy for executing the job (threaded, subprocess, docker).
        params: Optional parameters for the job.
        messages: Optional list of messages associated with the job.
        workflow_id: The ID of the workflow to run.
        user_id: The ID of the user making the request.
        auth_token: Authentication token for the request.
        api_url: Optional API URL to use for the job.
        env: Optional environment variables for the job.
        graph: Optional graph data for the job.
        explicit_types: Whether to use explicit types, defaults to False.
    """

    type: Literal["run_job_request"] = "run_job_request"
    job_type: str = "workflow"
    execution_strategy: ExecutionStrategy = ExecutionStrategy.THREADED
    params: Any | None = None
    messages: list[Message] | None = None
    workflow_id: str = ""
    user_id: str = ""
    auth_token: str = ""
    api_url: str | None = None
    env: dict[str, Any] | None = None
    graph: Graph | None = None
    explicit_types: bool | None = False
    resource_limits: ResourceLimits | None = None
