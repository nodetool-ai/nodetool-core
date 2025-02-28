from nodetool.metadata.types import Message
from nodetool.types.graph import Graph


from pydantic import BaseModel


from typing import Any, Literal


class RunJobRequest(BaseModel):
    """
    A request model for running a workflow.

    Attributes:
        type: The type of request, always "run_job_request".
        job_type: The type of job to run, defaults to "workflow".
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
    params: Any | None = None
    messages: list[Message] | None = None
    workflow_id: str = ""
    user_id: str = ""
    auth_token: str = ""
    api_url: str | None = None
    env: dict[str, Any] | None = None
    graph: Graph | None = None
    explicit_types: bool | None = False
