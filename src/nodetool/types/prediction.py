import base64
from typing import Any, List, Literal

from pydantic import BaseModel, Field

from nodetool.metadata.types import Provider


class Prediction(BaseModel):
    """
    A prediction made by a remote model.
    """

    type: Literal["prediction"] = "prediction"

    id: str
    user_id: str
    node_id: str
    workflow_id: str | None = None
    provider: str | None = None
    model: str | None = None
    version: str | None = None
    node_type: str | None = None
    status: str
    params: dict[str, Any] = Field(default_factory=dict)
    data: Any | None = None
    cost: float | None = None
    logs: str | None = None
    error: str | None = None
    duration: float | None = None
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


class PredictionList(BaseModel):
    next: str | None
    predictions: List[Prediction]


class PredictionCreateRequest(BaseModel):
    """
    The request body for creating a prediction.
    """

    provider: Provider
    model: str
    node_id: str
    params: dict[str, Any] = Field(default_factory=dict)
    version: str | None = None
    workflow_id: str | None = None


class PredictionResult(BaseModel):
    type: Literal["prediction_result"] = "prediction_result"
    prediction: Prediction
    encoding: Literal["json"] | Literal["base64"]
    content: Any

    def decode_content(self) -> Any:
        if self.encoding == "base64":
            return base64.b64decode(self.content)
        return self.content
