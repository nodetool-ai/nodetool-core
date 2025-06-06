#!/usr/bin/env python

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from nodetool.types.prediction import (
    PredictionCreateRequest,
    Prediction,
    PredictionList,
)
from nodetool.api.utils import current_user
from nodetool.common.environment import Environment
from typing import Optional
from nodetool.models.prediction import (
    Prediction as PredictionModel,
)

log = Environment.get_logger()
router = APIRouter(prefix="/api/predictions", tags=["predictions"])


def from_model(prediction: PredictionModel):
    return Prediction(
        id=prediction.id,
        user_id=prediction.user_id,
        node_id=prediction.node_id,
        workflow_id=prediction.workflow_id,
        provider=prediction.provider,
        model=prediction.model,
        status=prediction.status,
        logs=prediction.logs,
        error=prediction.error,
        duration=prediction.duration,
        created_at=(
            prediction.created_at.isoformat() if prediction.created_at else None
        ),
        started_at=(
            prediction.started_at.isoformat() if prediction.started_at else None
        ),
        completed_at=(
            prediction.completed_at.isoformat() if prediction.completed_at else None
        ),
    )


@router.get("/")
async def index(
    cursor: Optional[str] = None,
    page_size: Optional[int] = None,
    user: str = Depends(current_user),
) -> PredictionList:
    """
    Returns all assets for a given user or workflow.
    """
    if page_size is None:
        page_size = 100

    predictions, next_cursor = PredictionModel.paginate(
        user_id=user,
        limit=page_size,
        start_key=cursor,
    )

    predictions = [from_model(p) for p in predictions]

    return PredictionList(next=next_cursor, predictions=predictions)


@router.get("/{id}")
async def get(id: str, user: str = Depends(current_user)) -> Prediction:
    pred = PredictionModel.find(user, id)
    if pred is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return from_model(pred)


@router.post("/")
async def create(req: PredictionCreateRequest, user: str = Depends(current_user)):
    prediction = PredictionModel.create(
        model=req.model,
        node_id=req.node_id,
        user_id=user,
        workflow_id=req.workflow_id,
        provider=req.provider.value,
        started_at=datetime.now(),
    )
    return from_model(prediction)
