"""
Optional Replicate dynamic schema resolution endpoint.
When nodetool-replicate is installed, POST /api/replicate/resolve-dynamic-schema
resolves model_info (model identifier or URL) and returns dynamic_properties,
dynamic_inputs, and dynamic_outputs for the ReplicateAI node.
Requires REPLICATE_API_TOKEN to fetch model schemas from the Replicate API.
"""

import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from nodetool.api.utils import current_user


class ResolveReplicateDynamicSchemaRequest(BaseModel):
    model_info: str


router = APIRouter(prefix="/api/replicate", tags=["replicate"])


@router.post("/resolve-dynamic-schema")
async def resolve_replicate_dynamic_schema(
    body: ResolveReplicateDynamicSchemaRequest,
    user: str = Depends(current_user),
) -> dict:
    """
    Resolve Replicate dynamic schema from model_info (model identifier or URL).
    Returns dynamic_properties, dynamic_inputs, and dynamic_outputs for the UI
    to update the node.
    Requires nodetool-replicate to be installed.
    """
    try:
        from nodetool.nodes.replicate.dynamic_schema import (
            resolve_dynamic_schema,
        )
    except ImportError as e:
        raise HTTPException(
            501,
            "Replicate dynamic schema resolution requires nodetool-replicate to be installed.",
        ) from e

    model_info = body.model_info.strip()
    if not model_info:
        raise HTTPException(
            400,
            "model_info is required (paste a model identifier like runwayml/gen-4.5 or a Replicate URL)",
        )

    from nodetool.security.secret_helper import get_secret

    api_token = await get_secret("REPLICATE_API_TOKEN", user)
    if not api_token:
        api_token = os.environ.get("REPLICATE_API_TOKEN", "")

    try:
        result = await resolve_dynamic_schema(model_info, api_token)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    return result
