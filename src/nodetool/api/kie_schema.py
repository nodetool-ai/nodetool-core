"""
Optional Kie.ai dynamic schema resolution endpoint.
When nodetool-base is installed with kie nodes, POST /api/kie/resolve-dynamic-schema
parses pasted kie.ai API documentation and returns dynamic_properties, dynamic_inputs,
and dynamic_outputs for the KieAI dynamic node.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


class ResolveKieDynamicSchemaRequest(BaseModel):
    model_info: str


router = APIRouter(prefix="/api/kie", tags=["kie"])


@router.post("/resolve-dynamic-schema")
async def resolve_kie_dynamic_schema(body: ResolveKieDynamicSchemaRequest) -> dict:
    """
    Resolve Kie.ai dynamic schema from pasted API documentation.
    Returns dynamic_properties, dynamic_inputs, and dynamic_outputs
    for the UI to update the node.
    Requires nodetool-base with kie nodes to be installed.
    """
    try:
        from nodetool.nodes.kie.dynamic_schema import resolve_dynamic_schema
    except ImportError as e:
        raise HTTPException(
            501,
            "Kie.ai dynamic schema resolution requires nodetool-base to be installed.",
        ) from e

    model_info = body.model_info.strip()
    if not model_info:
        raise HTTPException(
            400,
            "model_info is required (paste kie.ai API documentation)",
        )

    try:
        result = await resolve_dynamic_schema(model_info)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    return result
