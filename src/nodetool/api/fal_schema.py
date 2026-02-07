"""
Optional FAL dynamic schema resolution endpoint.
When nodetool-fal is installed, POST /api/fal/resolve-dynamic-schema resolves
model_info (pasted OpenAPI JSON, llms.txt, URL, or endpoint id) and returns
dynamic_properties and dynamic_outputs for the FalAI node.
Avoids CORS by proxying the fal.ai OpenAPI fetch server-side; pasted OpenAPI
JSON is parsed directly without any fetch.
"""

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/fal", tags=["fal"])


@router.post("/resolve-dynamic-schema")
async def resolve_fal_dynamic_schema(body: dict) -> dict:
    """
    Resolve FAL dynamic schema from model_info (pasted llms.txt, URL, or endpoint id).
    Returns dynamic_properties and dynamic_outputs for the UI to update the node.
    Requires nodetool-fal to be installed.
    """
    try:
        from nodetool.nodes.fal.dynamic_schema import resolve_dynamic_schema
    except ImportError as e:
        raise HTTPException(
            501,
            "FAL dynamic schema resolution requires nodetool-fal to be installed.",
        ) from e

    model_info = (body.get("model_info") or "").strip()
    if not model_info:
        raise HTTPException(
            400,
            "model_info is required (paste OpenAPI JSON, llms.txt, URL, or endpoint id)",
        )

    try:
        result = await resolve_dynamic_schema(model_info)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    return result
