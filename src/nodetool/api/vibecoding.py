"""
VibeCoding API router for generating HTML apps for workflows.

Provides endpoints for generating custom HTML frontends for Nodetool workflows
using the VibeCoding agent.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from nodetool.agents.vibecoding import VibeCodingAgent
from nodetool.api.utils import current_user
from nodetool.api.workflow import from_model
from nodetool.config.logging_config import get_logger
from nodetool.models.workflow import Workflow as WorkflowModel

log = get_logger(__name__)

router = APIRouter(prefix="/api/vibecoding", tags=["vibecoding"])


class GenerateRequest(BaseModel):
    """Request model for HTML generation."""

    workflow_id: str
    prompt: str
    thread_id: str | None = None  # Optional, for future use


class Template(BaseModel):
    """A starter template for HTML generation."""

    id: str
    name: str
    description: str
    prompt: str


@router.post("/generate")
async def generate_html(
    request: GenerateRequest,
    user: str = Depends(current_user),
) -> StreamingResponse:
    """
    Generate HTML app for a workflow based on user prompt.

    Returns streaming response with the generated HTML.
    The HTML will be wrapped in ```html ... ``` code blocks.
    """
    # 1. Load workflow and verify access
    workflow = await WorkflowModel.get(request.workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.access != "public" and workflow.user_id != user:
        raise HTTPException(status_code=403, detail="Access denied")

    # 2. Convert to API type with schemas
    workflow_data = await from_model(workflow)

    # 3. Create agent and generate
    agent = VibeCodingAgent(workflow_data)

    async def stream_response():
        try:
            async for chunk in agent.generate(request.prompt, user_id=user):
                yield chunk
        except Exception as e:
            log.error(f"Error generating HTML: {e}", exc_info=True)
            yield f"\n\nError: {str(e)}"

    return StreamingResponse(
        stream_response(),
        media_type="text/plain",
    )


@router.get("/templates", response_model=list[Template])
async def get_templates() -> list[Template]:
    """
    Return starter templates for common workflow patterns.
    """
    return [
        Template(
            id="minimal",
            name="Minimal",
            description="Clean, minimal interface with basic styling",
            prompt="Create a minimal, clean interface with a white background and simple form styling.",
        ),
        Template(
            id="dark",
            name="Dark Mode",
            description="Dark themed interface",
            prompt="Create a dark-themed interface with a dark background, light text, and subtle accent colors.",
        ),
        Template(
            id="gradient",
            name="Gradient",
            description="Modern gradient backgrounds",
            prompt="Create a modern interface with subtle gradient backgrounds and rounded corners.",
        ),
        Template(
            id="professional",
            name="Professional",
            description="Business/enterprise styling",
            prompt="Create a professional, enterprise-style interface suitable for business applications.",
        ),
        Template(
            id="playful",
            name="Playful",
            description="Fun, colorful interface with animations",
            prompt="Create a playful, colorful interface with fun animations and a friendly feel.",
        ),
    ]
