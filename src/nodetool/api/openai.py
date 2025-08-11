from typing import List

import json
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from rich.console import Console

from nodetool.api.model import get_language_models
from nodetool.api.workflow import from_model
from nodetool.chat.chat_sse_runner import ChatSSERunner
from nodetool.types.workflow import Workflow
from nodetool.models.workflow import Workflow as WorkflowModel

console = Console()


def create_openai_compatible_router(
    provider: str,
    default_model: str = "gpt-oss:20b",
    tools: List[str] | None = None,
) -> APIRouter:
    """Create an APIRouter exposing OpenAI-compatible endpoints.

    Endpoints:
      - POST /v1/chat/completions
      - GET /v1/models
    """

    router = APIRouter(prefix="/v1")

    tools = tools or []

    @router.post("/chat/completions")
    async def openai_chat_completions(request: Request):
        """OpenAI-compatible chat completions endpoint mirroring /chat/sse behaviour."""
        try:
            data = await request.json()
            auth_header = request.headers.get("authorization", "")
            auth_token = (
                auth_header.replace("Bearer ", "")
                if auth_header.startswith("Bearer ")
                else None
            )
            if auth_token:
                data["auth_token"] = auth_token
            
            workflows, _ = WorkflowModel.paginate(limit=1000)

            runner = ChatSSERunner(
                auth_token,
                default_model=default_model,
                default_provider=provider,
                workflows=[from_model(workflow) for workflow in workflows],
            )

            # Determine if the client requested streaming (default true)
            stream = data.get("stream", True)
            if not stream:
                # Collect the streamed chunks into a single response object
                chunks: List[str] = []
                async for event in runner.process_single_request(data):
                    if event.startswith("data: "):
                        payload = event[len("data: ") :].strip()
                        if payload == "[DONE]":
                            break
                        chunks.append(payload)
                if chunks:
                    return json.loads(chunks[-1])
                return {}
            else:
                return StreamingResponse(
                    runner.process_single_request(data),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    },
                )
        except Exception as e:  # noqa: BLE001
            console.print(f"OpenAI Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.get("/models")
    async def openai_models():
        """Returns list of models filtered by provider in OpenAI format."""
        try:
            all_models = await get_language_models()
            filtered = [
                m
                for m in all_models
                if (
                    (m.provider.value if hasattr(m.provider, "value") else m.provider)
                    == provider
                )
            ]
            data = [
                {
                    "id": m.id or m.name,
                    "object": "model",
                    "created": 0,
                    "owned_by": provider,
                }
                for m in filtered
            ]
            return {"object": "list", "data": data}
        except Exception as e:  # noqa: BLE001
            console.print(f"OpenAI Models error: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    return router


