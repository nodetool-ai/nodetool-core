from typing import List

import json
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from rich.console import Console

from nodetool.api.model import get_language_models
from nodetool.chat.chat_sse_runner import ChatSSERunner
from nodetool.types.workflow import Workflow


console = Console()


def create_openai_compatible_router(
    provider: str,
    default_model: str = "gemma3n:latest",
    tools: List[str] | None = None,
    workflows: List[Workflow] | None = None,
) -> APIRouter:
    """Create an APIRouter exposing OpenAI-compatible endpoints.

    Endpoints:
      - POST /v1/chat/completions
      - GET /v1/models
    """

    router = APIRouter(prefix="/v1")

    tools = tools or []
    workflows = workflows or []

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

            runner = ChatSSERunner(
                auth_token,
                default_model=default_model,
                default_provider=provider,
                tools=tools,
                workflows=workflows,
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
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "Authorization, Content-Type",
                        "Access-Control-Allow-Methods": "POST, OPTIONS",
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


