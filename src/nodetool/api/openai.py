import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from nodetool.api.utils import current_user
from nodetool.api.workflow import from_model
from nodetool.chat.chat_sse_runner import ChatSSERunner
from nodetool.config.logging_config import get_logger
from nodetool.ml.models.language_models import get_all_language_models
from nodetool.models.workflow import Workflow as WorkflowModel
from nodetool.runtime.resources import get_static_auth_provider

log = get_logger(__name__)


def create_openai_compatible_router(
    provider: str,
    default_model: str = "gpt-oss:20b",
    tools: list[str] | None = None,
) -> APIRouter:
    """Create an APIRouter exposing OpenAI-compatible endpoints.

    Endpoints:
      - POST /v1/chat/completions
      - GET /v1/models
    """

    router = APIRouter(prefix="/v1")

    tools = tools or []

    @router.post("/chat/completions")
    async def openai_chat_completions(request: Request, user: str = Depends(current_user)):
        """OpenAI-compatible chat completions endpoint mirroring /chat/sse behaviour."""
        try:
            data = await request.json()
            static_provider = get_static_auth_provider()
            auth_token = static_provider.extract_token_from_headers(request.headers)
            if auth_token:
                data["auth_token"] = auth_token
            data["user_id"] = user

            workflows, _ = await WorkflowModel.paginate(user_id=user, limit=1000)

            runner = ChatSSERunner(
                auth_token,
                default_model=default_model,
                default_provider=provider,
                workflows=[from_model(workflow) for workflow in workflows],
            )
            runner.user_id = user

            # Determine if the client requested streaming (default true)
            stream = data.get("stream", True)
            if not stream:
                # Collect the streamed chunks into a single response object
                chunks: list[str] = []
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
        except Exception as e:
            log.error(f"OpenAI Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.get("/models")
    async def openai_models(user: str = Depends(current_user)):
        """Returns list of models in OpenAI format."""
        try:
            all_models = await get_all_language_models(user)
            data = [
                {
                    "id": m.id or m.name,
                    "object": "model",
                    "created": 0,
                    "owned_by": m.provider.value,
                }
                for m in all_models
            ]
            return {"object": "list", "data": data}
        except Exception as e:
            log.error(f"OpenAI Models error: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    return router
