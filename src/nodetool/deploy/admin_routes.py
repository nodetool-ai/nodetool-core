"""
Admin routes for the lightweight NodeTool FastAPI server.

This module encapsulates endpoints under /admin, including:
- HuggingFace model download (SSE)
- Ollama model download (SSE)
- Cache scan and size
- Delete HF model
- Workflow registry status
"""

from __future__ import annotations

import json
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from nodetool.deploy.admin_operations import (
    download_hf_model,
    download_ollama_model,
    scan_hf_cache,
    calculate_cache_size,
    delete_hf_model,
)
from nodetool.config.environment import Environment
from typing import Any, Dict

from nodetool.models.asset import Asset
from nodetool.models.database_adapter import DatabaseAdapter
from nodetool.models.workflow import Workflow


def get_model_adapter(table: str) -> DatabaseAdapter:
    if table == "workflows":
        return Workflow.adapter()
    elif table == "assets":
        return Asset.adapter()
    else:
        raise ValueError(f"Unknown table: {table}")


def create_admin_router() -> APIRouter:
    router = APIRouter()

    @router.post("/admin/models/huggingface/download")
    async def download_huggingface_model_endpoint(request: Request):
        """Download HuggingFace model with optional streaming progress."""
        try:
            data = await request.json()
            repo_id = data.get("repo_id")

            if not repo_id:
                raise HTTPException(status_code=400, detail="repo_id is required")

            async def generate_sse():
                try:
                    async for chunk in download_hf_model(
                        repo_id=repo_id,
                        cache_dir=data.get("cache_dir", "/app/.cache/huggingface/hub"),
                        file_path=data.get("file_path"),
                        ignore_patterns=data.get("ignore_patterns"),
                        allow_patterns=data.get("allow_patterns"),
                        stream=data.get("stream", True),
                    ):
                        yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:  # noqa: BLE001
                    error_data = {"status": "error", "error": str(e)}
                    yield f"data: {json.dumps(error_data)}\n\n"

            return StreamingResponse(
                generate_sse(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Authorization, Content-Type",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                },
            )
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"HuggingFace download error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/admin/models/ollama/download")
    async def download_ollama_model_endpoint(request: Request):
        """Download Ollama model with optional streaming progress."""
        try:
            data = await request.json()
            model_name = data.get("model_name")

            if not model_name:
                raise HTTPException(status_code=400, detail="model_name is required")

            async def generate_sse():
                try:
                    async for chunk in download_ollama_model(
                        model_name=model_name, stream=data.get("stream", True)
                    ):
                        yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:  # noqa: BLE001
                    error_data = {"status": "error", "error": str(e)}
                    yield f"data: {json.dumps(error_data)}\n\n"

            return StreamingResponse(
                generate_sse(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Authorization, Content-Type",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                },
            )
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"Ollama download error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/admin/cache/scan")
    async def scan_cache():
        """Scan HuggingFace cache directory."""
        try:
            results = []
            async for chunk in scan_hf_cache():
                results.append(chunk)
            return results[0] if results else {"status": "error", "message": "No cache data"}
        except Exception as e:  # noqa: BLE001
            print(f"Cache scan error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/admin/cache/size")
    async def get_cache_size(cache_dir: str = "/app/.cache/huggingface/hub"):
        """Calculate total cache size."""
        try:
            results = []
            async for chunk in calculate_cache_size(cache_dir=cache_dir):
                results.append(chunk)
            return results[0] if results else {"status": "error", "message": "No size data"}
        except Exception as e:  # noqa: BLE001
            print(f"Cache size calculation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/admin/models/huggingface/{repo_id:path}")
    async def delete_huggingface_model_endpoint(repo_id: str):
        """Delete HuggingFace model from cache."""
        try:
            results = []
            async for chunk in delete_hf_model(repo_id=repo_id):
                results.append(chunk)
            return results[0] if results else {"status": "error", "message": "Delete failed"}
        except Exception as e:  # noqa: BLE001
            print(f"HuggingFace model deletion error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


    # Database adapter operations
    @router.post("/admin/db/{table}/save")
    async def db_save(table: str, item: Dict[str, Any]):
        """Save an item to the specified table using the database adapter."""
        try:
            adapter = get_model_adapter(table)
            adapter.save(item)
            return {"status": "ok"}
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/admin/db/{table}/{key}")
    async def db_get(table: str, key: str):
        """Get an item by primary key from the specified table."""
        try:
            adapter = get_model_adapter(table)
            item = adapter.get(key)
            if item is None:
                raise HTTPException(status_code=404, detail="Not found")
            return item
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/admin/db/{table}/{key}")
    async def db_delete(table: str, key: str):
        """Delete an item by primary key from the specified table."""
        try:
            adapter = get_model_adapter(table)
            adapter.delete(key)
            return {"status": "ok"}
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e))

    return router


