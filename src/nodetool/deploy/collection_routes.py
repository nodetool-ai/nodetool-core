"""
Lightweight collection index route for the FastAPI server.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Header, UploadFile

from nodetool.config.environment import Environment
from nodetool.indexing.service import index_file_to_collection


def create_collection_router() -> APIRouter:
    router = APIRouter()

    @router.post("/collections/{name}/index")
    async def index(
        name: str,
        file: UploadFile = File(...),
        authorization: Optional[str] = Header(None),
    ):
        token = "local_token"

        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, file.filename or "uploaded_file")
        try:
            with open(tmp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            file_path = tmp_path
            mime_type = file.content_type or "application/octet-stream"

            error = await index_file_to_collection(name, file_path, mime_type, token)
            if error:
                return {"path": file.filename or "unknown", "error": error}

            return {"path": file.filename or "unknown", "error": None}
        except Exception as e:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).error(f"Error indexing file {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            shutil.rmtree(tmp_dir)
            await file.close()

    return router

