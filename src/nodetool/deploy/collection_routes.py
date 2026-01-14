"""
Lightweight collection index route for the FastAPI server.
"""

from __future__ import annotations

import os
import shutil
import tempfile

from fastapi import APIRouter, File, Header, HTTPException, UploadFile

from nodetool.config.logging_config import get_logger
from nodetool.indexing.service import index_file_to_collection

log = get_logger(__name__)


def create_collection_router() -> APIRouter:
    router = APIRouter()

    @router.post("/collections/{name}/index")
    async def index(
        name: str,
        file: UploadFile = File(...),
        authorization: str | None = Header(None),
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
        except Exception as e:
            log.error(f"Error indexing file {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e
        finally:
            shutil.rmtree(tmp_dir)
            await file.close()

    return router
