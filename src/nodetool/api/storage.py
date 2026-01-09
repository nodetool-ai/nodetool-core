#!/usr/bin/env python

import os
import re
from datetime import UTC, timezone
from email.utils import parsedate_to_datetime
from io import BytesIO
from tempfile import SpooledTemporaryFile
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from nodetool.api.utils import CurrentUser, current_user
from nodetool.config.logging_config import get_logger
from nodetool.runtime.resources import require_scope
from nodetool.types.content_types import EXTENSION_TO_CONTENT_TYPE

log = get_logger(__name__)
router = APIRouter(prefix="/api/storage", tags=["storage"])
temp_router = APIRouter(prefix="/api/storage/temp", tags=["temp"])


def validate_key(key: str) -> str:
    """
    Validate and normalize a storage key, preventing traversal while allowing nested prefixes.

    Returns the normalized key (POSIX-style separators). Raises HTTPException if validation fails.
    """
    normalized = key.replace("\\", "/")

    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if not parts:
        raise HTTPException(status_code=400, detail="Invalid key: key must not be empty")

    if normalized.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid key: absolute paths are not allowed")

    if any(part == ".." for part in parts):
        raise HTTPException(status_code=400, detail="Invalid key: path traversal is not allowed")

    return "/".join(parts)


async def _head_file(storage, key: str):
    """
    Common logic for returning file metadata.
    """
    safe_key = validate_key(key)
    if not await storage.file_exists(safe_key):
        raise HTTPException(status_code=404)

    last_modified = await storage.get_mtime(safe_key)
    if not last_modified:
        raise HTTPException(status_code=404)

    return Response(
        status_code=200,
        headers={
            "Last-Modified": last_modified.strftime("%a, %d %b %Y %H:%M:%S GMT"),
        },
    )


async def _get_file(storage, key: str, request: Request):
    """
    Common logic for returning file as a stream with range support.
    """
    safe_key = validate_key(key)
    if not await storage.file_exists(safe_key):
        raise HTTPException(status_code=404)

    last_modified = await storage.get_mtime(safe_key)
    if not last_modified:
        raise HTTPException(status_code=404)

    if "If-Modified-Since" in request.headers:
        if_modified_since = parsedate_to_datetime(request.headers["If-Modified-Since"])
        last_modified = last_modified.replace(tzinfo=UTC)
        if if_modified_since >= last_modified:
            raise HTTPException(status_code=304)

    ext = os.path.splitext(safe_key)[-1][1:]
    media_type = EXTENSION_TO_CONTENT_TYPE.get(ext, "application/octet-stream")
    headers = {
        "Last-Modified": last_modified.strftime("%a, %d %b %Y %H:%M:%S GMT"),
        "Accept-Ranges": "bytes",
        "Content-Type": media_type,
    }

    range_header = request.headers.get("Range")
    start: Optional[int] = 0
    end: Optional[int] = None

    if range_header:
        try:
            range_match = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if range_match:
                start = int(range_match.group(1))
                end_str = range_match.group(2)
                end = int(end_str)
            else:
                raise ValueError("Invalid range format")

            stream = BytesIO()
            await storage.download(safe_key, stream)
            data = stream.getvalue()

            headers["Content-Range"] = f"bytes {start}-{end}/{len(data)}"
            headers["Content-Length"] = str(end - start + 1)

            return Response(
                content=data[start : end + 1],
                status_code=206,
                headers=headers,
            )
        except ValueError:
            # If range is invalid, ignore it and return full content
            pass

    size = await storage.get_size(safe_key)
    headers["Content-Length"] = str(size)

    return StreamingResponse(
        content=storage.download_stream(safe_key),
        headers=headers,
    )


async def _put_file(storage, key: str, request: Request):
    """
    Common logic for uploading/updating files.
    """
    safe_key = validate_key(key)
    with SpooledTemporaryFile(max_size=10 * 1024 * 1024) as buffer:  # spools to disk if large
        async for chunk in request.stream():
            buffer.write(chunk)
        buffer.seek(0)
        await storage.upload(safe_key, buffer)

    # return the same xml response as aws s3 upload_fileobj
    return Response(status_code=200, content=b"")


async def _delete_file(storage, key: str):
    """
    Common logic for deleting files.
    """
    safe_key = validate_key(key)
    if not await storage.file_exists(safe_key):
        return Response(status_code=404)
    await storage.delete(safe_key)


# Asset storage endpoints
@router.head("/{key}")
async def head(key: str):
    """
    Returns the metadata for the file with the given key.
    """
    storage = require_scope().get_asset_storage()
    return await _head_file(storage, key)


@router.get("/{key}")
async def get(key: str, request: Request):
    """
    Returns the file as a stream for the given key, supporting range queries.
    """
    storage = require_scope().get_asset_storage()
    return await _get_file(storage, key, request)


@router.put("/{key}")
async def update(key: str, request: Request, _user: str = Depends(CurrentUser())):
    """
    Updates or creates the file for the given key.
    """
    storage = require_scope().get_asset_storage()
    return await _put_file(storage, key, request)


@router.delete("/{key}")
async def delete(key: str, _user: str = Depends(CurrentUser())):
    """
    Deletes the asset for the given key.
    """
    storage = require_scope().get_asset_storage()
    return await _delete_file(storage, key)


# Temp storage endpoints
@temp_router.head("/{key}")
async def temp_head(key: str):
    """
    Returns the metadata for the temp file with the given key.
    """
    storage = require_scope().get_temp_storage()
    return await _head_file(storage, key)


@temp_router.get("/{key}")
async def temp_get(key: str, request: Request):
    """
    Returns the temp file as a stream for the given key, supporting range queries.
    """
    storage = require_scope().get_temp_storage()
    return await _get_file(storage, key, request)


@temp_router.put("/{key}")
async def temp_update(key: str, request: Request, _user: str = Depends(CurrentUser())):
    """
    Updates or creates the temp file for the given key.
    """
    storage = require_scope().get_temp_storage()
    return await _put_file(storage, key, request)


@temp_router.delete("/{key}")
async def temp_delete(key: str, _user: str = Depends(CurrentUser())):
    """
    Deletes the temp asset for the given key.
    """
    storage = require_scope().get_temp_storage()
    return await _delete_file(storage, key)
