"""Asset storage utilities for auto-saving node outputs.

This module provides functions for automatically saving assets from node outputs,
including support for various URI schemes and content types.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
from io import BytesIO
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

import httpx

from nodetool.metadata.types import (
    AssetRef,
    AudioRef,
    DataframeRef,
    DocumentRef,
    ExcelRef,
    FolderRef,
    ImageRef,
    JSONRef,
    Model3DRef,
    SVGRef,
    TextRef,
    VideoRef,
)

if TYPE_CHECKING:
    from nodetool.workflows.base_node import BaseNode
    from nodetool.workflows.processing_context import ProcessingContext

logger = logging.getLogger(__name__)


def find_asset_refs(obj: Any, path: str = "") -> list[tuple[str, AssetRef]]:
    """Recursively find all AssetRef instances in an object.

    Args:
        obj: The object to scan (dict, list, tuple, or AssetRef)
        path: Current path in the object hierarchy (for tracking location)

    Returns:
        List of tuples containing (path, AssetRef) for each found asset
    """
    refs: list[tuple[str, AssetRef]] = []

    if isinstance(obj, AssetRef):
        refs.append((path, obj))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            refs.extend(find_asset_refs(value, new_path))
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            new_path = f"{path}[{idx}]"
            refs.extend(find_asset_refs(value, new_path))

    return refs


def get_content_type_for_asset_ref(asset_ref: AssetRef) -> str:
    """Get the appropriate content type for an AssetRef based on its type.

    Args:
        asset_ref: The asset reference to determine content type for

    Returns:
        MIME content type string
    """
    if isinstance(asset_ref, ImageRef):
        return "image/png"
    elif isinstance(asset_ref, AudioRef):
        return "audio/wav"
    elif isinstance(asset_ref, VideoRef):
        return "video/mp4"
    elif isinstance(asset_ref, TextRef):
        return "text/plain"
    elif isinstance(asset_ref, DocumentRef):
        return "application/pdf"
    elif isinstance(asset_ref, DataframeRef):
        return "application/json"
    elif isinstance(asset_ref, ExcelRef):
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif isinstance(asset_ref, Model3DRef):
        return "model/gltf-binary"
    elif isinstance(asset_ref, FolderRef):
        return "folder"
    elif isinstance(asset_ref, JSONRef):
        return "application/json"
    elif isinstance(asset_ref, SVGRef):
        return "image/svg+xml"
    else:
        return "application/octet-stream"


def get_extension_for_content_type(content_type: str) -> str:
    """Get the file extension for a given MIME content type.

    Args:
        content_type: MIME content type string

    Returns:
        File extension with leading dot (e.g., '.jpg', '.mp3')
    """
    extension_map = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/svg+xml": ".svg",
        "audio/mp3": ".mp3",
        "audio/mpeg": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/ogg": ".ogg",
        "audio/flac": ".flac",
        "video/mp4": ".mp4",
        "video/webm": ".webm",
        "video/quicktime": ".mov",
        "video/x-msvideo": ".avi",
        "text/plain": ".txt",
        "text/html": ".html",
        "text/css": ".css",
        "text/javascript": ".js",
        "application/json": ".json",
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "model/gltf-binary": ".glb",
        "model/gltf+json": ".gltf",
    }
    return extension_map.get(content_type, "")


def object_to_bytes(obj: Any, asset_ref: AssetRef) -> bytes | None:
    """Convert a Python object to bytes based on the asset ref type.

    Args:
        obj: The Python object to convert
        asset_ref: The asset reference (used to determine conversion method)

    Returns:
        Bytes representation of the object, or None if conversion failed
    """
    if isinstance(asset_ref, ImageRef):
        # Handle PIL Image
        try:
            from PIL import Image

            if isinstance(obj, Image.Image):
                buf = BytesIO()
                obj.save(buf, format="PNG")
                return buf.getvalue()
        except Exception as e:
            logger.debug(f"Failed to convert image object: {e}")

    elif isinstance(asset_ref, AudioRef):
        # Handle AudioSegment
        try:
            from pydub import AudioSegment

            if isinstance(obj, AudioSegment):
                buf = BytesIO()
                obj.export(buf, format="mp3")
                return buf.getvalue()
        except Exception as e:
            logger.debug(f"Failed to convert audio object: {e}")

    elif isinstance(asset_ref, TextRef):
        # Handle string
        if isinstance(obj, str):
            return obj.encode("utf-8")

    elif isinstance(asset_ref, DataframeRef):
        # Handle pandas DataFrame
        try:
            import pandas as pd

            if isinstance(obj, pd.DataFrame):
                return obj.to_json(orient="records").encode("utf-8")
        except Exception as e:
            logger.debug(f"Failed to convert dataframe object: {e}")

    # For bytes, return as-is
    if isinstance(obj, bytes):
        return obj

    return None


def convert_asset_data_to_bytes(asset_ref: AssetRef, path: str) -> BytesIO | None:
    """Convert asset_ref.data to BytesIO.

    Args:
        asset_ref: The asset reference containing data
        path: Path in the result hierarchy (for logging)

    Returns:
        BytesIO containing the data, or None if conversion failed
    """
    if isinstance(asset_ref, DataframeRef):
        # Convert DataFrame data to JSON bytes
        json_str = json.dumps(asset_ref.data)
        return BytesIO(json_str.encode("utf-8"))
    elif isinstance(asset_ref, (JSONRef, SVGRef)):
        # JSONRef and SVGRef have string data
        if isinstance(asset_ref.data, str):
            return BytesIO(asset_ref.data.encode("utf-8"))
        elif isinstance(asset_ref.data, bytes):
            return BytesIO(asset_ref.data)
        else:
            logger.warning("JSONRef/SVGRef data is not string or bytes at %s", path)
            return None
    elif isinstance(asset_ref.data, bytes):
        return BytesIO(asset_ref.data)
    else:
        # Try to convert to bytes
        try:
            return BytesIO(bytes(asset_ref.data))
        except Exception:
            logger.warning("Could not convert data to bytes for asset at %s", path)
            return None


def resolve_memory_uri(asset_ref: AssetRef, path: str) -> BytesIO | None:
    """Resolve a memory:// URI to BytesIO.

    Args:
        asset_ref: The asset reference with a memory:// URI
        path: Path in the result hierarchy (for logging)

    Returns:
        BytesIO containing the data, or None if resolution failed
    """
    from nodetool.runtime.resources import require_scope

    scope = require_scope()
    obj = scope.get_memory_uri_cache().get(asset_ref.uri)
    if obj is not None:
        data_bytes = object_to_bytes(obj, asset_ref)
        if data_bytes:
            return BytesIO(data_bytes)
        else:
            logger.warning(
                "Could not convert memory object to bytes for asset at %s", path
            )
            return None
    else:
        logger.warning("Memory URI not found in cache for asset at %s", path)
        return None


async def download_http_uri(uri: str, path: str) -> BytesIO | None:
    """Download content from an HTTP/HTTPS URL.

    Args:
        uri: The HTTP/HTTPS URL to download
        path: Path in the result hierarchy (for logging)

    Returns:
        BytesIO containing the downloaded data, or None if download failed
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(uri)
            response.raise_for_status()
            return BytesIO(response.content)
    except httpx.HTTPError as http_err:
        logger.warning(
            "Failed to download asset from URL %s at %s: %s", uri, path, http_err
        )
        return None


def read_file_uri(uri: str, path: str) -> BytesIO | None:
    """Read content from a file:// URI.

    Args:
        uri: The file:// URI to read
        path: Path in the result hierarchy (for logging)

    Returns:
        BytesIO containing the file data, or None if read failed
    """
    try:
        parsed = urlparse(uri)
        # Handle both file:///path and file://localhost/path
        file_path = unquote(parsed.path)
        with open(file_path, "rb") as f:
            return BytesIO(f.read())
    except OSError as file_err:
        logger.warning("Failed to read file from %s at %s: %s", uri, path, file_err)
        return None


def decode_data_uri(uri: str, path: str) -> BytesIO | None:
    """Decode a data: URI.

    Args:
        uri: The data: URI to decode (e.g., data:image/png;base64,...)
        path: Path in the result hierarchy (for logging)

    Returns:
        BytesIO containing the decoded data, or None if decoding failed
    """
    try:
        # Format: data:[<mediatype>][;base64],<data>
        header, encoded_data = uri.split(",", 1)
        data_bytes = base64.b64decode(encoded_data) if ";base64" in header else unquote(encoded_data).encode("utf-8")
        return BytesIO(data_bytes)
    except (ValueError, binascii.Error) as data_err:
        logger.warning("Failed to decode data URI at %s: %s", path, data_err)
        return None


async def resolve_asset_content(
    asset_ref: AssetRef, path: str
) -> BytesIO | None:
    """Resolve asset content from data or URI.

    Supports:
    - Direct data on the asset_ref
    - memory:// URIs
    - http:// and https:// URIs
    - file:// URIs
    - data: URIs

    Args:
        asset_ref: The asset reference to resolve
        path: Path in the result hierarchy (for logging)

    Returns:
        BytesIO containing the asset data, or None if resolution failed
    """
    # Handle direct data
    if asset_ref.data:
        return convert_asset_data_to_bytes(asset_ref, path)

    # Handle various URI schemes
    uri = asset_ref.uri
    if not uri:
        logger.debug("Skipping asset at %s - no data or uri", path)
        return None

    if uri.startswith("memory://"):
        return resolve_memory_uri(asset_ref, path)
    elif uri.startswith("http://") or uri.startswith("https://"):
        return await download_http_uri(uri, path)
    elif uri.startswith("file://"):
        return read_file_uri(uri, path)
    elif uri.startswith("data:"):
        return decode_data_uri(uri, path)
    else:
        logger.debug("Skipping asset at %s - unsupported URI type: %s", path, uri)
        return None


async def auto_save_assets(
    node: BaseNode,
    result: dict[str, Any],
    context: ProcessingContext,
) -> None:
    """Automatically save assets from node outputs.

    Scans the result dictionary for AssetRef instances and saves them to storage
    with proper tracking (node_id, job_id, workflow_id).

    Args:
        node: The node that produced the result
        result: The result dictionary containing node outputs
        context: The processing context with workflow/job information
    """
    if not result:
        return

    logger.debug(
        "Auto-saving assets for node %s (%s)",
        node.get_title(),
        node._id,
    )

    asset_refs = find_asset_refs(result)

    if not asset_refs:
        logger.debug(
            "No AssetRefs found in result for node %s (%s)",
            node.get_title(),
            node._id,
        )
        return

    logger.info(
        "Found %d asset(s) to auto-save for node %s (%s)",
        len(asset_refs),
        node.get_title(),
        node._id,
    )

    for path, asset_ref in asset_refs:
        try:
            # Skip if asset already has an asset_id (already saved)
            if asset_ref.asset_id:
                logger.debug(
                    "Skipping asset at %s - already has asset_id: %s",
                    path,
                    asset_ref.asset_id,
                )
                continue

            # Skip if no data to save
            if not asset_ref.data and not asset_ref.uri:
                logger.debug("Skipping asset at %s - no data or uri", path)
                continue

            # Get content type and resolve content
            content_type = get_content_type_for_asset_ref(asset_ref)
            content = await resolve_asset_content(asset_ref, path)

            if content is None:
                continue

            # Generate asset name
            asset_name = f"{node.get_title()}_{path}_{node._id[:8]}"

            # Create and save the asset
            asset = await context.create_asset(
                name=asset_name,
                content_type=content_type,
                content=content,
                node_id=node._id,
            )

            # Update the AssetRef with the new asset_id and canonical URI (with extension)
            extension = get_extension_for_content_type(content_type)
            asset_ref.asset_id = asset.id
            asset_ref.uri = f"asset://{asset.id}{extension}"

            logger.info(
                "Auto-saved asset %s for node %s (%s) at %s",
                asset.id,
                node.get_title(),
                node._id,
                path,
            )

        except Exception as e:
            logger.error(
                "Failed to auto-save asset at %s for node %s (%s): %s",
                path,
                node.get_title(),
                node._id,
                e,
                exc_info=True,
            )
