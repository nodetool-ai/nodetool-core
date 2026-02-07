from __future__ import annotations

import logging
from io import BytesIO
from typing import IO, cast

import aiohttp
import httpx
import numpy as np
import PIL.Image

from nodetool.media.image.image_utils import (
    numpy_to_pil_image,
    pil_to_png_bytes,
)
from nodetool.runtime.resources import require_scope

log = logging.getLogger(__name__)


def _normalize_image_like_to_png_bytes(obj: object) -> bytes:
    """
    Best-effort conversion of common image-like objects to PNG bytes.
    Supports PIL.Image.Image, numpy.ndarray, raw bytes (if image), and file-like.
    Raises ValueError if the object cannot be converted.
    """
    if isinstance(obj, PIL.Image.Image):
        return pil_to_png_bytes(obj)

    if isinstance(obj, np.ndarray):
        pil_img = numpy_to_pil_image(obj)
        return pil_to_png_bytes(pil_img)

    if isinstance(obj, bytes | bytearray):
        raw = bytes(obj)
        try:
            with PIL.Image.open(BytesIO(raw)) as img:
                return pil_to_png_bytes(img)
        except Exception as e:
            raise ValueError(f"Bytes are not a decodable image: {e}") from e

    if hasattr(obj, "read"):
        file_like = cast("IO[bytes]", obj)
        try:
            pos = file_like.tell() if hasattr(file_like, "tell") else None
            if hasattr(file_like, "seek"):
                file_like.seek(0)
            raw = file_like.read()
            if pos is not None and hasattr(file_like, "seek"):
                file_like.seek(pos)
            with PIL.Image.open(BytesIO(raw)) as img:
                return pil_to_png_bytes(img)
        except Exception as e:
            raise ValueError(f"File-like object could not be decoded as image: {e}") from e

    raise ValueError(f"Unsupported object type for image conversion: {type(obj)}")


def _parse_data_uri(uri: str) -> tuple[str, bytes]:
    """Parse a data: URI and return (mime, bytes)."""
    try:
        header, b64data = uri.split(",", 1)
        mime_type = "application/octet-stream"
        if ";" in header:
            meta = header[5:]
            parts = meta.split(";")
            if parts and "/" in parts[0]:
                mime_type = parts[0]
        import base64

        raw = base64.b64decode(b64data.encode("utf-8") if not isinstance(b64data, bytes) else b64data)
        return mime_type, raw
    except Exception as e:
        # Tests expect the phrase "Invalid data URI" to appear
        raise ValueError(f"Invalid data URI: {e}") from e


def _fetch_file_uri(uri: str) -> tuple[str, bytes]:
    """Read file:// URI and return (mime, bytes).

    Use builtins.open so tests can mock file IO with patch("builtins.open").
    """
    import mimetypes
    import pathlib

    path = uri[len("file://") :]
    # Normalize path
    p = pathlib.Path(path)
    with open(p, "rb") as f:  # type: ignore[arg-type]
        data = f.read()
    mime_type, _ = mimetypes.guess_type(uri)
    if not mime_type:
        mime_type = "application/octet-stream"
    return mime_type, data


async def _fetch_http_uri_async(uri: str) -> tuple[str, bytes]:
    """Fetch content from an HTTP/HTTPS URL. Local storage URLs are handled by the caller."""
    async with aiohttp.ClientSession() as session, session.get(uri) as response:
        response.raise_for_status()
        data = await response.read()
        content_type = response.headers.get("Content-Type")
        mime_type: str | None = None
        if content_type:
            mime_type = content_type.split(";", 1)[0]
        if not mime_type:
            import mimetypes

            mime_type, _ = mimetypes.guess_type(uri)
        if not mime_type:
            mime_type = "application/octet-stream"
        return mime_type, data


def _is_local_storage_url(uri: str) -> bool:
    """Check if this is a local storage URL that should be read directly from storage."""
    import re

    # Match localhost or 127.0.0.1 with any port, accessing /api/storage/
    pattern = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?/api/storage/"
    return bool(re.match(pattern, uri))


def _extract_storage_key_from_url(uri: str) -> str:
    """Extract the storage key from a local storage URL."""
    import re

    # Extract the key from URLs like: http://localhost:7777/api/storage/828ae5ded94411f0884a000022ae8b15.png
    match = re.search(r"/api/storage/(.+)$", uri)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract storage key from URL: {uri}")


async def _fetch_local_storage_async(uri: str) -> tuple[str, bytes]:
    """Read directly from local storage instead of making HTTP request (async version)."""
    import mimetypes

    key = _extract_storage_key_from_url(uri)

    # Get storage from the current scope
    scope = require_scope()
    storage = scope.get_asset_storage()

    exists = await storage.file_exists(key)
    if not exists:
        raise ValueError(f"Storage file not found: {key}")

    # Download the file
    stream = BytesIO()
    await storage.download(key, stream)
    data = stream.getvalue()

    # Guess mime type from the key
    mime_type, _ = mimetypes.guess_type(key)
    if not mime_type:
        mime_type = "application/octet-stream"

    return mime_type, data


def _fetch_memory_uri(uri: str) -> tuple[str, bytes]:
    obj = None
    try:
        obj = require_scope().get_memory_uri_cache().get(uri)
    except Exception as e:
        log.debug(f"Failed to get from memory URI cache: {e}")
    if obj is None:
        raise ValueError(f"No cached object for memory URI: {uri}")

    # Prefer image normalization when possible
    try:
        data = _normalize_image_like_to_png_bytes(obj)
        return "image/png", data
    except Exception as e:
        log.debug(f"Failed to normalize image: {e}")

    # Fallbacks
    if isinstance(obj, bytes | bytearray):
        return "application/octet-stream", bytes(obj)

    raise ValueError(f"Unsupported object type for memory URI: {type(obj)}")


def _parse_asset_id_from_uri(uri: str) -> str:
    """Parse asset ID from asset:// URI.

    Format: asset://{asset_id} or asset://{asset_id}.{extension}

    Args:
        uri: The asset:// URI to parse

    Returns:
        The asset ID
    """
    if not uri.startswith("asset://"):
        raise ValueError(f"Invalid asset URI: {uri}")

    # Remove the asset:// prefix
    path = uri[len("asset://"):]

    # Remove extension if present (everything after the first dot)
    asset_id = path.split(".")[0] if "." in path else path

    if not asset_id:
        raise ValueError(f"Invalid asset URI - no asset ID: {uri}")

    return asset_id


async def _fetch_asset_uri_async(uri: str) -> tuple[str, bytes]:
    """Fetch content from an asset:// URI (async version).

    Args:
        uri: The asset:// URI to fetch

    Returns:
        Tuple of (mime_type, data_bytes)
    """
    from nodetool.models.asset import Asset

    asset_id = _parse_asset_id_from_uri(uri)

    # Fetch the asset from the database
    asset = await Asset.get(asset_id)
    if not asset:
        raise ValueError(f"Asset not found: {asset_id}")

    # Get storage and download the file
    scope = require_scope()
    storage = scope.get_asset_storage()

    exists = await storage.file_exists(asset.file_name)
    if not exists:
        raise ValueError(f"Asset file not found in storage: {asset.file_name}")

    stream = BytesIO()
    await storage.download(asset.file_name, stream)
    data = stream.getvalue()

    # Use the asset's content type
    mime_type = asset.content_type if asset.content_type else "application/octet-stream"

    return mime_type, data


async def fetch_uri_bytes_and_mime_async(uri: str) -> tuple[str, bytes]:
    """
    Fetch content from a URI and return (mime_type, data_bytes).
    Supports data:, memory://, file://, asset://, and http(s) URIs.
    """
    if uri.startswith("data:"):
        return _parse_data_uri(uri)
    if uri.startswith("memory://"):
        return _fetch_memory_uri(uri)
    if uri.startswith("file://"):
        return _fetch_file_uri(uri)
    if uri.startswith("asset://"):
        return await _fetch_asset_uri_async(uri)
    if uri.startswith("http://") or uri.startswith("https://"):
        # Check for local storage URLs first - read directly instead of HTTP call
        if _is_local_storage_url(uri):
            return await _fetch_local_storage_async(uri)
        return await _fetch_http_uri_async(uri)
    # Explicitly reject unsupported schemes (e.g., ftp)
    raise ValueError(f"Unsupported URI scheme: {uri.split(':', 1)[0]}://")
