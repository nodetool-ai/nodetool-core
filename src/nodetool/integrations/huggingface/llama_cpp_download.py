"""Download utilities for llama.cpp models.

Downloads GGUF files from HuggingFace to the llama.cpp native cache directory
using llama.cpp's flat file naming convention:
- {org}_{repo}_{filename}.gguf
- {org}_{repo}_{filename}.gguf.etag
- manifest={org}={repo}={tag}.json

Cache directories by platform:
- Linux: ~/.cache/llama.cpp/
- macOS: ~/Library/Caches/llama.cpp/
- Windows: %LOCALAPPDATA%/llama.cpp/
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import httpx

from nodetool.config.logging_config import get_logger
from nodetool.providers.llama_server_manager import get_llama_cpp_cache_dir

if TYPE_CHECKING:
    from typing import Callable

log = get_logger(__name__)


def get_llama_cpp_model_filename(repo_id: str, filename: str) -> str:
    """Get the llama.cpp cache filename for a model.

    llama.cpp uses flat naming: {org}_{repo}_{filename}

    Args:
        repo_id: HuggingFace repo ID (e.g., "ggml-org/gemma-3-1b-it-GGUF")
        filename: GGUF file name (e.g., "gemma-3-1b-it-Q4_K_M.gguf")

    Returns:
        Flattened filename (e.g., "ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf")
    """
    return f"{repo_id.replace('/', '_')}_{filename}"


def get_llama_cpp_model_path(repo_id: str, filename: str) -> Path:
    """Get the expected path for a model in the llama.cpp cache.

    Args:
        repo_id: HuggingFace repo ID (e.g., "ggml-org/gemma-3-1b-it-GGUF")
        filename: GGUF file name (e.g., "gemma-3-1b-it-Q4_K_M.gguf")

    Returns:
        Path to the expected model file location.
    """
    cache_dir = get_llama_cpp_cache_dir()
    flat_filename = get_llama_cpp_model_filename(repo_id, filename)
    return Path(cache_dir) / flat_filename


def is_llama_cpp_model_cached(repo_id: str, filename: str) -> bool:
    """Check if a GGUF model exists in the llama.cpp cache.

    Args:
        repo_id: HuggingFace repo ID
        filename: GGUF file name

    Returns:
        True if the model file exists, False otherwise.
    """
    model_path = get_llama_cpp_model_path(repo_id, filename)
    return model_path.exists()


async def download_llama_cpp_model(
    repo_id: str,
    filename: str,
    token: str | None = None,
    progress_callback: Callable[[int, int | None], None] | None = None,
    cancel_event: Optional[asyncio.Event] = None,
    tag: str = "latest",
) -> Path:
    """Download a GGUF model to the llama.cpp cache directory.

    Downloads directly to the llama.cpp native cache using their flat
    filename convention: {org}_{repo}_{filename}.gguf

    Also creates:
    - {flat_filename}.etag for cache validation
    - manifest={org}={repo}={tag}.json for llama.cpp compatibility

    Args:
        repo_id: HuggingFace repo ID (e.g., "ggml-org/gemma-3-1b-it-GGUF")
        filename: GGUF file name (e.g., "gemma-3-1b-it-Q4_K_M.gguf")
        token: Optional HuggingFace token for gated models
        progress_callback: Optional callback(downloaded_bytes, total_bytes)
        cancel_event: Optional event to cancel the download
        tag: Version tag (default: "latest")

    Returns:
        Path to the downloaded model file
    """
    cache_dir = get_llama_cpp_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)

    flat_filename = get_llama_cpp_model_filename(repo_id, filename)
    output_path = Path(cache_dir) / flat_filename
    etag_path = Path(cache_dir) / f"{flat_filename}.etag"

    # Manifest path: manifest={org}={repo}={tag}.json
    org, repo = repo_id.split("/", 1) if "/" in repo_id else ("", repo_id)
    manifest_path = Path(cache_dir) / f"manifest={org}={repo}={tag}.json"

    log.info(f"Downloading {repo_id}/{filename} to llama.cpp cache: {output_path}")

    # Build HuggingFace URL
    hf_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
        # Get file metadata first
        head_resp = await client.head(hf_url, headers=headers)
        head_resp.raise_for_status()

        total_size = int(head_resp.headers.get("content-length", 0)) or None
        etag_raw = head_resp.headers.get("etag", "")
        etag_stripped = etag_raw.strip('"')

        # Check if already cached with same etag
        if output_path.exists() and etag_path.exists():
            cached_etag = etag_path.read_text().strip().strip('"')
            if cached_etag == etag_stripped:
                log.info(f"Model already cached with matching etag: {output_path}")
                if progress_callback and total_size:
                    progress_callback(total_size, total_size)
                return output_path

        # Download the file
        downloaded = 0
        temp_path = output_path.with_suffix(".tmp")
        log.info(f"Downloading {hf_url} to {temp_path}")

        async with client.stream("GET", hf_url, headers=headers) as response:
            response.raise_for_status()

            with open(temp_path, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                    if cancel_event and cancel_event.is_set():
                        temp_path.unlink(missing_ok=True)
                        raise asyncio.CancelledError("Download cancelled")

                    f.write(chunk)
                    downloaded += len(chunk)

                    if progress_callback:
                        progress_callback(len(chunk), total_size)

        # Rename temp file to final location
        temp_path.rename(output_path)
        log.info(f"Moved {temp_path} to {output_path}")

        # Write etag file (keep quotes for llama-server compatibility)
        if etag_raw:
            etag_path.write_text(etag_raw)

        # Create manifest file for llama.cpp compatibility
        manifest = {
            "name": repo,
            "version": tag,
            "ggufFile": {
                "rfilename": filename,
                "size": total_size or downloaded,
            },
            "metadata": {
                "author": org,
                "repo_id": repo_id,
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))

        log.info(f"Downloaded model to: {output_path}")
        log.info(f"Created manifest: {manifest_path}")
        return output_path
