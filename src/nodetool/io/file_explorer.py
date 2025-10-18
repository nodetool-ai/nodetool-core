from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Optional

from huggingface_hub.constants import HF_HUB_CACHE

from nodetool.config.logging_config import get_logger


log = get_logger(__name__)


def get_ollama_models_dir() -> Optional[Path]:
    """Return the Ollama models directory if it can be determined.

    Order of precedence:
    - Respect `OLLAMA_MODELS` env var if present (expanded, best-effort resolved).
    - OS-specific default locations.

    Returns None if a reasonable location cannot be determined.
    """
    # 1. Env override takes precedence
    custom_path = os.environ.get("OLLAMA_MODELS")
    if custom_path:
        try:
            p = Path(custom_path).expanduser()
            try:
                p = p.resolve(strict=False)
            except Exception:
                # Non-fatal: path may not fully resolve if not yet created
                pass
            log.debug(f"Using Ollama models directory from OLLAMA_MODELS env var: {p}")
            return p
        except Exception as e:
            log.error(
                f"Failed to process OLLAMA_MODELS environment variable '{custom_path}': {e}"
            )

    try:
        if sys.platform == "win32":
            path = Path(os.environ["USERPROFILE"]) / ".ollama" / "models"
        elif sys.platform == "darwin":
            path = Path.home() / ".ollama" / "models"
        else:  # Linux and other UNIX-like
            user_path = Path.home() / ".ollama" / "models"
            if user_path.exists() and user_path.is_dir():
                path = user_path
            else:
                path = Path("/usr/share/ollama/.ollama/models")

        if path and path.exists() and path.is_dir():
            return path.resolve()
        return None
    except Exception as e:
        log.error(f"Error determining Ollama models directory: {e}")
        return None


def get_valid_explorable_roots() -> List[Path]:
    """Return a list of safe roots allowed to open in file explorers.

    Currently includes the Ollama models directory and the Hugging Face hub
    cache directory when they can be determined.
    """
    safe_roots: list[Path] = []

    ollama_dir = get_ollama_models_dir()
    if isinstance(ollama_dir, Path):
        safe_roots.append(ollama_dir)

    try:
        hf_cache_path = Path(HF_HUB_CACHE).resolve()
        if hf_cache_path.exists() and hf_cache_path.is_dir():
            safe_roots.append(hf_cache_path)
        else:
            log.warning(
                f"Hugging Face cache directory {hf_cache_path} does not exist or is not a directory."
            )
    except Exception as e:
        log.error(f"Error determining Hugging Face cache directory: {e}")

    return safe_roots


async def open_in_explorer(path: str) -> dict:
    """Open the given path in the system file explorer if within safe roots.

    Returns a dict with either a success payload or an error message.
    This function performs OS-specific logic and uses async subprocess calls
    to avoid blocking the event loop.
    """
    safe_roots = get_valid_explorable_roots()
    if not safe_roots:
        return {
            "status": "error",
            "message": "Cannot open path: No safe directories (like Ollama or Hugging Face cache) could be determined.",
        }

    path_to_open: Optional[str] = None
    try:
        requested_path = Path(path).resolve()
        is_safe_path = any(
            requested_path.is_relative_to(root_dir) for root_dir in safe_roots
        )

        if not is_safe_path:
            log.warning(
                "Path traversal attempt: User path %s is not within any of the configured safe directories: %s",
                requested_path,
                safe_roots,
            )
            return {
                "status": "error",
                "message": "Access denied: Path is outside the allowed directories.",
            }

        path_to_open = str(requested_path)

        if sys.platform == "win32":
            proc = await asyncio.create_subprocess_exec("explorer", path_to_open)
        elif sys.platform == "darwin":
            proc = await asyncio.create_subprocess_exec("open", path_to_open)
        else:
            proc = await asyncio.create_subprocess_exec("xdg-open", path_to_open)

        return_code = await proc.wait()
        if return_code != 0:
            raise RuntimeError(f"Explorer command exited with code {return_code}")

        return {"status": "success", "path": path_to_open}
    except Exception as e:
        log.error(
            "Failed to open path %s in explorer: %s",
            path_to_open if path_to_open else path,
            e,
        )
        return {
            "status": "error",
            "message": "An internal error occurred while attempting to open the path. Please check server logs for details.",
        }
