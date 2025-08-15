from __future__ import annotations

import platform
import sys
from importlib import metadata
from typing import Dict, Optional
from pathlib import Path

from nodetool.common.settings import (
    get_log_path,
    get_system_data_path,
    get_system_file_path,
    SETTINGS_FILE,
    SECRETS_FILE,
)


def get_os_info() -> Dict[str, str]:
    return {
        "platform": sys.platform,
        "release": platform.release(),
        "arch": platform.machine(),
    }


def _safe_version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except Exception:
        return None


def get_versions_info() -> Dict[str, str | None]:
    # CUDA version (best-effort via PyTorch)
    cuda_version: Optional[str] = None
    try:
        import torch  # type: ignore

        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        if not cuda_version and hasattr(torch, "version"):
            # Try nvcc version via torch if available
            nv: Optional[str] = getattr(torch.version, "nvcc", None)  # type: ignore
            if nv:
                cuda_version = nv
    except Exception:
        cuda_version = None

    return {
        "python": platform.python_version(),
        "nodetool_core": _safe_version("nodetool-core"),
        "nodetool_base": _safe_version("nodetool-base"),
        "cuda": cuda_version,
    }


def get_paths_info() -> Dict[str, str]:
    settings_path = str(get_system_file_path(SETTINGS_FILE))
    secrets_path = str(get_system_file_path(SECRETS_FILE))
    data_dir = str(get_system_data_path(""))
    core_logs_dir = str(get_system_data_path("logs"))
    core_log_file = str(get_log_path("nodetool.log"))

    # Additional caches/paths
    # Hugging Face cache (root "hub" path)
    try:
        from huggingface_hub.constants import HF_HUB_CACHE  # type: ignore

        huggingface_cache_dir = str(Path(HF_HUB_CACHE).resolve())
    except Exception:
        huggingface_cache_dir = ""

    # Ollama models directory
    try:
        # Internal helper determines OS-specific location
        from nodetool.api.model import _get_ollama_models_dir  # type: ignore

        ollama_path = _get_ollama_models_dir()
        ollama_models_dir = str(ollama_path) if ollama_path else ""
    except Exception:
        ollama_models_dir = ""

    # Electron paths (best-effort strings)
    if sys.platform == "win32":
        electron_user_data = "%APPDATA%/nodetool-electron"
        electron_log_file = "%APPDATA%/nodetool-electron/nodetool.log"
        electron_logs_dir = "%APPDATA%/nodetool-electron/logs"
        electron_main_log_file = "%APPDATA%/nodetool-electron/logs/main.log"
    elif sys.platform == "darwin":
        electron_user_data = "~/Library/Application Support/nodetool-electron"
        electron_log_file = (
            "~/Library/Application Support/nodetool-electron/nodetool.log"
        )
        electron_logs_dir = "~/Library/Logs/nodetool-electron"
        electron_main_log_file = "~/Library/Logs/nodetool-electron/main.log"
    else:
        # Linux and others
        electron_user_data = "~/.config/nodetool-electron"
        electron_log_file = "~/.config/nodetool-electron/nodetool.log"
        electron_logs_dir = "~/.config/nodetool-electron/logs"
        electron_main_log_file = "~/.config/nodetool-electron/logs/main.log"

    return {
        "settings_path": settings_path,
        "secrets_path": secrets_path,
        "data_dir": data_dir,
        "core_logs_dir": core_logs_dir,
        "core_log_file": core_log_file,
        "ollama_models_dir": ollama_models_dir,
        "huggingface_cache_dir": huggingface_cache_dir,
        "electron_user_data": electron_user_data,
        "electron_log_file": electron_log_file,
        "electron_logs_dir": electron_logs_dir,
        "electron_main_log_file": electron_main_log_file,
    }


