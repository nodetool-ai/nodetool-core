from __future__ import annotations

import platform
import sys
from importlib import metadata
from typing import Dict

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
    return {
        "python": platform.python_version(),
        "nodetool_core": _safe_version("nodetool-core"),
        "nodetool_base": _safe_version("nodetool-base"),
    }


def get_paths_info() -> Dict[str, str]:
    settings_path = str(get_system_file_path(SETTINGS_FILE))
    secrets_path = str(get_system_file_path(SECRETS_FILE))
    data_dir = str(get_system_data_path(""))
    core_logs_dir = str(get_system_data_path("logs"))
    core_log_file = str(get_log_path("nodetool.log"))

    # Electron paths (best-effort strings)
    if sys.platform == "win32":
        electron_user_data = "%APPDATA%/nodetool-electron"
        electron_log_file = "%APPDATA%/nodetool-electron/nodetool.log"
        electron_logs_dir = "%APPDATA%/nodetool-electron/logs"
    elif sys.platform == "darwin":
        electron_user_data = "~/Library/Application Support/nodetool-electron"
        electron_log_file = (
            "~/Library/Application Support/nodetool-electron/nodetool.log"
        )
        electron_logs_dir = "~/Library/Logs/nodetool-electron"
    else:
        # Linux and others
        electron_user_data = "~/.config/nodetool-electron"
        electron_log_file = "~/.config/nodetool-electron/nodetool.log"
        electron_logs_dir = "~/.config/nodetool-electron/logs"

    return {
        "settings_path": settings_path,
        "secrets_path": secrets_path,
        "data_dir": data_dir,
        "core_logs_dir": core_logs_dir,
        "core_log_file": core_log_file,
        "electron_user_data": electron_user_data,
        "electron_log_file": electron_log_file,
        "electron_logs_dir": electron_logs_dir,
    }


