from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

from nodetool.common.environment import Environment
from nodetool.common.settings import (
    get_log_path,
    get_system_data_path,
    get_system_file_path,
    SETTINGS_FILE,
    SECRETS_FILE,
)


def _exists(path: Path) -> bool:
    try:
        return path.exists()
    except Exception:
        return False


def _is_writable(path: Path) -> bool:
    try:
        if path.is_dir():
            test_file = path / ".permission_check"
            with open(test_file, "w") as f:
                f.write("ok")
            test_file.unlink(missing_ok=True)
            return True
        else:
            parent = path if path.suffix == "" else path.parent
            parent.mkdir(parents=True, exist_ok=True)
            test_file = parent / ".permission_check"
            with open(test_file, "w") as f:
                f.write("ok")
            test_file.unlink(missing_ok=True)
            return True
    except Exception:
        return False


def _check(id: str, ok: bool, details: str = "", hint: str = "") -> Dict:
    return {
        "id": id,
        "status": "ok" if ok else "error",
        "details": details or None,
        "fix_hint": hint or None,
    }


def run_health_checks() -> Dict[str, object]:
    checks: List[Dict] = []

    # Paths
    settings_path = get_system_file_path(SETTINGS_FILE)
    secrets_path = get_system_file_path(SECRETS_FILE)
    logs_dir = get_system_data_path("logs")
    core_log_file = get_log_path("nodetool.log")
    data_dir = get_system_data_path("")

    # Existence
    checks.append(
        _check(
            "settings_file_exists",
            _exists(settings_path),
            str(settings_path),
            "Create settings via UI or place file at this path.",
        )
    )
    checks.append(
        _check(
            "secrets_file_exists",
            _exists(secrets_path),
            str(secrets_path),
            "Add API keys as needed; this file is optional.",
        )
    )

    # Writability
    checks.append(
        _check(
            "config_dir_writable",
            _is_writable(settings_path.parent) and _is_writable(secrets_path.parent),
            f"settings_dir={settings_path.parent}; secrets_dir={secrets_path.parent}",
            "Fix directory permissions.",
        )
    )
    checks.append(
        _check(
            "logs_dir_writable",
            _is_writable(Path(logs_dir)),
            str(logs_dir),
            "Fix directory permissions.",
        )
    )

    # Database
    db_path = Environment.get_db_path()
    if db_path and db_path != ":memory:":
        db_parent = Path(db_path).parent
        checks.append(
            _check(
                "db_ready",
                _is_writable(db_parent),
                f"db_parent={db_parent}",
                "Ensure DB directory exists and is writable.",
            )
        )
    else:
        checks.append(
            _check(
                "db_ready",
                True,
                "memory or external DB",
                "",
            )
        )

    # Temp storage
    tmp_dir = get_system_file_path("tmp").parent
    checks.append(
        _check(
            "temp_storage_writable",
            _is_writable(tmp_dir),
            str(tmp_dir),
            "Ensure temp directory is writable.",
        )
    )

    # Assets folder
    asset_folder = Path(Environment.get_asset_folder())
    checks.append(
        _check(
            "asset_folder_writable",
            _is_writable(asset_folder),
            str(asset_folder),
            "Ensure asset folder exists and is writable.",
        )
    )

    # Chroma
    chroma_url = Environment.get("CHROMA_URL", None)
    chroma_path = Environment.get("CHROMA_PATH", None)
    if chroma_url:
        # Keep fast: don't attempt network call now; mark as warn with info
        checks.append(
            {
                "id": "chroma_configured",
                "status": "warn",
                "details": f"CHROMA_URL set: {chroma_url}",
                "fix_hint": "Optionally add a connectivity check later.",
            }
        )
    elif chroma_path:
        checks.append(
            _check(
                "chroma_path_writable",
                _is_writable(Path(str(chroma_path))),
                str(chroma_path),
                "Ensure Chroma path exists and is writable.",
            )
        )

    # Comfy
    comfy_folder = Environment.get("COMFY_FOLDER", None)
    if comfy_folder:
        p = Path(str(comfy_folder))
        checks.append(
            _check(
                "comfy_folder_present",
                _exists(p),
                str(p),
                "Update COMFY_FOLDER to a valid path.",
            )
        )

    # GPU
    device = Environment.get_torch_device()
    device_status = "ok" if device else "warn"
    checks.append(
        {
            "id": "gpu_available",
            "status": device_status,
            "details": str(device) if device else "No GPU detected (cpu)",
            "fix_hint": None,
        }
    )

    # Providers (presence only)
    provider_keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "HF_TOKEN",
        "REPLICATE_API_TOKEN",
    ]
    present = [k for k in provider_keys if Environment.get(k, None)]
    checks.append(
        {
            "id": "providers_configured",
            "status": "ok" if present else "warn",
            "details": f"Present: {', '.join(present) if present else 'none'}",
            "fix_hint": "Add provider API keys in settings if needed.",
        }
    )

    summary = {
        "ok": sum(1 for c in checks if c["status"] == "ok"),
        "warn": sum(1 for c in checks if c["status"] == "warn"),
        "error": sum(1 for c in checks if c["status"] == "error"),
    }

    return {"checks": checks, "summary": summary}


