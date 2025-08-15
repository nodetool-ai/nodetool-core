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
    settings_ok = _exists(settings_path)
    checks.append(
        _check(
            "settings_file_exists",
            settings_ok,
            str(settings_path),
            "" if settings_ok else "Create settings.yaml here to configure NodeTool.",
        )
    )
    secrets_ok = _exists(secrets_path)
    checks.append(
        _check(
            "secrets_file_exists",
            secrets_ok,
            str(secrets_path),
            "" if secrets_ok else "Optional: create secrets.yaml here to add API keys (e.g. OPENAI_API_KEY).",
        )
    )

    # Writability
    cfg_write_ok = _is_writable(settings_path.parent) and _is_writable(secrets_path.parent)
    checks.append(
        _check(
            "config_dir_writable",
            cfg_write_ok,
            f"settings_dir={settings_path.parent}; secrets_dir={secrets_path.parent}",
            "" if cfg_write_ok else "Make these folders writable so files can be saved.",
        )
    )
    logs_write_ok = _is_writable(Path(logs_dir))
    checks.append(
        _check(
            "logs_dir_writable",
            logs_write_ok,
            str(logs_dir),
            "" if logs_write_ok else "Make this folder writable so logs can be saved.",
        )
    )

    # Database
    db_path = Environment.get_db_path()
    if db_path and db_path != ":memory:":
        db_parent = Path(db_path).parent
        db_ok = _is_writable(db_parent)
        checks.append(
            _check(
                "db_ready",
                db_ok,
                f"db_parent={db_parent}",
                "" if db_ok else "Create the folder and make it writable to create the database file.",
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
    tmp_ok = _is_writable(tmp_dir)
    checks.append(
        _check(
            "temp_storage_writable",
            tmp_ok,
            str(tmp_dir),
            "" if tmp_ok else "Make this temp folder writable.",
        )
    )

    # Assets folder
    asset_folder = Path(Environment.get_asset_folder())
    assets_ok = _is_writable(asset_folder)
    checks.append(
        _check(
            "asset_folder_writable",
            assets_ok,
            str(asset_folder),
            "" if assets_ok else "Create the folder and make it writable so assets can be saved.",
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
                "fix_hint": "If you plan to use Chroma, ensure the server is reachable.",
            }
        )
    elif chroma_path:
        chroma_path_ok = _is_writable(Path(str(chroma_path)))
        checks.append(
            _check(
                "chroma_path_writable",
                chroma_path_ok,
                str(chroma_path),
                "" if chroma_path_ok else "Create the folder and make it writable; Chroma stores data here.",
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
            "details": str(device) if device else "CPU only",
            "fix_hint": "" if device else "Install compatible GPU drivers/CUDA (or continue on CPU).",
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
            "fix_hint": "" if present else "Add provider API keys in Settings if you plan to use them.",
        }
    )

    summary = {
        "ok": sum(1 for c in checks if c["status"] == "ok"),
        "warn": sum(1 for c in checks if c["status"] == "warn"),
        "error": sum(1 for c in checks if c["status"] == "error"),
    }

    return {"checks": checks, "summary": summary}


