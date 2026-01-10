import asyncio
import json
import os
import platform
import re
import shutil
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from nodetool.config.settings import get_system_data_path, load_settings
from nodetool.models.workflow import Workflow as WorkflowModel
from nodetool.system import system_stats

# Secret redaction patterns
#
# Note: Regex-based secret detection has inherent limitations:
# - May miss secrets with unusual formats (false negatives)
# - May incorrectly flag non-sensitive data (false positives)
# - New API key formats may not be covered
#
# For comprehensive secret management, consider using dedicated secret
# scanning tools in CI/CD pipelines. This implementation provides
# best-effort redaction for debug bundles intended for support purposes.

# Patterns that indicate a value might be a secret
SECRET_KEY_PATTERNS = re.compile(
    r"(api[_-]?key|[_-]token$|^token[_-]|secret|password|credential|bearer|access[_-]?key|private[_-]?key)",
    re.IGNORECASE,
)
# Keys that should NEVER be redacted (even if they match other patterns)
SAFE_KEY_PATTERNS = re.compile(
    r"^(id|_id|node_id|workflow_id|user_id|thread_id|message_id|job_id|parent_id|"
    r"source_id|target_id|ref|uuid|name|type|updated_at|created_at)$",
    re.IGNORECASE,
)
# Patterns that look like actual secret values (API keys, tokens, etc.)
SECRET_VALUE_PATTERNS = re.compile(
    r"^(sk-[a-zA-Z0-9]{20,}|"  # OpenAI keys
    r"sk-ant-[a-zA-Z0-9-]{20,}|"  # Anthropic keys
    r"hf_[a-zA-Z0-9]{20,}|"  # HuggingFace tokens
    r"r8_[a-zA-Z0-9]{20,}|"  # Replicate tokens
    r"fal_[a-zA-Z0-9]{20,}|"  # FAL keys
    r"eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)$"  # JWT tokens
)
REDACTED = "[REDACTED]"


def _redact_secrets(data: Any, parent_key: str = "") -> Any:
    """
    Recursively redact potential secrets from data structures.

    Redacts values when:
    - The key name suggests it's a secret (api_key, password, etc.)
    - The value looks like a known secret pattern (OpenAI key, JWT, etc.)

    Does NOT redact:
    - Keys that are known safe (id, name, type, timestamps, etc.)
    """
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            str_key = str(key).lower()
            # Skip redaction for known safe keys
            if SAFE_KEY_PATTERNS.match(str_key):
                result[key] = _redact_secrets(value, str_key)
            # Check if key name suggests a secret
            elif SECRET_KEY_PATTERNS.search(str_key):
                if value and isinstance(value, str) and len(value) > 0:
                    result[key] = REDACTED
                else:
                    result[key] = value
            else:
                result[key] = _redact_secrets(value, str_key)
        return result
    elif isinstance(data, list):
        return [_redact_secrets(item, parent_key) for item in data]
    elif isinstance(data, str):
        # Check if string value looks like a secret (but only for specific patterns)
        if len(data) >= 20 and SECRET_VALUE_PATTERNS.match(data):
            return REDACTED
        return data
    else:
        return data


def _redact_log_secrets(log_content: str) -> str:
    """
    Redact potential secrets from log file content.
    """
    # Patterns to redact in logs
    patterns = [
        (
            r'(api[_-]?key|token|secret|password|bearer|authorization)["\s:=]+["\']?([a-zA-Z0-9_-]{20,})["\']?',
            r"\1: " + REDACTED,
        ),
        (r"(sk-[a-zA-Z0-9]{20,})", REDACTED),  # OpenAI
        (r"(sk-ant-[a-zA-Z0-9-]{20,})", REDACTED),  # Anthropic
        (r"(hf_[a-zA-Z0-9]{20,})", REDACTED),  # HuggingFace
        (r"(r8_[a-zA-Z0-9]{20,})", REDACTED),  # Replicate
        (r"(fal_[a-zA-Z0-9]{20,})", REDACTED),  # FAL
        (r"(eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)", REDACTED),  # JWT
    ]

    result = log_content
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


router = APIRouter(prefix="/api/debug", tags=["debug"])


class DebugBundleRequest(BaseModel):
    workflow_id: str | None = Field(default=None)
    graph: dict[str, Any] | None = Field(default=None)
    errors: list[str] | None = Field(default=None)
    preferred_save: str | None = Field(default=None, description="desktop or downloads preference")


class DebugBundleResponse(BaseModel):
    file_path: str
    filename: str
    message: str


def _get_default_save_dir(preferred: str | None) -> Path:
    home = Path.home()
    candidates: list[Path] = []
    if preferred == "desktop":
        candidates = [home / "Desktop", home / "Downloads", home]
    elif preferred == "downloads":
        candidates = [home / "Downloads", home / "Desktop", home]
    else:
        # Default to Downloads folder (more appropriate for debug bundles)
        candidates = [home / "Downloads", home / "Desktop", home]
    for c in candidates:
        try:
            if c.exists() and c.is_dir():
                return c
        except Exception:
            continue
    return home


def _get_nodetool_version() -> str:
    try:
        from importlib.metadata import version

        # Try common distributions
        for dist_name in [
            "nodetool",
            "nodetool-core",
            "nodetool_core",
        ]:
            try:
                return version(dist_name)
            except Exception:
                continue
    except Exception:
        pass
    # Fallback to dev timestamp
    return f"dev-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"


def _get_gpu_name() -> str | None:
    nvml = system_stats.nvml
    if nvml is None:
        return None

    did_init = False
    try:
        nvml.nvmlInit()
        did_init = True
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        name = nvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="ignore")
        return str(name)
    except Exception:
        return None
    finally:
        if did_init:
            with suppress(Exception):
                nvml.nvmlShutdown()


def _collect_env_info() -> dict[str, Any]:
    stats = system_stats.get_system_stats()
    import shutil as _shutil

    disk = _shutil.disk_usage(str(Path.home()))
    gpu_name = _get_gpu_name()
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "memory_total_gb": stats.memory_total_gb,
        "memory_used_gb": stats.memory_used_gb,
        "memory_percent": stats.memory_percent,
        "vram_total_gb": stats.vram_total_gb,
        "vram_used_gb": getattr(stats, "vram_used_gb", None),
        "vram_percent": stats.vram_percent,
        "gpu_model": gpu_name,
        "disk_total_gb": round(disk.total / (1024**3), 2),
        "disk_free_gb": round(disk.free / (1024**3), 2),
        "nodetool_version": _get_nodetool_version(),
    }


def _collect_config_info() -> dict[str, Any]:
    from nodetool.config.environment import Environment
    from nodetool.security.secret_helper import get_secret_sync

    # Infer run mode (best-effort)
    run_mode = "cloud" if Environment.is_production() else "local"

    settings = load_settings()

    def has(key: str) -> bool:
        # Check settings, then use get_secret_sync which checks env vars and database
        v = settings.get(key) or get_secret_sync(key)
        return bool(v)

    return {
        "run_mode": run_mode,
        "is_production": Environment.is_production(),
        "storage": "s3" if Environment.get_s3_access_key_id() is not None else "file",
        "providers": {
            "openai": has("OPENAI_API_KEY"),
            "anthropic": has("ANTHROPIC_API_KEY"),
            "gemini": has("GEMINI_API_KEY"),
            "huggingface": has("HF_TOKEN"),
            "replicate": has("REPLICATE_API_TOKEN"),
            "fal": has("FAL_API_KEY"),
            "elevenlabs": has("ELEVENLABS_API_KEY"),
        },
    }


def _write_readme(target_root: Path) -> None:
    text = (
        "NodeTool Debug Bundle\n\n"
        "Attach this ZIP when reporting a bug.\n\n"
        "Contents:\n"
        "- logs/nodetool.log\n"
        "- workflow/last-template.json (redacted)\n"
        "- env/system.json\n"
        "- env/config.json\n"
    )
    (target_root / "README.txt").write_text(text, encoding="utf-8")


def _create_zip(src_dir: Path, zip_dest: Path) -> None:
    # Ensure parent exists
    zip_dest.parent.mkdir(parents=True, exist_ok=True)
    # Use shutil make_archive for robustness
    zip_dest.with_suffix("")
    # shutil.make_archive adds extension itself; to control exact name we write to temp then rename
    with tempfile.TemporaryDirectory(prefix="nodetool-debug-") as tmp_dir:
        tmp_base = os.path.join(tmp_dir, "archive")
        archive_path = shutil.make_archive(tmp_base, "zip", root_dir=str(src_dir))
        shutil.move(archive_path, str(zip_dest))


@router.post("/export", response_model=DebugBundleResponse)
async def export_debug_bundle(payload: DebugBundleRequest) -> DebugBundleResponse:
    # Prepare temp staging dir
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    staging_dir = Path(tempfile.mkdtemp(prefix=f"nodetool-debug-{ts}-"))
    staging_logs_dir = staging_dir / "logs"
    workflow_dir = staging_dir / "workflow"
    env_dir = staging_dir / "env"

    # Ensure staging directories exist
    staging_logs_dir.mkdir(parents=True, exist_ok=True)
    workflow_dir.mkdir(parents=True, exist_ok=True)
    env_dir.mkdir(parents=True, exist_ok=True)

    # Collect logs from system data path
    system_logs_dir = get_system_data_path("logs")
    log_file_path = system_logs_dir / "nodetool.log"

    if log_file_path.exists():
        try:
            log_content = log_file_path.read_text(encoding="utf-8")
            if log_content.strip():
                # Redact secrets from log content before saving
                redacted_log = _redact_log_secrets(log_content)
                (staging_logs_dir / "nodetool.log").write_text(redacted_log, encoding="utf-8")
            else:
                (staging_logs_dir / "nodetool.log").write_text("Log file exists but is empty.", encoding="utf-8")
        except Exception as e:
            (staging_logs_dir / "nodetool.log").write_text(f"Could not read log file: {e}", encoding="utf-8")
    else:
        (staging_logs_dir / "nodetool.log").write_text(f"Log file not found at {log_file_path}", encoding="utf-8")

    # Workflow info -> workflow/last-template.json
    workflow_payload: dict[str, Any] = {}
    if payload.graph is not None:
        workflow_payload["graph"] = payload.graph
    if payload.workflow_id:
        wf = await WorkflowModel.get(payload.workflow_id)
        if wf:
            workflow_payload.update(
                {
                    "id": wf.id,
                    "name": wf.name,
                    "updated_at": (wf.updated_at.isoformat() if hasattr(wf, "updated_at") else None),
                    "settings": wf.settings,
                    "graph": workflow_payload.get("graph", wf.graph),
                }
            )
    if payload.errors:
        workflow_payload["errors"] = payload.errors
    if not workflow_payload:
        workflow_payload = {"note": "No workflow context provided"}
    # Redact any secrets from workflow data before saving
    redacted_workflow = _redact_secrets(workflow_payload)
    (workflow_dir / "last-template.json").write_text(json.dumps(redacted_workflow, indent=2), encoding="utf-8")

    # Env info -> env/system.json and env/config.json
    system_info = _collect_env_info()
    config_info = _collect_config_info()
    (env_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")
    (env_dir / "config.json").write_text(json.dumps(config_info, indent=2), encoding="utf-8")

    # README
    _write_readme(staging_dir)

    # Create ZIP at Desktop/Downloads
    save_dir = _get_default_save_dir(payload.preferred_save)
    filename = f"nodetool-debug-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zip"
    zip_path = save_dir / filename
    await asyncio.to_thread(_create_zip, staging_dir, zip_path)
    shutil.rmtree(staging_dir, ignore_errors=True)

    return DebugBundleResponse(
        file_path=str(zip_path),
        filename=filename,
        message=f"Debug bundle saved to {zip_path}",
    )
