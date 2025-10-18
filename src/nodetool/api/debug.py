import asyncio
import json
import os
import platform
import re
import shutil
import tempfile
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from nodetool.config.settings import get_system_data_path, load_settings
from nodetool.models.workflow import Workflow as WorkflowModel
from nodetool.system import system_stats


router = APIRouter(prefix="/api/debug", tags=["debug"])


SENSITIVE_KEY_NAMES = {
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "bearer",
    "token",
    "secret",
    "password",
    "openai_api_key",
    "anthropic_api_key",
    "gemini_api_key",
    "hf_token",
    "replicate_api_token",
}

SENSITIVE_CONTENT_KEYS = {
    "prompt",
    "system_prompt",
    "messages",
    "instruction",
    "input",
}

SENSITIVE_REGEXES = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),  # OpenAI-style keys
    re.compile(r"hf_[A-Za-z0-9]{20,}"),  # HuggingFace tokens
    re.compile(r"eyJhbGciOi[A-Za-z0-9_\-\.]+"),  # JWT-like
]


class DebugBundleRequest(BaseModel):
    workflow_id: Optional[str] = Field(default=None)
    graph: Optional[Dict[str, Any]] = Field(default=None)
    errors: Optional[List[str]] = Field(default=None)
    preferred_save: Optional[str] = Field(
        default=None, description="desktop or downloads preference"
    )


class DebugBundleResponse(BaseModel):
    file_path: str
    filename: str
    message: str


def _get_default_save_dir(preferred: Optional[str]) -> Path:
    home = Path.home()
    candidates: List[Path] = []
    if preferred == "desktop":
        candidates = [home / "Desktop", home / "Downloads", home]
    elif preferred == "downloads":
        candidates = [home / "Downloads", home / "Desktop", home]
    else:
        candidates = [home / "Desktop", home / "Downloads", home]
    for c in candidates:
        try:
            if c.exists() and c.is_dir():
                return c
        except Exception:
            continue
    return home


def _read_last_lines(path: Path, max_lines: int) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            dq: deque[str] = deque(maxlen=max_lines)
            for line in f:
                dq.append(line.rstrip("\n"))
        return "\n".join(dq)
    except Exception:
        return ""


def _collect_candidate_log_files() -> List[Path]:
    logs_root = get_system_data_path("") / "logs"
    if not logs_root.exists():
        return []
    files: List[Path] = []
    try:
        for p in logs_root.iterdir():
            if p.is_file() and (p.suffix in {".log", ".jsonl", ".txt"}):
                files.append(p)
    except Exception:
        pass
    # Sort by modified time desc
    files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return files


def _regex_redact_text(text: str, secret_values: List[str]) -> str:
    redacted = text
    # direct secrets
    for secret in secret_values:
        if not secret or not isinstance(secret, str):
            continue
        # Replace long secrets only to avoid over-redaction
        if len(secret) >= 6:
            redacted = redacted.replace(secret, "***")
    # regex patterns
    for pat in SENSITIVE_REGEXES:
        redacted = pat.sub("***", redacted)
    # redact common header/value pairs
    redacted = re.sub(
        r"(Authorization\s*:\s*)(Bearer\s+\S+)", r"\1***", redacted, flags=re.IGNORECASE
    )
    return redacted


def _redact_json(obj: Any, secret_values: List[str]) -> Any:
    try:
        if isinstance(obj, dict):
            redacted: Dict[str, Any] = {}
            for k, v in obj.items():
                lk = str(k).lower()
                if lk in SENSITIVE_KEY_NAMES or lk in SENSITIVE_CONTENT_KEYS:
                    redacted[k] = "***"
                elif lk == "headers" and isinstance(v, dict):
                    headers = dict(v)
                    for hk in list(headers.keys()):
                        if hk.lower() == "authorization":
                            headers[hk] = "***"
                    redacted[k] = headers
                else:
                    redacted[k] = _redact_json(v, secret_values)
            return redacted
        elif isinstance(obj, list):
            return [_redact_json(v, secret_values) for v in obj]
        elif isinstance(obj, str):
            return _regex_redact_text(obj, secret_values)
        else:
            return obj
    except Exception:
        return "***"


def _redact_file_to(target_path: Path, content: str, secret_values: List[str]) -> None:
    redacted = _regex_redact_text(content, secret_values)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(redacted, encoding="utf-8")


def _write_json_file(
    target_path: Path, data: Dict[str, Any], secret_values: List[str]
) -> None:
    redacted_data = _redact_json(data, secret_values)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(redacted_data, indent=2), encoding="utf-8")


def _get_secrets_values() -> List[str]:
    try:
        settings, secrets = load_settings()
        secret_values: List[str] = []
        for v in secrets.values():
            if isinstance(v, str) and v:
                secret_values.append(v)
        # Also include env vars for common keys if present
        for key in list(SENSITIVE_KEY_NAMES):
            env_val = os.environ.get(key.upper())
            if env_val:
                secret_values.append(env_val)
        return secret_values
    except Exception:
        return []


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
    return f"dev-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"


def _get_gpu_name() -> Optional[str]:
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
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass


def _collect_env_info() -> Dict[str, Any]:
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


def _collect_config_info() -> Dict[str, Any]:
    from nodetool.config.environment import Environment

    # Infer run mode (best-effort)
    run_mode = "cloud" if Environment.is_production() else "local"

    settings, secrets = load_settings()

    def has(key: str) -> bool:
        v = secrets.get(key) or settings.get(key) or os.environ.get(key)
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


def _safe_json_dump(data: Dict[str, Any]) -> str:
    try:
        return json.dumps(data)
    except Exception:
        return "{}"


def _write_readme(target_root: Path) -> None:
    text = (
        "NodeTool Debug Bundle\n\n"
        "Attach this ZIP when reporting a bug.\n\n"
        "Contents:\n"
        "- logs/app.log (last lines)\n"
        "- logs/run.log (latest run logs, redacted)\n"
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
    tmp_base = tempfile.mktemp(prefix="nodetool-debug-")
    archive_path = shutil.make_archive(tmp_base, "zip", root_dir=str(src_dir))
    shutil.move(archive_path, str(zip_dest))


@router.post("/export", response_model=DebugBundleResponse)
async def export_debug_bundle(payload: DebugBundleRequest) -> DebugBundleResponse:
    secret_values = _get_secrets_values()

    # Prepare temp staging dir
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    staging_dir = Path(tempfile.mkdtemp(prefix=f"nodetool-debug-{ts}-"))
    logs_dir = staging_dir / "logs"
    workflow_dir = staging_dir / "workflow"
    env_dir = staging_dir / "env"

    # Collect logs -> logs/app.log and logs/run.log
    candidate_logs = _collect_candidate_log_files()
    app_log_texts: List[str] = []
    run_log_text = ""
    for i, p in enumerate(candidate_logs[:10]):
        last_lines = _read_last_lines(p, 500)
        if not last_lines:
            continue
        # Keep the most recent .jsonl as run.log source
        if i == 0:
            run_log_text = last_lines
        header = f"===== {p.name} =====\n"
        app_log_texts.append(header + last_lines)

    if app_log_texts:
        _redact_file_to(logs_dir / "app.log", "\n\n".join(app_log_texts), secret_values)
    else:
        _redact_file_to(logs_dir / "app.log", "(no app logs found)", secret_values)

    if run_log_text:
        # Attempt to parse JSONL and redact semantically; fallback to regex
        redacted_lines: List[str] = []
        for line in run_log_text.splitlines():
            try:
                obj = json.loads(line)
                obj = _redact_json(obj, secret_values)
                redacted_lines.append(json.dumps(obj))
            except Exception:
                redacted_lines.append(_regex_redact_text(line, secret_values))
        (logs_dir / "run.log").parent.mkdir(parents=True, exist_ok=True)
        (logs_dir / "run.log").write_text("\n".join(redacted_lines), encoding="utf-8")
    else:
        _redact_file_to(
            logs_dir / "run.log", "(no recent run logs found)", secret_values
        )

    # Workflow info -> workflow/last-template.json
    workflow_payload: Dict[str, Any] = {}
    if payload.graph is not None:
        workflow_payload["graph"] = payload.graph
    if payload.workflow_id:
        wf = await WorkflowModel.get(payload.workflow_id)
        if wf:
            workflow_payload.update(
                {
                    "id": wf.id,
                    "name": wf.name,
                    "updated_at": (
                        wf.updated_at.isoformat() if hasattr(wf, "updated_at") else None
                    ),
                    "settings": wf.settings,
                    "graph": (
                        wf.graph
                        if "graph" not in workflow_payload
                        else workflow_payload["graph"]
                    ),
                }
            )
    if payload.errors:
        workflow_payload["errors"] = payload.errors
    if not workflow_payload:
        workflow_payload = {"note": "No workflow context provided"}
    _write_json_file(
        workflow_dir / "last-template.json", workflow_payload, secret_values
    )

    # Env info -> env/system.json and env/config.json
    system_info = _collect_env_info()
    config_info = _collect_config_info()
    _write_json_file(env_dir / "system.json", system_info, secret_values)
    _write_json_file(env_dir / "config.json", config_info, secret_values)

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
