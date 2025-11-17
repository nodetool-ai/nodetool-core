import asyncio
import json
import os
import platform
import shutil
import tempfile
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from nodetool.config.settings import get_system_data_path, load_settings
from nodetool.models.workflow import Workflow as WorkflowModel
from nodetool.system import system_stats

router = APIRouter(prefix="/api/debug", tags=["debug"])


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
            with suppress(Exception):
                nvml.nvmlShutdown()


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
    tmp_base = tempfile.mktemp(prefix="nodetool-debug-")
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
        app_log_texts = log_file_path.read_text(encoding="utf-8").split("\n")
        if not app_log_texts:
            raise ValueError("No app logs found")
        # Copy log file to staging directory
        (staging_logs_dir / "nodetool.log").write_text(
            "\n".join(app_log_texts), encoding="utf-8"
        )
    else:
        raise ValueError(f"Log file not found at {log_file_path}")

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
                      "graph": workflow_payload.get("graph", wf.graph),
                }
            )
    if payload.errors:
        workflow_payload["errors"] = payload.errors
    if not workflow_payload:
        workflow_payload = {"note": "No workflow context provided"}
    (workflow_dir / "last-template.json").write_text(
        json.dumps(workflow_payload, indent=2), encoding="utf-8"
    )

    # Env info -> env/system.json and env/config.json
    system_info = _collect_env_info()
    config_info = _collect_config_info()
    (env_dir / "system.json").write_text(
        json.dumps(system_info, indent=2), encoding="utf-8"
    )
    (env_dir / "config.json").write_text(
        json.dumps(config_info, indent=2), encoding="utf-8"
    )

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
