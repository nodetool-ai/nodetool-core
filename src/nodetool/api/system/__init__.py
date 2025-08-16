from typing import List, Literal, Optional, Any, Dict
from fastapi import APIRouter
from pydantic import BaseModel

from nodetool.common.system_stats import get_system_stats, SystemStats

from .info import get_os_info, get_versions_info, get_paths_info
from .health import run_health_checks


router = APIRouter(prefix="/api/system", tags=["system"])


class OSInfo(BaseModel):
    platform: str
    release: str
    arch: str


class VersionsInfo(BaseModel):
    python: Optional[str] = None
    nodetool_core: Optional[str] = None
    nodetool_base: Optional[str] = None
    cuda: Optional[str] = None


class PathsInfo(BaseModel):
    settings_path: str
    secrets_path: str
    data_dir: str
    core_logs_dir: str
    core_log_file: str
    ollama_models_dir: str
    huggingface_cache_dir: str
    electron_user_data: str
    electron_log_file: str
    electron_logs_dir: str
    electron_main_log_file: str


class SystemInfoResponse(BaseModel):
    os: OSInfo
    versions: VersionsInfo
    paths: PathsInfo


class HealthCheck(BaseModel):
    id: str
    status: Literal["ok", "warn", "error"]
    details: Optional[str] = None
    fix_hint: Optional[str] = None


class HealthSummary(BaseModel):
    ok: int
    warn: int
    error: int


class HealthResponse(BaseModel):
    checks: List[HealthCheck]
    summary: HealthSummary


_CACHE: dict[str, tuple[float, dict]] = {}
_TTL_SECONDS = 3.0


@router.get("/")
async def get_system_info() -> SystemInfoResponse:
    import time

    now = time.time()
    cached = _CACHE.get("system_info")
    if cached and (now - cached[0]) < _TTL_SECONDS:
        payload = cached[1]
    else:
        os_info = get_os_info()
        versions = get_versions_info()
        paths = get_paths_info()
        payload = {
            "os": os_info,
            "versions": versions,
            "paths": paths,
        }
        _CACHE["system_info"] = (now, payload)

    return SystemInfoResponse(
        os=OSInfo(**payload["os"]),
        versions=VersionsInfo(**payload["versions"]),
        paths=PathsInfo(**payload["paths"]),
    )


@router.get("/health")
async def get_system_health() -> HealthResponse:
    try:
        result: Dict[str, Any] = run_health_checks()
        
        # Validate the structure of the result
        if not isinstance(result, dict):
            raise ValueError("Health check result must be a dictionary")
        
        checks_list: List[Dict[str, Any]] = result.get("checks", []) or []
        if not isinstance(checks_list, list):
            raise ValueError("Health check 'checks' must be a list")
        
        # Validate each check has required fields
        validated_checks = []
        for check in checks_list:
            if not isinstance(check, dict):
                continue  # Skip invalid checks
            if "id" not in check or "status" not in check:
                continue  # Skip checks missing required fields
            validated_checks.append(check)
        
        checks_models = [HealthCheck(**c) for c in validated_checks]
        
        summary_data = result.get("summary", {}) or {}
        if not isinstance(summary_data, dict):
            summary_data = {"ok": 0, "warn": 0, "error": 0}
        
        # Ensure summary has required fields with defaults
        summary_data.setdefault("ok", 0)
        summary_data.setdefault("warn", 0) 
        summary_data.setdefault("error", 0)
        
        summary = HealthSummary(**summary_data)
        return HealthResponse(checks=checks_models, summary=summary)
        
    except Exception as e:
        # Return a safe fallback response if health checks fail
        return HealthResponse(
            checks=[HealthCheck(
                id="health_check_error",
                status="error",
                details=f"Health check system error: {str(e)}",
                fix_hint="Check system logs for more details"
            )],
            summary=HealthSummary(ok=0, warn=0, error=1)
        )


@router.get("/stats")
async def get_stats() -> SystemStats:
    return get_system_stats()



