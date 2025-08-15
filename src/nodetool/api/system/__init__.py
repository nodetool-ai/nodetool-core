from typing import List, Literal, Optional
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


class PathsInfo(BaseModel):
    settings_path: str
    secrets_path: str
    data_dir: str
    core_logs_dir: str
    core_log_file: str
    electron_user_data: str
    electron_log_file: str
    electron_logs_dir: str


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


@router.get("/")
async def get_system_info() -> SystemInfoResponse:
    os_info = get_os_info()
    versions = get_versions_info()
    paths = get_paths_info()

    return SystemInfoResponse(
        os=OSInfo(**os_info),
        versions=VersionsInfo(**versions),
        paths=PathsInfo(**paths),
    )


@router.get("/health")
async def get_system_health() -> HealthResponse:
    result = run_health_checks()
    checks_models = [HealthCheck(**c) for c in result["checks"]]
    summary = HealthSummary(**result["summary"])
    return HealthResponse(checks=checks_models, summary=summary)


@router.get("/stats")
async def get_stats() -> SystemStats:
    return get_system_stats()



