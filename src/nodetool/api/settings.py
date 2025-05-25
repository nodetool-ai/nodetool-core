from fastapi import APIRouter, HTTPException
from nodetool.common.environment import Environment
from nodetool.common.settings import load_settings, save_settings
from nodetool.common.configuration import Setting, get_settings_registry
from pydantic import BaseModel
from typing import Any, Dict, List


router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingsResponse(BaseModel):
    settings: List[Setting]


class SettingsUpdateRequest(BaseModel):
    settings: Dict[str, Any]
    secrets: Dict[str, Any]


@router.get("/")
async def get_settings() -> SettingsResponse:
    if Environment.is_production():
        raise HTTPException(
            status_code=403, detail="Settings cannot be read in production"
        )

    settings = get_settings_registry()

    return SettingsResponse(settings=settings)


@router.put("/")
async def update_settings(
    req: SettingsUpdateRequest,
) -> Dict[str, str]:
    if Environment.is_production():
        raise HTTPException(
            status_code=403, detail="Settings cannot be updated in production"
        )

    settings, secrets = load_settings()

    settings.update(req.settings)
    secrets.update(req.secrets)

    save_settings(settings, secrets)

    Environment.load_settings()

    return {"message": "Settings updated successfully"}
