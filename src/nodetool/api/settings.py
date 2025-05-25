from fastapi import APIRouter, HTTPException
from nodetool.common.environment import Environment
from nodetool.common.settings import load_settings, save_settings
from nodetool.common.configuration import Setting, get_settings_registry
from pydantic import BaseModel
from typing import Any, Dict, List, Optional


router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingWithValue(BaseModel):
    package_name: str
    env_var: str
    group: str
    description: str
    is_secret: bool
    value: Optional[Any] = None


class SettingsResponse(BaseModel):
    settings: List[SettingWithValue]


class SettingsUpdateRequest(BaseModel):
    settings: Dict[str, Any]
    secrets: Dict[str, Any]


@router.get("/")
async def get_settings() -> SettingsResponse:
    if Environment.is_production():
        raise HTTPException(
            status_code=403, detail="Settings cannot be read in production"
        )

    settings_registry = get_settings_registry()
    current_settings, current_secrets = load_settings()
    
    settings_with_values = []
    for setting in settings_registry:
        value = None
        if setting.is_secret:
            # For secrets, create a placeholder matching the length of the actual value
            secret_value = current_secrets.get(setting.env_var)
            if secret_value:
                value = "*" * len(str(secret_value))
            else:
                value = None
        else:
            # For non-secrets, return the actual value
            value = current_settings.get(setting.env_var)
        
        settings_with_values.append(
            SettingWithValue(
                package_name=setting.package_name,
                env_var=setting.env_var,
                group=setting.group,
                description=setting.description,
                is_secret=setting.is_secret,
                value=value
            )
        )

    return SettingsResponse(settings=settings_with_values)


@router.put("/")
async def update_settings(
    req: SettingsUpdateRequest,
) -> Dict[str, str]:
    if Environment.is_production():
        raise HTTPException(
            status_code=403, detail="Settings cannot be updated in production"
        )

    settings, secrets = load_settings()

    # Update settings (non-secrets)
    settings.update(req.settings)
    
    # Update secrets, but skip if the value is all asterisks (placeholder)
    for key, value in req.secrets.items():
        if value and isinstance(value, str) and all(c == '*' for c in value):
            # Skip placeholder values - don't update the secret
            continue
        secrets[key] = value

    save_settings(settings, secrets)

    Environment.load_settings()

    return {"message": "Settings updated successfully"}
