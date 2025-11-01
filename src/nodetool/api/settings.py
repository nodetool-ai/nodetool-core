import os
from fastapi import APIRouter, Depends, HTTPException
from nodetool.config.environment import Environment
from nodetool.config.settings import load_settings, save_settings
from nodetool.config.configuration import get_settings_registry, get_secrets_registry
from nodetool.models.secret import Secret
from nodetool.api.utils import current_user
from pydantic import BaseModel
from typing import Any, Dict, List, Optional


router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingWithValue(BaseModel):
    package_name: str
    env_var: str
    group: str
    description: str
    enum: Optional[List[str]] = None
    value: Optional[Any] = None


class SettingsResponse(BaseModel):
    settings: List[SettingWithValue]


class SettingsUpdateRequest(BaseModel):
    settings: Dict[str, Any]
    secrets: Dict[str, Any]


class SecretCreateRequest(BaseModel):
    key: str
    value: str
    description: Optional[str] = None


class SecretUpdateRequest(BaseModel):
    value: str
    description: Optional[str] = None


class SecretResponse(BaseModel):
    id: Optional[str] = None
    user_id: Optional[str] = None
    key: str
    description: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    is_configured: bool = False


class SecretsListResponse(BaseModel):
    secrets: List[SecretResponse]
    next_key: Optional[str] = None


@router.get("/")
async def get_settings(user: str = Depends(current_user)) -> SettingsResponse:
    if Environment.is_production():
        raise HTTPException(
            status_code=403, detail="Settings cannot be read in production"
        )

    settings_registry = get_settings_registry()
    current_settings = load_settings()  # Load only settings.yaml, not secrets.yaml

    settings_with_values = []
    for setting in settings_registry:
        value = current_settings.get(setting.env_var)

        settings_with_values.append(
            SettingWithValue(
                package_name=setting.package_name,
                env_var=setting.env_var,
                group=setting.group,
                description=setting.description,
                value=value,
                enum=setting.enum,
            )
        )

    return SettingsResponse(settings=settings_with_values)


@router.put("/")
async def update_settings(
    req: SettingsUpdateRequest,
    user: str = Depends(current_user)
) -> Dict[str, str]:
    if Environment.is_production():
        raise HTTPException(
            status_code=403, detail="Settings cannot be updated in production"
        )

    settings = load_settings()  # Load only settings.yaml, ignore secrets.yaml

    # Update settings (non-secrets) to settings.yaml
    settings.update(req.settings)

    # Update secrets to database (not secrets.yaml)
    for key, value in req.secrets.items():
        if value and isinstance(value, str) and all(c == "*" for c in value):
            # Skip placeholder values - don't update the secret
            continue

        # Save to encrypted database
        await Secret.upsert(
            user_id=user,
            key=key,
            value=value,
            description=f"Secret for {key}"
        )

    # Save only non-secret settings to settings.yaml
    # Secrets are now in the database, so we don't save them to secrets.yaml
    save_settings(settings, {})

    Environment.load_settings()

    return {"message": "Settings updated successfully"}


# ============================================================================
# Encrypted Secrets Endpoints
# ============================================================================


@router.get("/secrets")
async def list_secrets(
    limit: int = 100,
    start_key: Optional[str] = None,
    user: str = Depends(current_user)
) -> SecretsListResponse:
    """
    List all possible secrets from the settings registry.

    For each possible secret, returns:
    - id, user_id, key, description, created_at, updated_at if the user has configured it
    - null values if the secret is not configured

    Returns metadata only (no decrypted values).
    """
    secrets_registry = get_secrets_registry()

    # Build response with all possible secrets
    secrets_response = []
    for setting in secrets_registry:
        secret = await Secret.find(user, setting.env_var)
        if secret:
            secrets_response.append(
                SecretResponse(
                    id=secret.id,
                    user_id=secret.user_id,
                    key=setting.env_var,
                    description=secret.description or setting.description,
                    created_at=secret.created_at.isoformat(),
                    updated_at=secret.updated_at.isoformat(),
                    is_configured=True
                )
            )
        else:
            # Create a response for unconfigured secret
            secrets_response.append(
                SecretResponse(
                    key=setting.env_var,
                    description=setting.description,
                    is_configured=False
                )
            )

    # Apply limit
    limited_secrets = secrets_response[:limit]
    next_key = None
    if len(secrets_response) > limit:
        next_key = secrets_response[limit].key

    return SecretsListResponse(
        secrets=limited_secrets,
        next_key=next_key
    )


@router.get("/secrets/{key}")
async def get_secret(
    key: str,
    decrypt: bool = False,
    user: str = Depends(current_user)
) -> Dict[str, Any]:
    """
    Get a specific secret by key.

    Args:
        key: The secret key.
        decrypt: If True, return the decrypted value. WARNING: Use with caution!

    Returns:
        Secret metadata and optionally the decrypted value.
    """
    secret = await Secret.find(user, key)

    if not secret:
        raise HTTPException(status_code=404, detail="Secret not found")

    response = secret.to_dict_safe()
    response["is_configured"] = True

    if decrypt:
        try:
            response["value"] = await secret.get_decrypted_value()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to decrypt secret: {str(e)}"
            )

    return response


@router.put("/secrets/{key}")
async def update_secret(
    key: str,
    req: SecretUpdateRequest,
    user: str = Depends(current_user)
) -> SecretResponse:
    """
    Update or create a secret.

    If the secret exists, updates its value and description.
    If the secret does not exist, creates it (only if it's in the settings registry).
    """
    # Validate that the key is in the secrets registry
    secrets_registry = get_secrets_registry()
    secret_setting = next((s for s in secrets_registry if s.env_var == key), None)
    if not secret_setting:
        raise HTTPException(
            status_code=404,
            detail=f"Secret key '{key}' is not available. Only secrets from the registry can be configured."
        )

    try:
        # Use upsert to create or update the secret
        secret = await Secret.upsert(
            user_id=user,
            key=key,
            value=req.value,
            description=req.description or secret_setting.description
        )

        return SecretResponse(
            id=secret.id,
            user_id=secret.user_id,
            key=secret.key,
            description=secret.description,
            created_at=secret.created_at.isoformat(),
            updated_at=secret.updated_at.isoformat(),
            is_configured=True
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update secret: {str(e)}"
        )


@router.delete("/secrets/{key}")
async def delete_secret(
    key: str,
    user: str = Depends(current_user)
) -> Dict[str, str]:
    """
    Delete a secret.
    """
    success = await Secret.delete_secret(user, key)

    if not success:
        raise HTTPException(status_code=404, detail="Secret not found")

    return {"message": "Secret deleted successfully"}
