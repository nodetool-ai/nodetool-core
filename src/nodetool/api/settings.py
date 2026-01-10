import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from nodetool.api.utils import current_user
from nodetool.config.configuration import get_secrets_registry, get_settings_registry
from nodetool.config.environment import Environment
from nodetool.config.settings import load_settings, save_settings
from nodetool.models.secret import Secret

router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingWithValue(BaseModel):
    package_name: str
    env_var: str
    group: str
    description: str
    enum: list[str] | None = None
    value: Any | None = None
    is_secret: bool = False


class SettingsResponse(BaseModel):
    settings: list[SettingWithValue]


class SettingsUpdateRequest(BaseModel):
    settings: dict[str, Any]
    secrets: dict[str, Any]


class SecretCreateRequest(BaseModel):
    key: str
    value: str
    description: str | None = None


class SecretUpdateRequest(BaseModel):
    value: str
    description: str | None = None


class SecretResponse(BaseModel):
    id: str | None = None
    user_id: str | None = None
    key: str
    description: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    is_configured: bool = False


class SecretsListResponse(BaseModel):
    secrets: list[SecretResponse]
    next_key: str | None = None


@router.get("/")
async def get_settings(user: str = Depends(current_user)) -> SettingsResponse:
    if Environment.is_production():
        raise HTTPException(status_code=403, detail="Settings cannot be read in production")

    settings_registry = get_settings_registry()
    secrets_registry = get_secrets_registry()
    current_settings = load_settings()  # Load only settings.yaml, not secrets.yaml

    # Load secret values from database
    from nodetool.models.secret import Secret

    secret_values = {}
    for secret_setting in secrets_registry:
        secret = await Secret.find(user, secret_setting.env_var)
        if secret:
            # Don't return the actual secret value, just indicate it's configured
            secret_values[secret_setting.env_var] = "****"
        else:
            secret_values[secret_setting.env_var] = None

    settings_with_values = []

    # Add regular settings
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
                is_secret=False,
            )
        )

    # Add secrets (convert Secret to Setting format)
    for secret_setting in secrets_registry:
        value = secret_values.get(secret_setting.env_var)

        settings_with_values.append(
            SettingWithValue(
                package_name=secret_setting.package_name,
                env_var=secret_setting.env_var,
                group=secret_setting.group,
                description=secret_setting.description,
                value=value,
                enum=None,
                is_secret=True,
            )
        )

    return SettingsResponse(settings=settings_with_values)


@router.put("/")
async def update_settings(req: SettingsUpdateRequest, user: str = Depends(current_user)) -> dict[str, str]:
    if Environment.is_production():
        raise HTTPException(status_code=403, detail="Settings cannot be updated in production")

    settings = load_settings()  # Load only settings.yaml, ignore secrets.yaml

    # Update settings (non-secrets) to settings.yaml
    settings.update(req.settings)

    # Update secrets to database (not secrets.yaml)
    for key, value in req.secrets.items():
        if value and isinstance(value, str) and all(c == "*" for c in value):
            # Skip placeholder values - don't update the secret
            continue

        # Save to encrypted database
        await Secret.upsert(user_id=user, key=key, value=value, description=f"Secret for {key}")

    # Save only non-secret settings to settings.yaml
    # Secrets are now in the database, so we don't save them to secrets.yaml
    save_settings(settings)

    Environment.load_settings()

    return {"message": "Settings updated successfully"}


# ============================================================================
# Encrypted Secrets Endpoints
# ============================================================================


@router.get("/secrets")
async def list_secrets(
    limit: int = 100, start_key: str | None = None, user: str = Depends(current_user)
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
                    is_configured=True,
                )
            )
        else:
            # Create a response for unconfigured secret
            secrets_response.append(
                SecretResponse(key=setting.env_var, description=setting.description, is_configured=False)
            )

    # Apply limit
    limited_secrets = secrets_response[:limit]
    next_key = None
    if len(secrets_response) > limit:
        next_key = secrets_response[limit].key

    return SecretsListResponse(secrets=limited_secrets, next_key=next_key)


@router.get("/secrets/{key}")
async def get_secret(key: str, decrypt: bool = False, user: str = Depends(current_user)) -> dict[str, Any]:
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
            raise HTTPException(status_code=500, detail=f"Failed to decrypt secret: {str(e)}") from e

    return response


@router.put("/secrets/{key}")
async def update_secret(key: str, req: SecretUpdateRequest, user: str = Depends(current_user)) -> SecretResponse:
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
            detail=f"Secret key '{key}' is not available. Only secrets from the registry can be configured.",
        )

    try:
        # Use upsert to create or update the secret
        secret = await Secret.upsert(
            user_id=user, key=key, value=req.value, description=req.description or secret_setting.description
        )

        return SecretResponse(
            id=secret.id,
            user_id=secret.user_id,
            key=secret.key,
            description=secret.description,
            created_at=secret.created_at.isoformat(),
            updated_at=secret.updated_at.isoformat(),
            is_configured=True,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update secret: {str(e)}") from e


@router.delete("/secrets/{key}")
async def delete_secret(key: str, user: str = Depends(current_user)) -> dict[str, str]:
    """
    Delete a secret.
    """
    success = await Secret.delete_secret(user, key)

    if not success:
        raise HTTPException(status_code=404, detail="Secret not found")

    return {"message": "Secret deleted successfully"}
