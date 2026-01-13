from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from nodetool.api.utils import current_user
from nodetool.models.secret import Secret


class EncryptedSecretPayload(BaseModel):
    user_id: str
    key: str
    encrypted_value: str
    description: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


router = APIRouter(prefix="/admin/secrets", tags=["admin-secrets"])


@router.post("/import")
async def import_secrets(
    secrets_payload: list[EncryptedSecretPayload],
    __user: str = Depends(current_user),
) -> dict[str, int]:
    """Import encrypted secrets (requires shared master key)."""
    try:
        imported = 0
        for item in secrets_payload:
            await Secret.upsert_encrypted(
                user_id=item.user_id,
                key=item.key,
                encrypted_value=item.encrypted_value,
                description=item.description,
                created_at=item.created_at,
                updated_at=item.updated_at,
            )
            imported += 1
        return {"imported": imported}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
