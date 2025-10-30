from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from nodetool.api.utils import current_user
from nodetool.models.secret import Secret


class EncryptedSecretPayload(BaseModel):
    user_id: str
    key: str
    encrypted_value: str
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


router = APIRouter(prefix="/admin/secrets", tags=["admin-secrets"])


@router.post("/import")
async def import_secrets(
    secrets_payload: List[EncryptedSecretPayload],
    user: str = Depends(current_user),
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
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
