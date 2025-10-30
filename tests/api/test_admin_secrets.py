import os
from datetime import datetime

import pytest

from nodetool.models.secret import Secret
from nodetool.security.crypto import SecretCrypto
from nodetool.security.master_key import MasterKeyManager


@pytest.mark.asyncio
async def test_import_encrypted_secrets(client, headers, monkeypatch):
    master_key = SecretCrypto.generate_master_key()
    monkeypatch.setenv("SECRETS_MASTER_KEY", master_key)
    MasterKeyManager.clear_cache()

    user_id = "user-123"
    key = "OPENAI_API_KEY"
    plaintext = "sk-test"
    encrypted = SecretCrypto.encrypt(plaintext, master_key, user_id)

    payload = [
        {
            "user_id": user_id,
            "key": key,
            "encrypted_value": encrypted,
            "description": "imported",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
    ]

    response = client.post("/admin/secrets/import", json=payload, headers=headers)
    assert response.status_code == 200
    assert response.json()["imported"] == 1

    secret = await Secret.find(user_id, key)
    assert secret is not None
    assert secret.encrypted_value == encrypted
    assert secret.description == "imported"
    decrypted = await secret.get_decrypted_value()
    assert decrypted == plaintext

    # Update existing secret
    new_encrypted = SecretCrypto.encrypt("sk-new", master_key, user_id)
    payload[0]["encrypted_value"] = new_encrypted
    payload[0]["description"] = "updated"

    response = client.post("/admin/secrets/import", json=payload, headers=headers)
    assert response.status_code == 200
    assert response.json()["imported"] == 1

    secret = await Secret.find(user_id, key)
    assert secret.encrypted_value == new_encrypted
    assert secret.description == "updated"
    assert await secret.get_decrypted_value() == "sk-new"
