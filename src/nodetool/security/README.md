# NodeTool Security

Encrypted secret storage and key management.

## Components

- `crypto.py` — `SecretCrypto`: AES-256 encryption using Fernet, per-user key derivation (PBKDF2-SHA256)
- `master_key.py` — `MasterKeyManager`: Master key from env var or system keychain
- `secret_helper.py` — `get_secret()`, `get_secret_required()`: Secret resolution (env var → encrypted DB → default)
- `auth_provider.py` — `AuthProvider` protocol for authentication backends
- `providers/` — Auth provider implementations (local, static token, multi-user, supabase)

## Usage

```python
from nodetool.security.secret_helper import get_secret

api_key = await get_secret("OPENAI_API_KEY", user_id)
```

## Secret Resolution Order

1. Environment variable (`os.environ`)
2. Encrypted database (`Secret` model)
3. Not found (returns `None` or raises)
