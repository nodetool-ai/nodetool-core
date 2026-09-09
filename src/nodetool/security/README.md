# NodeTool Security

Encrypted secret storage and key management.

## Components

- `crypto.py` — `SecretCrypto`: Fernet symmetric encryption (AES-128-CBC with HMAC-SHA256 authentication), per-user key derivation (PBKDF2-SHA256)
- `master_key.py` — `MasterKeyManager`: Master key from env var or system keychain
- `secret_helper.py` — `get_secret()`, `get_secret_required()`: Secret resolution (env var → default)

## Usage

```python
from nodetool.security.secret_helper import get_secret

api_key = await get_secret("OPENAI_API_KEY", user_id)
```

## Secret Resolution Order

For the lean Python worker, secrets come from environment variables. The TS
server handles database-stored secrets and passes them to the worker via env.

1. Environment variable (`os.environ`)
2. Not found (returns the provided default / `None`, or raises for `get_secret_required`)
