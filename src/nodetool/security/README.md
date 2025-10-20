# NodeTool Secret Management System

This module provides secure, encrypted storage for sensitive data (API keys, passwords, tokens) in NodeTool.

## Overview

The secret management system consists of several components:

1. **Encryption** (`crypto.py`): AES-256 encryption using Fernet (symmetric encryption)
2. **Key Management** (`master_key.py`): Master key storage in system keychain or AWS Secrets Manager
3. **Database Model** (`models/secret.py`): Encrypted secret storage in database
4. **API Endpoints** (`api/settings.py`): REST API for managing secrets
5. **AWS Utility** (`aws_secrets_util.py`): CLI tool for AWS Secrets Manager integration

## Features

- **Per-user encryption**: Each user's secrets are encrypted with a unique derived key (master key + user_id as salt)
- **Multiple storage backends**: System keychain (macOS/Windows/Linux) or AWS Secrets Manager
- **Automatic key generation**: If no master key exists, one is generated and stored automatically
- **Secure API**: REST endpoints for CRUD operations on encrypted secrets
- **User isolation**: Users cannot access each other's secrets
- **Migration support**: Tools for backing up and restoring master keys

## Architecture

### Encryption Flow

1. **Master Key** is retrieved from (in order):
   - `SECRETS_MASTER_KEY` environment variable
   - AWS Secrets Manager (if `AWS_SECRETS_MASTER_KEY_NAME` is set)
   - System keychain
   - Auto-generated and stored in keychain

2. **User-specific Key Derivation**:
   - Master key + user_id (as salt) → Derived encryption key (PBKDF2-SHA256, 100k iterations)
   - Each user gets a unique encryption key even with shared master key

3. **Encryption**:
   - Plaintext secret → Fernet encryption → Base64-encoded ciphertext
   - Stored in database with user_id and key name

### Database Schema

```sql
CREATE TABLE nodetool_secrets (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    key TEXT NOT NULL,
    encrypted_value TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    UNIQUE(user_id, key)
);
```

## Usage

### Retrieving Secrets (Recommended)

Use the helper functions to get secret values at runtime:

```python
from nodetool.security import get_secret, get_secret_required

# Get a secret (returns None if not found)
api_key = await get_secret("OPENAI_API_KEY", user_id="user_123")

# Get a required secret (raises ValueError if not found)
api_key = await get_secret_required("OPENAI_API_KEY", user_id="user_123")

# For system-wide secrets (env only, synchronous)
from nodetool.security import get_secret_sync
api_key = get_secret_sync("OPENAI_API_KEY", default="sk-default")

# Check if a secret exists
from nodetool.security import has_secret
if await has_secret("OPENAI_API_KEY", user_id="user_123"):
    api_key = await get_secret("OPENAI_API_KEY", user_id="user_123")
```

The helper functions check in this order:
1. Environment variable (e.g., `OPENAI_API_KEY`)
2. Encrypted database secret for the user
3. Default value (if provided)

### Managing Secrets (Direct Database Access)

```python
from nodetool.models.secret import Secret

# Create a secret
secret = await Secret.create(
    user_id="user_123",
    key="OPENAI_API_KEY",
    value="sk-...",
    description="OpenAI API key for production"
)

# Retrieve a secret
secret = await Secret.find("user_123", "OPENAI_API_KEY")
decrypted_value = await secret.get_decrypted_value()

# Update a secret
await secret.update_value("new_sk-...")

# Delete a secret
await Secret.delete_secret("user_123", "OPENAI_API_KEY")

# List all secrets for a user
secrets, next_key = await Secret.list_for_user("user_123")
```

### API Endpoints

All endpoints are under `/api/settings/secrets` and require authentication:

```bash
# List all secrets (metadata only) for authenticated user
GET /api/settings/secrets
Headers: Authorization: Bearer <jwt_token>

# Get a specific secret (with optional decryption)
GET /api/settings/secrets/OPENAI_API_KEY?decrypt=true
Headers: Authorization: Bearer <jwt_token>

# Create or update a secret
POST /api/settings/secrets
Headers:
  Authorization: Bearer <jwt_token>
  Content-Type: application/json
Body:
{
  "key": "OPENAI_API_KEY",
  "value": "sk-...",
  "description": "OpenAI API key"
}

# Update a secret
PUT /api/settings/secrets/OPENAI_API_KEY
Headers:
  Authorization: Bearer <jwt_token>
  Content-Type: application/json
Body:
{
  "value": "new_sk-...",
  "description": "Updated key"
}

# Delete a secret
DELETE /api/settings/secrets/OPENAI_API_KEY
Headers: Authorization: Bearer <jwt_token>
```

**Authentication:**
- In development (local mode), authentication is bypassed and defaults to user_id="1"
- In production, requires JWT token from Supabase authentication
- User can only access their own secrets (enforced by database queries)

### Master Key Management

#### Local Development (Keychain)

The master key is automatically generated and stored in your system keychain:

- **macOS**: Keychain Access
- **Windows**: Credential Manager
- **Linux**: Secret Service (GNOME Keyring, KWallet)

No configuration required!

#### Production (AWS Secrets Manager)

For production deployments across multiple instances:

1. **Generate and store master key in AWS**:
```bash
python -m nodetool.security.aws_secrets_util generate \
  --secret-name nodetool-master-key \
  --region us-east-1
```

2. **Configure NodeTool to use AWS**:
```bash
export AWS_SECRETS_MASTER_KEY_NAME=nodetool-master-key
export AWS_REGION=us-east-1
```

3. **Store existing local key to AWS**:
```bash
python -m nodetool.security.aws_secrets_util store \
  --secret-name nodetool-master-key
```

#### Manual Key Management

**Export master key (for backup)**:
```python
from nodetool.security.master_key import MasterKeyManager

key = MasterKeyManager.export_master_key()
print(f"Master key: {key}")
# Store this securely! Anyone with this key can decrypt all secrets.
```

**Set custom master key**:
```python
from nodetool.security.master_key import MasterKeyManager

MasterKeyManager.set_master_key("your-master-key-here")
```

**Use environment variable**:
```bash
export SECRETS_MASTER_KEY="your-master-key-here"
```

## Secret Storage Priority

Secrets are loaded in the following priority order:

1. **Environment Variables** (highest priority)
   - System environment variables like `export OPENAI_API_KEY=sk-...`
   - Useful for CI/CD, Docker containers, and system-wide secrets
   - Not user-specific

2. **Encrypted Database** (recommended for user secrets)
   - Per-user encrypted secrets in the database
   - Managed via API or Secret model
   - User-isolated and encrypted at rest

3. **secrets.yaml** (deprecated, backwards compatibility only)
   - Legacy YAML file storage
   - **Not recommended**: Use database storage instead
   - Will be phased out in future versions

### Migration from secrets.yaml

If you have existing secrets in `~/.config/nodetool/secrets.yaml`, migrate them to the database:

```python
import yaml
from pathlib import Path
from nodetool.models.secret import Secret
from nodetool.config.settings import get_system_file_path

async def migrate_secrets_to_database(user_id: str):
    """Migrate secrets from secrets.yaml to encrypted database."""
    secrets_file = get_system_file_path("secrets.yaml")

    if not secrets_file.exists():
        print("No secrets.yaml found, nothing to migrate")
        return

    with open(secrets_file, "r") as f:
        secrets = yaml.safe_load(f) or {}

    for key, value in secrets.items():
        if value:  # Skip empty values
            await Secret.upsert(
                user_id=user_id,
                key=key,
                value=str(value),
                description=f"Migrated from secrets.yaml"
            )
            print(f"Migrated: {key}")

    print(f"Migration complete! Migrated {len(secrets)} secrets")
    print("You can now delete secrets.yaml")

# Run migration
await migrate_secrets_to_database("user_123")
```

## Security Considerations

### Best Practices

1. **Backup your master key**: If you lose it, all secrets are unrecoverable
2. **Rotate keys periodically**: Use the migration tools to re-encrypt with a new master key
3. **Use AWS Secrets Manager in production**: More secure than environment variables
4. **Never commit master keys**: Add to `.gitignore`
5. **Restrict API access in production**: The endpoints are development-only by default
6. **Migrate from secrets.yaml**: Use encrypted database storage instead

### Threat Model

**Protected against**:
- Database breaches (secrets are encrypted at rest)
- User isolation breaches (each user has unique derived keys)
- Accidental exposure of database dumps

**Not protected against**:
- Compromised master key (all secrets can be decrypted)
- Server-side code execution (attacker can call decryption functions)
- Memory dumps while secrets are decrypted

### Key Derivation

Using user_id as salt provides:
- **Isolation**: Each user's secrets are encrypted with a different key
- **Protection**: Even if one user's derived key is compromised, other users are safe
- **Determinism**: Same master key + user_id always produces the same derived key

## Migration and Backup

### Backup Master Key

```bash
# Export from keychain
python -c "from nodetool.security.master_key import MasterKeyManager; print(MasterKeyManager.export_master_key())"

# Store in AWS Secrets Manager
python -m nodetool.security.aws_secrets_util store --secret-name nodetool-master-key-backup
```

### Restore Master Key

```bash
# From AWS
python -m nodetool.security.aws_secrets_util retrieve --secret-name nodetool-master-key

# Set manually
python -c "from nodetool.security.master_key import MasterKeyManager; MasterKeyManager.set_master_key('your-key-here')"
```

### Rotate Master Key

⚠️ **Advanced**: Requires re-encrypting all secrets

```python
from nodetool.models.secret import Secret
from nodetool.security.master_key import MasterKeyManager
from nodetool.security.crypto import SecretCrypto

# 1. Get all secrets with old key
old_key = MasterKeyManager.get_master_key()
all_secrets = await Secret.query(limit=10000)

# 2. Decrypt all with old key
decrypted = {}
for secret in all_secrets:
    user_id = secret.user_id
    key = secret.key
    value = SecretCrypto.decrypt(secret.encrypted_value, old_key, user_id)
    decrypted[(user_id, key)] = (value, secret.description)

# 3. Generate and set new master key
new_key = SecretCrypto.generate_master_key()
MasterKeyManager.set_master_key(new_key)

# 4. Re-encrypt all secrets with new key
for (user_id, key), (value, description) in decrypted.items():
    await Secret.upsert(user_id, key, value, description)
```

## Troubleshooting

### "Failed to store master key in system keychain"

**Solution**: Set master key manually via environment variable:
```bash
export SECRETS_MASTER_KEY=$(python -c "from nodetool.security.crypto import SecretCrypto; print(SecretCrypto.generate_master_key())")
```

### "Access denied to AWS secret"

**Solution**: Ensure your IAM role/user has these permissions:
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "secretsmanager:GetSecretValue",
      "secretsmanager:CreateSecret",
      "secretsmanager:PutSecretValue"
    ],
    "Resource": "arn:aws:secretsmanager:*:*:secret:nodetool-*"
  }]
}
```

### "cryptography.fernet.InvalidToken"

**Cause**: Wrong master key or corrupted data

**Solution**:
- Ensure you're using the correct master key
- Check if `AWS_SECRETS_MASTER_KEY_NAME` points to the right secret
- Verify user_id matches the one used during encryption

## Testing

Run the test suite:
```bash
# Test crypto utilities
pytest tests/security/test_crypto.py -v

# Test master key management
pytest tests/security/test_master_key.py -v

# Test Secret model
pytest tests/models/test_secret.py -v
```

## Dependencies

- `cryptography>=43.0.0`: Encryption (Fernet/AES)
- `keyring>=25.5.0`: System keychain access
- `boto3` (optional): AWS Secrets Manager integration
