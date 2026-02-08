from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from nodetool.security.auth_provider import AuthProvider, AuthResult, TokenType


class UserInfo(BaseModel):
    """User information in users.yaml."""

    user_id: str = Field(..., description="Unique user identifier (UUID)")
    username: str = Field(..., description="Username for display")
    role: str = Field("user", description="User role")
    token_hash: str = Field(..., description="SHA256 hash of bearer token")
    created_at: str = Field(..., description="ISO timestamp when user was created")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v):
        if v not in ("admin", "user"):
            raise ValueError("role must be 'admin' or 'user'")
        return v


class UsersFile(BaseModel):
    """Structure of users.yaml file."""

    users: dict[str, UserInfo] = Field(default_factory=dict)
    version: str = "1.0"


class MultiUserAuthProvider(AuthProvider):
    """File-based multi-user authentication provider.

    Validates bearer tokens against a YAML file containing user records.
    Each user has a username, role, and hashed token.
    """

    def __init__(self, users_file: str):
        self.users_file = Path(users_file).expanduser()
        self._users_cache: Optional[dict[str, UserInfo]] = None
        self._cache_modified_time: Optional[float] = None

    def _load_users(self) -> dict[str, UserInfo]:
        """Load users from YAML file with caching."""
        if not self.users_file.exists():
            return {}

        # Check if file has been modified
        modified_time = None
        try:
            modified_time = self.users_file.stat().st_mtime
            if self._cache_modified_time == modified_time and self._users_cache is not None:
                return self._users_cache
        except OSError:
            pass

        # Load from file
        import yaml

        with open(self.users_file) as f:
            data = yaml.safe_load(f) or {}

        users_file_obj = UsersFile.model_validate(data)
        self._users_cache = users_file_obj.users
        self._cache_modified_time = modified_time
        return self._users_cache

    def _hash_token(self, token: str) -> str:
        """Compute SHA256 hash of token."""
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    async def verify_token(self, token: str) -> AuthResult:
        """Validate a bearer token against users.yaml."""
        if not token:
            return AuthResult(ok=False, error="Missing bearer token")

        users = self._load_users()
        token_hash = self._hash_token(token)

        # Find user by token hash
        for _username, user in users.items():
            if user.token_hash == token_hash:
                return AuthResult(ok=True, user_id=user.user_id, token_type=TokenType.USER)

        return AuthResult(ok=False, error="Invalid bearer token")

    def get_user(self, user_id: str) -> Optional[UserInfo]:
        """Get user information by user_id."""
        users = self._load_users()
        for user in users.values():
            if user.user_id == user_id:
                return user
        return None

    def list_users(self) -> dict[str, UserInfo]:
        """List all users (for management)."""
        return self._load_users()

    def clear_caches(self) -> None:
        """Clear the users cache."""
        self._users_cache = None
        self._cache_modified_time = None
