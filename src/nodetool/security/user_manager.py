from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

from nodetool.security.providers.multi_user import UserInfo, UsersFile


class CreateUserResult(BaseModel):
    """Result of creating a user."""

    username: str
    user_id: str
    role: str
    token: str
    created_at: str


class UserManager:
    """Manages users.yaml file for file-based multi-user auth."""

    def __init__(self, users_file: Optional[str] = None):
        if users_file:
            self.users_file = Path(users_file).expanduser()
        else:
            # Read from USERS_FILE environment variable
            from nodetool.config.environment import Environment

            users_file_env = Environment.get("USERS_FILE")
            if users_file_env:
                self.users_file = Path(users_file_env).expanduser()
            else:
                # Fallback to system file path
                from nodetool.config.settings import get_system_file_path

                self.users_file = get_system_file_path("users.yaml")

    def _load(self) -> UsersFile:
        """Load users file."""
        if not self.users_file.exists():
            return UsersFile(users={}, version="1.0")

        with open(self.users_file) as f:
            data = yaml.safe_load(f) or {}

        return UsersFile.model_validate(data)

    def _save(self, users_file: UsersFile) -> None:
        """Save users file with secure permissions."""
        self.users_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.users_file, "w") as f:
            yaml.dump(users_file.model_dump(mode="json", exclude_none=True), f)

        self.users_file.chmod(0o600)

    def _generate_user_id(self, username: str) -> str:
        """Generate unique user_id from username."""
        import uuid

        return f"user_{username}_{uuid.uuid4().hex[:8]}"

    def _generate_token(self) -> str:
        """Generate cryptographically secure bearer token."""
        return secrets.token_urlsafe(32)

    def _hash_token(self, token: str) -> str:
        """Compute SHA256 hash of token."""
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def add_user(self, username: str, role: str = "user") -> CreateUserResult:
        """Add a new user.

        Args:
            username: Username for user
            role: Either "admin" or "user"

        Returns:
            CreateUserResult with plaintext token (only shown once!)

        Raises:
            ValueError: If username already exists
        """
        users_file = self._load()

        if username in users_file.users:
            raise ValueError(f"User '{username}' already exists")

        token = self._generate_token()
        user_id = self._generate_user_id(username)
        created_at = datetime.now(UTC).isoformat()

        user_info = UserInfo(
            user_id=user_id, username=username, role=role, token_hash=self._hash_token(token), created_at=created_at
        )

        users_file.users[username] = user_info
        self._save(users_file)

        return CreateUserResult(username=username, user_id=user_id, role=role, token=token, created_at=created_at)

    def remove_user(self, username: str) -> None:
        """Remove a user."""
        users_file = self._load()

        if username not in users_file.users:
            raise ValueError(f"User '{username}' not found")

        del users_file.users[username]
        self._save(users_file)

    def reset_token(self, username: str) -> CreateUserResult:
        """Generate new token for existing user.

        Args:
            username: Username to reset token for

        Returns:
            CreateUserResult with new plaintext token

        Raises:
            ValueError: If user not found
        """
        users_file = self._load()

        if username not in users_file.users:
            raise ValueError(f"User '{username}' not found")

        existing_user = users_file.users[username]
        token = self._generate_token()
        created_at = datetime.now(UTC).isoformat()

        updated_user = UserInfo(
            user_id=existing_user.user_id,
            username=username,
            role=existing_user.role,
            token_hash=self._hash_token(token),
            created_at=created_at,
        )

        users_file.users[username] = updated_user
        self._save(users_file)

        return CreateUserResult(
            username=username,
            user_id=existing_user.user_id,
            role=existing_user.role,
            token=token,
            created_at=created_at,
        )

    def list_users(self) -> dict[str, UserInfo]:
        """List all users."""
        return self._load().users

    def get_user(self, username: str) -> Optional[UserInfo]:
        """Get user by username."""
        users_file = self._load()
        return users_file.users.get(username)
