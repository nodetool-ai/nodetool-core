"""
Remote user management for self-hosted deployments.

This module provides SSH-based user management for Docker and SSH deployments.
Operations are performed remotely via SSH, never transmitting plaintext tokens.
"""

import base64
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
import yaml
from pydantic import BaseModel
from rich.console import Console

from nodetool.config.deployment import DockerDeployment, SSHDeployment
from nodetool.deploy.ssh import SSHConnection
from nodetool.security.providers.multi_user import UserInfo

console = Console()


class RemoteUserManager:
    """Manages users on remote deployments via SSH."""

    def __init__(
        self,
        deployment: DockerDeployment | SSHDeployment,
        users_file: str,
    ):
        self.deployment = deployment
        self.users_file = users_file
        self.ssh = SSHConnection(
            host=deployment.host,
            user=deployment.ssh.user,
            key_path=deployment.ssh.key_path,
            password=deployment.ssh.password,
            port=deployment.ssh.port,
        )

    def _hash_token(self, token: str) -> str:
        """Compute SHA256 hash of token."""
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _generate_user_id(self, username: str) -> str:
        """Generate unique user_id."""
        return f"user_{username}_{uuid.uuid4().hex[:8]}"

    def _load_remote_users(self) -> dict:
        """Load users from remote users.yaml."""
        try:
            exit_code, stdout, _stderr = self.ssh.execute(
                f"cat {self.users_file} 2>/dev/null || echo '{{}}'",
                check=False,
            )

            if exit_code != 0:
                return {}

            data = yaml.safe_load(stdout) or {}
            return data.get("users", {})
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load remote users: {e}[/]")
            return {}

    def _save_remote_users(self, users: dict) -> None:
        """Save users to remote users.yaml."""
        # Ensure directory exists
        users_dir = str(Path(self.users_file).parent)
        self.ssh.execute(f"mkdir -p {users_dir}")

        # Prepare file content
        content = yaml.dump({"users": users, "version": "1.0"}, default_flow_style=False)

        # Write via SSH (using base64 to avoid escaping issues)
        b64_content = base64.b64encode(content.encode()).decode()
        self.ssh.execute(f"echo {b64_content} | base64 -d > {self.users_file}")

        # Set restrictive permissions
        self.ssh.execute(f"chmod 0600 {self.users_file}")

    def add_user(self, username: str, role: str, token: str) -> None:
        """Add user to remote deployment.

        Args:
            username: Username to add
            role: User role ("admin" or "user")
            token: Plaintext bearer token (generated locally, hashed before storage)
        """
        users = self._load_remote_users()

        if username in users:
            raise ValueError(f"User '{username}' already exists on remote")

        # Generate user data
        user_id = self._generate_user_id(username)
        created_at = datetime.utcnow().isoformat() + "Z"

        user_info = {
            "user_id": user_id,
            "username": username,
            "role": role,
            "token_hash": self._hash_token(token),
            "created_at": created_at,
        }

        users[username] = user_info
        self._save_remote_users(users)

    def remove_user(self, username: str) -> None:
        """Remove user from remote deployment."""
        users = self._load_remote_users()

        if username not in users:
            raise ValueError(f"User '{username}' not found on remote")

        del users[username]
        self._save_remote_users(users)

    def reset_token(self, username: str, new_token: str) -> None:
        """Reset user token on remote deployment.

        Args:
            username: Username to reset
            new_token: New plaintext token (generated locally)
        """
        users = self._load_remote_users()

        if username not in users:
            raise ValueError(f"User '{username}' not found on remote")

        # Update user with new token hash
        existing_user = users[username]
        created_at = datetime.utcnow().isoformat() + "Z"

        users[username] = {
            "user_id": existing_user["user_id"],
            "username": username,
            "role": existing_user["role"],
            "token_hash": self._hash_token(new_token),
            "created_at": created_at,
        }

        self._save_remote_users(users)

    def list_users(self) -> dict[str, UserInfo]:
        """List all users from remote deployment."""
        users_dict = self._load_remote_users()

        # Convert to UserInfo objects
        return {username: UserInfo(**user_data) for username, user_data in users_dict.items()}
