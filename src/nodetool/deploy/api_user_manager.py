"""
API-based user management for remote deployments.

This module provides client-side user management via API.
Works with all deployment types: Docker, Root, GCP, RunPod.
"""

from typing import Optional

import httpx


class APIUserManager:
    """Manages users via API (works with ALL deployment types)."""

    def __init__(self, server_url: str, admin_token: str | None):
        """Initialize API user manager.

        Args:
            server_url: Base URL of the deployment (e.g., http://example.com:7777)
            admin_token: Bearer token of an admin user
        """
        if admin_token is None:
            raise ValueError("admin_token is required")
        self.server_url = server_url.rstrip("/")
        self.admin_token = admin_token

    async def list_users(self) -> list:
        """List all users via API.

        Returns:
            List of user dictionaries (with masked tokens)

        Raises:
            httpx.HTTPError: If API request fails
        """
        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {self.admin_token}",
                "Content-Type": "application/json",
            }
            response = await client.get(
                f"{self.server_url}/api/users/",
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("users", [])

    async def add_user(self, username: str, role: str = "user") -> dict:
        """Add user via API.

        Args:
            username: Username for the new user
            role: User role ("admin" or "user")

        Returns:
            User dictionary with plaintext token (only shown on creation)

        Raises:
            httpx.HTTPError: If API request fails
        """
        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {self.admin_token}",
                "Content-Type": "application/json",
            }
            payload = {"username": username, "role": role}
            response = await client.post(
                f"{self.server_url}/api/users/",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            return response.json()

    async def reset_token(self, username: str) -> dict:
        """Generate new token for user via API.

        Args:
            username: Username to reset token for

        Returns:
            User dictionary with new plaintext token

        Raises:
            httpx.HTTPError: If API request fails
        """
        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {self.admin_token}",
                "Content-Type": "application/json",
            }
            payload = {"username": username}
            response = await client.post(
                f"{self.server_url}/api/users/reset-token",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            return response.json()

    async def remove_user(self, username: str) -> dict:
        """Remove user via API.

        Args:
            username: Username to remove

        Returns:
            Success message dictionary

        Raises:
            httpx.HTTPError: If API request fails
        """
        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {self.admin_token}",
                "Content-Type": "application/json",
            }
            response = await client.delete(
                f"{self.server_url}/api/users/{username}",
                headers=headers,
            )
            response.raise_for_status()
            return response.json()
