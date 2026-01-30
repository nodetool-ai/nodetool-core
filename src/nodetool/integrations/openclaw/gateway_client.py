"""Client for communicating with OpenClaw Gateway.

This module provides functionality for registering with the Gateway,
sending messages, and maintaining heartbeat connections.
"""

import asyncio
from typing import Any, Optional

import aiohttp

from nodetool.config.logging_config import get_logger
from nodetool.integrations.openclaw.config import OpenClawConfig
from nodetool.integrations.openclaw.schemas import (
    GatewayMessage,
    NodeRegistration,
    NodeRegistrationResponse,
)

log = get_logger(__name__)


class GatewayClient:
    """Client for communicating with OpenClaw Gateway."""

    def __init__(self, config: Optional[OpenClawConfig] = None):
        """Initialize Gateway client.

        Args:
            config: OpenClaw configuration. If None, uses singleton instance.
        """
        self.config = config or OpenClawConfig.get_instance()
        self._session: Optional[aiohttp.ClientSession] = None
        self._registered = False
        self._node_token: Optional[str] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session exists."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the Gateway client and cleanup resources."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._session and not self._session.closed:
            await self._session.close()

    def _get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers for Gateway requests."""
        headers = {"Content-Type": "application/json"}

        # Use node token if registered, otherwise use gateway token
        token = self._node_token or self.config.gateway_token
        if token:
            headers["Authorization"] = f"Bearer {token}"

        return headers

    async def register(
        self, registration: NodeRegistration
    ) -> NodeRegistrationResponse:
        """Register this node with the OpenClaw Gateway.

        Args:
            registration: Node registration information.

        Returns:
            Registration response from the Gateway.

        Raises:
            aiohttp.ClientError: If registration fails.
        """
        session = await self._ensure_session()
        url = f"{self.config.gateway_url}/api/nodes/register"

        try:
            async with session.post(
                url, json=registration.model_dump(), headers=self._get_auth_headers()
            ) as response:
                response.raise_for_status()
                data = await response.json()
                result = NodeRegistrationResponse(**data)

                if result.success:
                    self._registered = True
                    if result.token:
                        self._node_token = result.token
                    log.info(
                        "Successfully registered with OpenClaw Gateway: node_id=%s",
                        result.node_id,
                    )
                else:
                    log.error(
                        "Failed to register with Gateway: %s", result.message
                    )

                return result

        except aiohttp.ClientError as e:
            log.error("Failed to register with Gateway: %s", e)
            raise

    async def send_message(self, message: GatewayMessage) -> dict[str, Any]:
        """Send a message through the Gateway.

        Args:
            message: Message to send.

        Returns:
            Response from the Gateway.

        Raises:
            aiohttp.ClientError: If message sending fails.
        """
        if not self._registered:
            raise RuntimeError("Node not registered with Gateway")

        session = await self._ensure_session()
        url = f"{self.config.gateway_url}/api/messages/send"

        async with session.post(
            url, json=message.model_dump(), headers=self._get_auth_headers()
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def send_heartbeat(self, status_data: dict[str, Any]) -> bool:
        """Send a heartbeat to the Gateway.

        Args:
            status_data: Current status information to send.

        Returns:
            True if heartbeat was successful, False otherwise.
        """
        if not self._registered:
            log.warning("Cannot send heartbeat: Node not registered")
            return False

        session = await self._ensure_session()
        url = f"{self.config.gateway_url}/api/nodes/{self.config.node_id}/heartbeat"

        try:
            async with session.post(
                url, json=status_data, headers=self._get_auth_headers()
            ) as response:
                response.raise_for_status()
                return True
        except aiohttp.ClientError as e:
            log.error("Failed to send heartbeat: %s", e)
            return False

    async def start_heartbeat(self, get_status_callback):
        """Start periodic heartbeat task.

        Args:
            get_status_callback: Async function that returns current status data.
        """
        if self._heartbeat_task:
            log.warning("Heartbeat task already running")
            return

        async def heartbeat_loop():
            while True:
                try:
                    await asyncio.sleep(self.config.heartbeat_interval)
                    status_data = await get_status_callback()
                    await self.send_heartbeat(status_data)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    log.error("Error in heartbeat loop: %s", e)

        self._heartbeat_task = asyncio.create_task(heartbeat_loop())
        log.info(
            "Started heartbeat task with interval %d seconds",
            self.config.heartbeat_interval,
        )

    @property
    def is_registered(self) -> bool:
        """Check if node is registered with Gateway."""
        return self._registered
