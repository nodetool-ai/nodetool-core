"""
Provides functionality for managing WebSocket connections and broadcasting updates,
primarily system statistics, to connected clients.
"""

from fastapi import WebSocket
import asyncio
from typing import Literal, Set

from pydantic import BaseModel

from nodetool.common.environment import Environment
from nodetool.common.system_stats import SystemStats
from nodetool.common.system_stats import get_system_stats


class SystemStatsUpdate(BaseModel):
    type: Literal["system_stats"] = "system_stats"
    stats: SystemStats


WebSocketUpdate = SystemStatsUpdate


class WebSocketUpdates:
    """
    Manages WebSocket connections and broadcasts updates to connected clients.

    This class handles accepting new connections, managing disconnections,
    and broadcasting system statistics updates periodically to all active clients.
    It ensures that the stats broadcasting task runs only when there are active
    connections.
    """

    def __init__(self):
        """Initializes the WebSocketUpdates manager."""
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self.log = Environment.get_logger()
        self.log.info("WebSocketUpdates: instance initialized")
        self._stats_task = None

    async def connect(self, websocket: WebSocket):
        """
        Accepts a new WebSocket connection and adds it to the active set.

        Starts the system stats broadcasting task if this is the first connection.

        Args:
            websocket: The WebSocket connection object.
        """
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
            self.log.info(
                f"WebSocketUpdates: New connection accepted. Total: {len(self.active_connections)}"
            )
            # Start stats broadcasting if this is the first connection
            if len(self.active_connections) == 1:
                await self._start_stats_broadcast()

    async def disconnect(self, websocket: WebSocket):
        """
        Removes a WebSocket connection from the active set upon disconnection.

        Stops the system stats broadcasting task if no connections remain.

        Args:
            websocket: The WebSocket connection object.
        """
        async with self._lock:
            self.active_connections.remove(websocket)
            self.log.info(
                f"WebSocketUpdates: disconnected. Remaining: {len(self.active_connections)}"
            )
            # Stop stats broadcasting if no connections remain
            if len(self.active_connections) == 0:
                await self._stop_stats_broadcast()

    async def _start_stats_broadcast(self):
        """Starts the background task that periodically broadcasts system stats."""
        if self._stats_task is None:
            self._stats_task = asyncio.create_task(self._broadcast_stats())
            self.log.info("WebSocketUpdates: Started system stats broadcasting")

    async def _stop_stats_broadcast(self):
        """Stops the background task that broadcasts system stats."""
        if self._stats_task:
            self._stats_task.cancel()
            self._stats_task = None
            self.log.info("WebSocketUpdates: Stopped system stats broadcasting")

    async def _broadcast_stats(self):
        """
        Periodically fetches system stats and broadcasts them to all clients.

        Runs in a loop until cancelled. Handles potential exceptions during
        stats fetching or broadcasting.
        """
        while True:
            try:
                stats = get_system_stats()
                await self.broadcast_update(SystemStatsUpdate(stats=stats))
                await asyncio.sleep(1)  # Wait for 1 second
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log.error(f"WebSocketUpdates: Error broadcasting stats: {str(e)}")
                await asyncio.sleep(1)  # Wait before retrying

    async def broadcast_update(self, update: WebSocketUpdate):
        """Broadcast any update to all connected clients"""
        """
        Broadcasts a given update message to all connected WebSocket clients.

        Args:
            update: The update object (e.g., SystemStatsUpdate) to broadcast.
        """
        json_message = update.model_dump_json()

        async with self._lock:
            for websocket in self.active_connections:
                await websocket.send_text(json_message)
                self.log.debug("WebSocketUpdates: Successfully sent message to client")

    async def handle_client(self, websocket: WebSocket):
        """
        Manages a single client connection lifecycle.

        Connects the client, listens for messages (currently logs them),
        and handles disconnection on error.

        Args:
            websocket: The WebSocket connection object for the client.
        """
        client_id = id(websocket)  # Use websocket id for tracking
        self.log.info(
            f"WebSocketUpdates: New client connection handler started (ID: {client_id})"
        )

        await self.connect(websocket)
        try:
            while True:
                message = await websocket.receive_text()
                self.log.debug(
                    f"WebSocketUpdates: Received message from client {client_id}: {message[:100]}..."
                )
        except Exception as e:
            self.log.error(
                f"WebSocketUpdates: Client connection error (ID: {client_id}): {str(e)}"
            )
            await self.disconnect(websocket)


# Global singleton instance
websocket_updates = WebSocketUpdates()
