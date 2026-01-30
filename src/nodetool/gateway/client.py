"""
Gateway WebSocket client for connecting NodeTool to OpenClaw Gateway.

This client allows NodeTool to act as a node in a distributed gateway
architecture, receiving and executing workflows from a central gateway server.
"""

import asyncio
import json
import uuid
from typing import TYPE_CHECKING, Any, Callable, Optional

import websockets

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

from nodetool.config.logging_config import get_logger
from nodetool.gateway.protocol import (
    AckMessage,
    CommandRequest,
    CommandResponse,
    ErrorMessage,
    GatewayMessage,
    NodeHeartbeat,
    NodeRegistration,
    WorkflowRequest,
    WorkflowResponse,
    WorkflowUpdate,
)
from nodetool.tools import (
    AssetTools,
    CollectionTools,
    JobTools,
    ModelTools,
    NodeTools,
    WorkflowTools,
)


class GatewayClient:
    """
    WebSocket client for connecting to OpenClaw Gateway.

    Features:
    - Automatic reconnection with exponential backoff
    - Heartbeat mechanism to maintain connection
    - Workflow execution with streaming updates
    - MCP-like command handling
    """

    def __init__(
        self,
        gateway_url: str,
        node_id: Optional[str] = None,
        auth_token: Optional[str] = None,
        user_id: str = "1",
        heartbeat_interval: float = 30.0,
        reconnect_delay: float = 5.0,
    ):
        """
        Initialize gateway client.

        Args:
            gateway_url: WebSocket URL of the gateway server
            node_id: Unique identifier for this node (auto-generated if None)
            auth_token: Authentication token for gateway connection
            user_id: User ID for workflow execution
            heartbeat_interval: Seconds between heartbeat messages
            reconnect_delay: Initial delay for reconnection attempts
        """
        self.gateway_url = gateway_url
        self.node_id = node_id or f"node-{uuid.uuid4()}"
        self.auth_token = auth_token
        self.user_id = user_id
        self.heartbeat_interval = heartbeat_interval
        self.reconnect_delay = reconnect_delay

        self.websocket: Optional[ClientConnection] = None
        self.running = False
        self.connected = False

        # Tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Request tracking
        self._pending_requests: dict[str, asyncio.Future] = {}

        # Message handlers
        self._handlers: dict[str, Callable] = {
            "workflow_request": self._handle_workflow_request,
            "command_request": self._handle_command_request,
            "ack": self._handle_ack,
            "error": self._handle_error,
        }

        self.log = get_logger(__name__)

    async def connect(self) -> bool:
        """
        Connect to the gateway server.

        Returns:
            True if connected successfully, False otherwise
        """
        try:
            # Build URL with auth token if provided
            url = self.gateway_url
            if self.auth_token:
                url = f"{url}?token={self.auth_token}"

            self.log.info(f"Connecting to gateway at {self.gateway_url}")
            self.websocket = await websockets.connect(url)
            self.connected = True
            self.running = True

            # Send node registration
            await self._send_registration()

            # Start background tasks
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            self.log.info(f"Successfully connected to gateway as node {self.node_id}")
            return True

        except Exception as e:
            self.log.error(f"Failed to connect to gateway: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        """Disconnect from the gateway server."""
        self.log.info("Disconnecting from gateway")
        self.running = False
        self.connected = False

        # Cancel tasks
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close websocket
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

        self.log.info("Disconnected from gateway")

    async def run(self):
        """
        Run the gateway client with automatic reconnection.

        This is the main entry point for running the client.
        """
        self.running = True

        while self.running:
            try:
                if await self.connect():
                    # Wait for disconnect
                    if self._receive_task:
                        await self._receive_task
            except Exception as e:
                self.log.error(f"Error in gateway client: {e}")

            if self.running:
                self.log.info(f"Reconnecting in {self.reconnect_delay} seconds...")
                await asyncio.sleep(self.reconnect_delay)
                # Exponential backoff (max 60 seconds)
                self.reconnect_delay = min(self.reconnect_delay * 2, 60.0)
            else:
                break

    async def _send_registration(self):
        """Send node registration message to gateway."""
        registration = NodeRegistration(
            node_id=self.node_id,
            capabilities={
                "workflow_execution": True,
                "commands": [
                    "list_workflows",
                    "get_workflow",
                    "run_workflow",
                    "list_jobs",
                    "get_job",
                    "get_job_logs",
                    "list_assets",
                    "list_nodes",
                    "list_models",
                ],
            },
            metadata={
                "user_id": self.user_id,
            },
        )
        await self._send_message(registration)

    async def _heartbeat_loop(self):
        """Send periodic heartbeat messages."""
        try:
            while self.running and self.connected:
                await asyncio.sleep(self.heartbeat_interval)
                if self.connected:
                    heartbeat = NodeHeartbeat(node_id=self.node_id)
                    await self._send_message(heartbeat)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.log.error(f"Error in heartbeat loop: {e}")

    async def _receive_loop(self):
        """Receive and process messages from gateway."""
        try:
            if not self.websocket:
                return

            async for message in self.websocket:
                try:
                    # Parse message
                    data = json.loads(message) if isinstance(message, str) else message

                    # Dispatch to handler
                    msg_type = data.get("type")
                    if msg_type in self._handlers:
                        await self._handlers[msg_type](data)
                    else:
                        self.log.warning(f"Unknown message type: {msg_type}")

                except Exception as e:
                    self.log.error(f"Error processing message: {e}")

        except websockets.exceptions.ConnectionClosed:
            self.log.warning("Gateway connection closed")
            self.connected = False
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.log.error(f"Error in receive loop: {e}")
            self.connected = False

    async def _send_message(self, message: GatewayMessage):
        """Send a message to the gateway."""
        if not self.websocket or not self.connected:
            raise RuntimeError("Not connected to gateway")

        try:
            payload = message.model_dump_json()
            await self.websocket.send(payload)
            self.log.debug(f"Sent message: {message.type}")
        except Exception as e:
            self.log.error(f"Error sending message: {e}")
            raise

    async def _handle_workflow_request(self, data: dict[str, Any]):
        """Handle workflow execution request from gateway."""
        try:
            request = WorkflowRequest.model_validate(data)
            self.log.info(f"Received workflow request: {request.request_id}")

            # Send acknowledgment
            ack = AckMessage(
                request_id=request.request_id,
                message="Workflow execution started",
            )
            await self._send_message(ack)

            # Execute workflow in background
            _ = asyncio.create_task(self._execute_workflow(request))  # noqa: RUF006

        except Exception as e:
            self.log.error(f"Error handling workflow request: {e}")
            error = ErrorMessage(error=str(e))
            await self._send_message(error)

    async def _execute_workflow(self, request: WorkflowRequest):
        """Execute a workflow and stream results back to gateway."""
        try:
            # Run workflow
            if request.workflow_id:
                result = await WorkflowTools.run_workflow_tool(
                    workflow_id=request.workflow_id,
                    params=request.params,
                    user_id=request.user_id,
                )
            elif request.graph:
                result = await WorkflowTools.run_graph(
                    graph=request.graph,
                    params=request.params,
                    user_id=request.user_id,
                )
            else:
                raise ValueError("Either workflow_id or graph must be provided")

            # Send success response
            response = WorkflowResponse(
                request_id=request.request_id,
                status="completed",
                result=result,
                job_id=result.get("job_id"),
            )
            await self._send_message(response)

        except Exception as e:
            self.log.error(f"Error executing workflow: {e}")
            # Send error response
            response = WorkflowResponse(
                request_id=request.request_id,
                status="failed",
                error=str(e),
            )
            await self._send_message(response)

    async def _handle_command_request(self, data: dict[str, Any]):
        """Handle command request from gateway (MCP-like commands)."""
        try:
            request = CommandRequest.model_validate(data)
            self.log.info(f"Received command: {request.command}")

            # Send acknowledgment
            ack = AckMessage(
                request_id=request.request_id,
                message=f"Executing command: {request.command}",
            )
            await self._send_message(ack)

            # Execute command in background
            _ = asyncio.create_task(self._execute_command(request))  # noqa: RUF006

        except Exception as e:
            self.log.error(f"Error handling command request: {e}")
            error = ErrorMessage(error=str(e))
            await self._send_message(error)

    async def _execute_command(self, request: CommandRequest):
        """Execute a command and send response."""
        try:
            # Map command to tool function
            result = await self._dispatch_command(request.command, request.args)

            # Send success response
            response = CommandResponse(
                request_id=request.request_id,
                status="success",
                result=result,
            )
            await self._send_message(response)

        except Exception as e:
            self.log.error(f"Error executing command: {e}")
            # Send error response
            response = CommandResponse(
                request_id=request.request_id,
                status="error",
                error=str(e),
            )
            await self._send_message(response)

    async def _dispatch_command(
        self, command: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Dispatch command to appropriate tool."""
        # Workflow commands
        if command == "list_workflows":
            return await WorkflowTools.list_workflows(**args)
        elif command == "get_workflow":
            return await WorkflowTools.get_workflow(**args)
        elif command == "run_workflow":
            return await WorkflowTools.run_workflow_tool(**args)
        elif command == "validate_workflow":
            return await WorkflowTools.validate_workflow(**args)

        # Job commands
        elif command == "list_jobs":
            return await JobTools.list_jobs(**args)
        elif command == "get_job":
            return await JobTools.get_job(**args)
        elif command == "get_job_logs":
            return await JobTools.get_job_logs(**args)
        elif command == "start_background_job":
            return await JobTools.start_background_job(**args)

        # Asset commands
        elif command == "list_assets":
            return await AssetTools.list_assets(**args)
        elif command == "get_asset":
            return await AssetTools.get_asset(**args)

        # Node commands
        elif command == "list_nodes":
            return await NodeTools.list_nodes(**args)
        elif command == "search_nodes":
            return await NodeTools.search_nodes(**args)
        elif command == "get_node_info":
            return await NodeTools.get_node_info(**args)

        # Model commands
        elif command == "list_models":
            return await ModelTools.list_models(**args)

        # Collection commands
        elif command == "list_collections":
            return await CollectionTools.list_collections(**args)
        elif command == "get_collection":
            return await CollectionTools.get_collection(**args)

        else:
            raise ValueError(f"Unknown command: {command}")

    async def _handle_ack(self, data: dict[str, Any]):
        """Handle acknowledgment message."""
        ack = AckMessage.model_validate(data)
        self.log.debug(f"Received ack for request: {ack.request_id}")

    async def _handle_error(self, data: dict[str, Any]):
        """Handle error message."""
        error = ErrorMessage.model_validate(data)
        self.log.error(f"Received error from gateway: {error.error}")
