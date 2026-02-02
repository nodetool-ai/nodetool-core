"""
Controller for persistent sub-graph execution.

Manages the lifecycle of a sub-graph that can receive multiple input injections
and return results without full workflow restart.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from nodetool.config.logging_config import get_logger
from nodetool.workflows.graph import Graph
from nodetool.workflows.processing_context import AssetOutputMode, ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.types import JobUpdate, ToolResultUpdate
from nodetool.workflows.workflow_runner import WorkflowRunner

if TYPE_CHECKING:
    from nodetool.types.api_graph import Edge
    from nodetool.types.api_graph import Graph as ApiGraph

log = get_logger(__name__)


class SubGraphController:
    """
    Controls a persistent sub-graph execution that stays alive across multiple invocations.

    Lifecycle:
    1. start() - Initialize the runner and start background execution
    2. inject_and_wait() - Inject inputs and wait for next result (called per tool invocation)
    3. stop() - Gracefully shut down the sub-graph

    Thread Safety:
        - inject_and_wait() is NOT thread-safe; calls must be serialized
        - stop() can be called from any context

    Attributes:
        graph: The API graph definition for the sub-graph
        input_node_ids: Mapping of input handle -> node ID for input injection
        runner: The persistent WorkflowRunner instance
        context: The persistent ProcessingContext
        result_queue: Queue for receiving ToolResultUpdate messages
    """

    def __init__(
        self,
        api_graph: ApiGraph,
        input_edges: list[Edge],
        parent_context: ProcessingContext,
    ):
        """
        Initialize the controller.

        Args:
            api_graph: The API graph to execute
            input_edges: Edges from external inputs to graph nodes (defines injection points)
            parent_context: Parent context for inheriting user_id, auth_token, device, etc.
        """
        self.api_graph = api_graph
        self.input_edges = input_edges
        self.parent_context = parent_context

        # Persistent state (initialized in start())
        self._runner: WorkflowRunner | None = None
        self._context: ProcessingContext | None = None
        self._result_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._background_task: asyncio.Task[None] | None = None
        self._started = False
        self._stopped = False

        # Build input injection map: handle_name -> target_node_id
        self._input_map: dict[str, str] = {
            edge.targetHandle: edge.target for edge in input_edges
        }

    @property
    def is_running(self) -> bool:
        """Check if the sub-graph is currently running."""
        return (
            self._started
            and not self._stopped
            and self._background_task is not None
            and not self._background_task.done()
        )

    async def start(self) -> None:
        """
        Start the sub-graph execution in the background.

        This initializes the runner, creates actors for all nodes, and starts
        the background task that collects results. The sub-graph will continue
        running until stop() is called.

        Raises:
            RuntimeError: If already started or stopped
        """
        if self._stopped:
            raise RuntimeError("SubGraphController already stopped; create a new instance")
        if self._started:
            raise RuntimeError("SubGraphController already started")

        self._started = True

        # Create persistent context with isolated message queue
        import queue as _queue

        isolated_queue: _queue.Queue[Any] = _queue.Queue()

        self._context = ProcessingContext(
            user_id=self.parent_context.user_id,
            auth_token=self.parent_context.auth_token,
            graph=Graph.from_dict(self.api_graph.model_dump()),
            message_queue=isolated_queue,
            device=self.parent_context.device,
            workspace_dir=self.parent_context.workspace_dir,
            asset_output_mode=getattr(
                self.parent_context, "asset_output_mode", AssetOutputMode.TEMP_URL
            ),
        )

        # Create persistent runner
        self._runner = WorkflowRunner(
            job_id=uuid4().hex,
            disable_caching=True,
        )

        # Start background collection task
        self._background_task = asyncio.create_task(
            self._run_and_collect(),
            name=f"subgraph-{self._runner.job_id}",
        )

        log.info(
            "SubGraphController started: job_id=%s, input_handles=%s",
            self._runner.job_id,
            list(self._input_map.keys()),
        )

    async def _run_and_collect(self) -> None:
        """
        Background task that runs the workflow and collects ToolResultUpdate messages.

        This task runs indefinitely (or until cancelled/error) and forwards
        ToolResultUpdate messages to the result queue.
        """
        assert self._context is not None
        assert self._runner is not None

        req = RunJobRequest(
            user_id=self._context.user_id,
            auth_token=self._context.auth_token,
            graph=self.api_graph,
        )

        try:
            async for msg in run_workflow(
                request=req,
                runner=self._runner,
                context=self._context,
                use_thread=False,  # Run in same thread to share event loop
                send_job_updates=False,
                initialize_graph=True,
                validate_graph=False,
            ):
                # Forward non-job messages to parent context
                if not isinstance(msg, JobUpdate):
                    self.parent_context.post_message(msg)

                # Capture tool results
                if isinstance(msg, ToolResultUpdate):
                    if msg.result is not None:
                        await self._result_queue.put(msg.result)
                        log.debug(
                            "SubGraphController captured result: %s",
                            list(msg.result.keys())
                            if isinstance(msg.result, dict)
                            else type(msg.result),
                        )
        except asyncio.CancelledError:
            log.info("SubGraphController background task cancelled")
            raise
        except Exception as e:
            log.error("SubGraphController background task error: %s", e)
            # Put error in queue so inject_and_wait can handle it
            await self._result_queue.put({"__error__": str(e)})
            raise

    async def inject_and_wait(
        self,
        params: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """
        Inject input parameters and wait for the next result.

        This method:
        1. Injects the provided params into the appropriate node inboxes
        2. Waits for a ToolResultUpdate to arrive in the result queue
        3. Returns the result

        Args:
            params: Input parameters matching the graph's input schema
            timeout: Optional timeout in seconds; None means wait indefinitely

        Returns:
            The result dictionary from ToolResultNode

        Raises:
            RuntimeError: If not started or already stopped
            asyncio.TimeoutError: If timeout is reached before result
            Exception: If the sub-graph encountered an error
        """
        if not self._started:
            raise RuntimeError("SubGraphController not started; call start() first")
        if self._stopped:
            raise RuntimeError("SubGraphController already stopped")
        if self._runner is None:
            raise RuntimeError("Runner not initialized")

        # Inject inputs into the appropriate node inboxes
        for handle, value in params.items():
            target_node_id = self._input_map.get(handle)
            if target_node_id is None:
                log.warning("No input mapping for handle: %s", handle)
                continue

            inbox = self._runner.node_inboxes.get(target_node_id)
            if inbox is None:
                log.warning("No inbox found for node: %s", target_node_id)
                continue

            await inbox.put(handle, value)
            log.debug(
                "Injected input: handle=%s, node=%s, value_type=%s",
                handle,
                target_node_id,
                type(value).__name__,
            )

        # Wait for result
        try:
            if timeout is not None:
                result = await asyncio.wait_for(
                    self._result_queue.get(),
                    timeout=timeout,
                )
            else:
                result = await self._result_queue.get()
        except TimeoutError:
            log.warning("inject_and_wait timed out after %s seconds", timeout)
            raise

        # Check for error
        if isinstance(result, dict) and "__error__" in result:
            raise Exception(result["__error__"])

        return result

    async def stop(self) -> None:
        """
        Stop the sub-graph execution gracefully.

        This cancels the background task and cleans up resources.
        Safe to call multiple times.
        """
        if self._stopped:
            return

        self._stopped = True

        # Cancel background task
        if self._background_task is not None:
            self._background_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._background_task
            self._background_task = None

        # Clean up runner
        if self._runner is not None:
            # Drain any pending messages
            if self._context is not None and self._context.graph is not None:
                self._runner.drain_active_edges(self._context, self._context.graph)
            self._runner = None

        self._context = None

        log.info("SubGraphController stopped")

    async def __aenter__(self) -> SubGraphController:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.stop()
