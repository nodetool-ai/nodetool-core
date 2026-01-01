"""Per-node async actor driving node execution.

This module implements the actor responsible for running a single node in an
async task. It integrates with ``NodeInbox`` for inputs and the
``WorkflowRunner`` for routing outputs.

Planned execution matrix
========================

To clarify the upcoming refactor, we document how nodes are expected to behave
depending on their streaming declarations:

=====================  =====================  ======================================
 streaming_input flag    streaming_output flag   Actor responsibility
=====================  =====================  ======================================
 ``False``               ``False``              Current buffered path: actor gathers
                                               up to one value per handle (subject
                                               to ``sync_mode``) and calls
                                               ``process()`` once.
 ``False``               ``True``               Planned change: actor still handles
                                               batching (respecting ``sync_mode``)
                                               but will invoke ``gen_process``
                                               **per batch** so streaming outputs
                                               pair with batched inputs.
 ``True``                ``False``              Discouraged pattern (node would have
                                               to drain inbox but only emit once).
 ``True``                ``True``               Existing streaming behaviour: actor
                                               calls ``node.run`` once and the node
                                               drains the inbox on its own using
                                               ``iter_input``/``iter_any``.
=====================  =====================  ======================================

Only nodes that explicitly opt into ``is_streaming_input()`` will receive a live
inbox; everyone else relies on the actor for alignment. This table serves as the
rationale for the changes that will follow.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from nodetool.config.logging_config import get_logger
from nodetool.ml.core.model_manager import ModelManager
from nodetool.models.run_node_state import RunNodeState
from nodetool.workflows.io import NodeInputs, NodeOutputs
from nodetool.workflows.torch_support import is_cuda_available
from nodetool.workflows.types import EdgeUpdate, NodeUpdate

if TYPE_CHECKING:
    from nodetool.workflows.base_node import BaseNode
    from nodetool.workflows.inbox import NodeInbox
    from nodetool.workflows.processing_context import ProcessingContext
    from nodetool.workflows.workflow_runner import WorkflowRunner


class NodeActor:
    """Drives a single node to completion.

    Orchestrates node execution with unified I/O wrappers and runner hooks.

    Args:
        runner: The active ``WorkflowRunner`` instance.
        node: The node instance to execute.
        context: The processing context associated with this run.
        inbox: The per-node inbox providing input values.
    """

    def __init__(
        self,
        runner: WorkflowRunner,
        node: BaseNode,
        context: ProcessingContext,
        inbox: NodeInbox,
    ) -> None:
        self.runner = runner
        self.node = node
        self.context = context
        self.inbox = inbox
        self._task: asyncio.Task | None = None
        self.logger = get_logger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def _filter_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """Filter out chunk data from the result since chunks are streamed separately.

        Args:
            result: The collected outputs dictionary.

        Returns:
            A new dictionary with 'chunk' key removed.
        """
        return {k: v for k, v in result.items() if k != "chunk"}

    def _inbound_handles(self) -> set[str]:
        """Return the set of inbound input handles for this node."""
        return {e.targetHandle for e in self.context.graph.edges if e.target == self.node._id}

    def _outbound_edges(self):
        """Return edges originating from this node (outbound)."""
        return [e for e in self.context.graph.edges if e.source == self.node._id]

    def _is_nonroutable_edge(self, edge) -> bool:
        """Return True if the upstream source suppresses routing for this output.

        Args:
            edge: The edge to inspect.

        Returns:
            True if the edge's source node reports ``should_route_output`` False
            for the given source handle; otherwise False.
        """
        try:
            src = self.context.graph.find_node(edge.source)
            if src is None:
                return False
            return not src.should_route_output(edge.sourceHandle)
        except Exception:
            return False

    def _effective_inbound_handles(self) -> set[str]:
        """Return inbound handles after filtering non-routable upstream-only handles."""
        handles: set[str] = set()
        for e in self.context.graph.edges:
            if e.target != self.node._id:
                continue
            # Collect all edges feeding this handle
            same_handle_edges = [
                ee
                for ee in self.context.graph.edges
                if ee.target == self.node._id and ee.targetHandle == e.targetHandle
            ]
            # Keep the handle if at least one upstream is routable
            if not all(self._is_nonroutable_edge(ee) for ee in same_handle_edges):
                handles.add(e.targetHandle)
        return handles

    def _only_nonroutable_upstreams(self) -> bool:
        """Return True if there are inbound edges and none are effectively routable."""
        # If there are no inbound edges, there's nothing to suppress
        has_inbound = any(e.target == self.node._id for e in self.context.graph.edges)
        if not has_inbound:
            return False
        # If no effective handles remain, all upstreams are non-routable
        return len(self._effective_inbound_handles()) == 0

    def _mark_inbound_edges_drained(self, handles: set[str] | list[str]) -> None:
        """Post drained updates for inbound edges targeting the provided handles."""
        if not handles:
            return
        handle_set = set(handles)
        for edge in self.context.graph.edges:
            if edge.target != self.node._id:
                continue
            if edge.targetHandle not in handle_set:
                continue
            self.context.post_message(
                EdgeUpdate(workflow_id=self.context.workflow_id, edge_id=edge.id or "", status="drained")
            )

    async def _gather_initial_inputs(self, handles: set[str] | None = None) -> dict[str, Any]:
        """Wait for exactly one value per specified inbound handle and return a map.

        Args:
            handles: Optional subset of inbound handles to gather from. If None,
                uses all effective inbound handles.

        Returns:
            Dict mapping handle name to its first available item.
        """
        # If handles are not specified, use the effective set that ignores Agent dynamic-only inputs
        handles = handles if handles is not None else self._effective_inbound_handles()
        if not handles:
            return {}
        if self.inbox is None:
            # Should not happen: inboxes are attached in runner
            return {}

        async def first_item(h: str):
            async for item in self.inbox.iter_input(h):
                return item
            # If EOS before any item, return a sentinel None
            return None

        tasks = {h: asyncio.create_task(first_item(h)) for h in handles}
        results = await asyncio.gather(*tasks.values())
        values: dict[str, Any] = {}
        for h, val in zip(tasks.keys(), results, strict=False):
            if val is not None:
                values[h] = val
        return values

    async def _mark_downstream_eos(self) -> None:
        """Mark end-of-stream on all inbound and outbound edges."""
        # Mark inbound edges as drained (we've consumed all input)
        self._mark_inbound_edges_drained(self._inbound_handles())
        
        # Mark outbound edges as drained and unblock downstream consumers
        for edge in self._outbound_edges():
            inbox = self.runner.node_inboxes.get(edge.target)
            if inbox is not None:
                inbox.mark_source_done(edge.targetHandle)
            # Notify listeners that this edge has been drained (EOS signaled)
            self.context.post_message(
                EdgeUpdate(workflow_id=self.context.workflow_id, edge_id=edge.id or "", status="drained"),
            )

    async def process_node_with_inputs(
        self,
        inputs: dict[str, Any],
    ) -> None:
        """Process a non-streaming node instance with resolved inputs.

        Mirrors the lifecycle previously hosted on ``WorkflowRunner``:

        1. Assign inputs to node properties.
        2. ``pre_process`` the node.
        3. Check and leverage cache when enabled.
        4. Execute the node (with GPU coordination when required).
        5. Cache results when appropriate.
        6. Emit completion updates and route outputs downstream.
        """

        context = self.context
        node = self.node

        self.logger.debug(
            "process_node_with_inputs for %s (%s) with inputs: %s",
            node.get_title(),
            node._id,
            list(inputs.keys()),
        )

        for name, value in inputs.items():
            try:
                error = node.assign_property(name, value)
                if error:
                    self.logger.error(
                        "Error assigning property %s to node %s: %s",
                        name,
                        node.id,
                        error,
                    )
                else:
                    # Notify frontend of property change
                    await node.send_update(context, "running", properties=[name])
            except Exception as exc:
                self.logger.error("Error assigning property %s to node %s", name, node.id)
                raise ValueError(f"Error assigning property {name}: {exc}") from exc

        await node.pre_process(context)

        cached_result: dict[str, Any] | None = None
        if node.is_cacheable() and not self.runner.disable_caching:
            cached_result = context.get_cached_result(node)

        if cached_result is not None:
            self.logger.info(
                "Using cached result for node: %s (%s)",
                node.get_title(),
                node._id,
            )
            result = cached_result
        else:
            requires_gpu = node.requires_gpu()
            driven_by_stream = context.graph.has_streaming_upstream(node._id)

            if requires_gpu and self.runner.device == "cpu":
                error_msg = f"Node {node.get_title()} ({node._id}) requires a GPU, but no GPU is available."
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)

            await node.send_update(context, "running", result=None)

            inbox = self.runner.node_inboxes.get(node._id)
            outputs_collector = NodeOutputs(self.runner, node, context, capture_only=True)
            node_inputs = NodeInputs(inbox) if inbox is not None else None

            if requires_gpu and self.runner.device != "cpu":
                from nodetool.workflows.workflow_runner import (
                    acquire_gpu_lock,
                    release_gpu_lock,
                )

                await acquire_gpu_lock(node, context)
                try:
                    if is_cuda_available():
                        ModelManager.free_vram_if_needed(
                            reason=f"Preloading model for node {node._id}",
                            required_free_gb=1.0,
                        )
                    self.runner.log_vram_usage(f"Node {node.get_title()} ({node._id}) VRAM before GPU processing")
                    await node.preload_model(context)
                    self.runner.log_vram_usage(f"Node {node.get_title()} ({node._id}) VRAM after preload_model")

                    await node.run(context, node_inputs, outputs_collector)  # type: ignore[arg-type]
                    self.runner.log_vram_usage(f"Node {node.get_title()} ({node._id}) VRAM after run completion")
                finally:
                    release_gpu_lock()
            else:
                await node.preload_model(context)
                await node.run(context, node_inputs, outputs_collector)  # type: ignore[arg-type]

            result = outputs_collector.collected()

            if node.is_cacheable() and not self.runner.disable_caching and not driven_by_stream:
                self.logger.debug(
                    "Caching result for node: %s (%s)",
                    node.get_title(),
                    node._id,
                )
                context.cache_result(node, result)

        await node.send_update(context, "completed", result=result)
        await self.runner.send_messages(node, result, context)
        # Note: drained updates are sent at end-of-stream in _mark_downstream_eos, not here

    async def process_streaming_node_with_inputs(
        self,
        inputs: dict[str, Any],
    ) -> None:
        """Process a streaming-output node for one aligned batch of inputs.

        This mirrors ``process_node_with_inputs`` but keeps ``NodeOutputs`` live so
        values are routed downstream as they are produced. It assumes the node does
        **not** consume the inbox itself (``is_streaming_input()`` is False).
        """

        context = self.context
        node = self.node

        self.logger.debug(
            "process_streaming_node_with_inputs for %s (%s) with inputs: %s",
            node.get_title(),
            node._id,
            list(inputs.keys()),
        )

        for name, value in inputs.items():
            try:
                error = node.assign_property(name, value)
                if error:
                    self.logger.error(
                        "Error assigning property %s to node %s: %s",
                        name,
                        node.id,
                        error,
                    )
                else:
                    # Notify frontend of property change (if method exists)
                    if hasattr(self.runner, "send_property_update"):
                        await self.runner.send_property_update(node, context, name)
            except Exception as exc:
                self.logger.error("Error assigning property %s to node %s", name, node.id)
                raise ValueError(f"Error assigning property {name}: {exc}") from exc

        await node.pre_process(context)

        safe_properties = [name for name in inputs if node.find_property(name) is not None]

        await node.send_update(context, "running", properties=safe_properties)

        requires_gpu = node.requires_gpu()
        node_inputs = NodeInputs(self.inbox)
        outputs = NodeOutputs(self.runner, node, context)

        completed_successfully = False
        try:
            if requires_gpu and self.runner.device == "cpu":
                error_msg = f"Node {node.get_title()} ({node._id}) requires a GPU, but no GPU is available."
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)

            if requires_gpu and self.runner.device != "cpu":
                from nodetool.workflows.workflow_runner import (
                    acquire_gpu_lock,
                    release_gpu_lock,
                )

                await acquire_gpu_lock(node, context)

                try:
                    if is_cuda_available():
                        ModelManager.free_vram_if_needed(
                            reason=f"Preloading model for node {node._id}",
                            required_free_gb=1.0,
                        )
                    self.runner.log_vram_usage(f"Node {node.get_title()} ({node._id}) VRAM before GPU processing")

                    await node.preload_model(context)

                    self.runner.log_vram_usage(f"Node {node.get_title()} ({node._id}) VRAM after preload_model")

                    await node.run(context, node_inputs, outputs)  # type: ignore[arg-type]

                    self.runner.log_vram_usage(f"Node {node.get_title()} ({node._id}) VRAM after run completion")
                except Exception as e:
                    self.logger.error(f"Error running node {node.get_title()} ({node._id}): {e}")
                    context.post_message(
                        NodeUpdate(
                            node_id=node.id,
                            node_name=node.get_title(),
                            node_type=node.get_node_type(),
                            status="error",
                            error=str(e),
                        )
                    )
                    raise
                finally:
                    release_gpu_lock()
            else:
                await node.preload_model(context)
                await node.run(context, node_inputs, outputs)  # type: ignore[arg-type]

            completed_successfully = True
        finally:
            if completed_successfully:
                # Send the actual collected results, filtering out chunk data
                await node.send_update(context, "completed", result=self._filter_result(outputs.collected()))
            # Note: drained updates are sent at end-of-stream in _mark_downstream_eos, not here

    async def _run_buffered_node(self) -> None:
        """Legacy buffered node execution (no streaming output)."""
        await self._run_non_streaming_internal(None)

    async def _run_streaming_output_batched_node(self) -> None:
        """Streaming-output node without streaming-input: batch inputs respecting sync mode."""
        await self._run_non_streaming_internal(
            processor_override=self.process_streaming_node_with_inputs,
        )

    async def _run_non_streaming_internal(
        self,
        processor_override: Callable[[dict[str, Any]], Awaitable[None]] | None,
    ) -> None:
        """Run a non-streaming-input node with fan-out per arriving input message."""
        node = self.node

        self.logger.debug(f"Running batched node {node.get_title()} ({node._id})")
        if self._only_nonroutable_upstreams():
            await self._mark_downstream_eos()
            return

        handles = self._effective_inbound_handles()
        streaming_output = node.__class__.is_streaming_output() and not node.__class__.is_streaming_input()
        processor = processor_override or (
            self.process_streaming_node_with_inputs if streaming_output else self.process_node_with_inputs
        )

        if not handles:
            await processor({})
        else:
            handle_streaming: dict[str, bool] = {}
            for handle in handles:
                edge = next(
                    (e for e in self.context.graph.edges if e.target == node._id and e.targetHandle == handle),
                    None,
                )
                handle_streaming[handle] = self.runner.edge_streams(edge) if edge is not None else False

            sync_mode = node.get_sync_mode()

            if sync_mode == "zip_all" and any(handle_streaming.values()):
                from collections import deque

                buffers: dict[str, deque[Any]] = {h: deque() for h in handles}
                sticky_values: dict[str, Any] = {}
                is_sticky: dict[str, bool] = {h: not handle_streaming.get(h, False) for h in handles}
                seen_counts: dict[str, int] = dict.fromkeys(handles, 0)

                def ready_to_zip() -> bool:
                    for handle in handles:
                        if is_sticky.get(handle, False):
                            if handle not in sticky_values and not buffers[handle]:
                                return False
                        elif not buffers[handle]:
                            return False
                    return True

                async for handle, item in self.inbox.iter_any():
                    if handle not in buffers:
                        continue

                    buffers[handle].append(item)
                    seen_counts[handle] = seen_counts.get(handle, 0) + 1

                    if is_sticky.get(handle, False):
                        sticky_values[handle] = item

                    while ready_to_zip():
                        batch: dict[str, Any] = {}
                        for h in handles:
                            if is_sticky.get(h, False):
                                if buffers[h]:
                                    sticky_values[h] = buffers[h].popleft()
                                batch[h] = sticky_values[h]
                                # Restore the sticky value for future batches
                                if sticky_values[h] is not None:
                                    buffers[h].appendleft(sticky_values[h])
                            else:
                                batch[h] = buffers[h].popleft()

                        await processor(dict(batch))

                        for h in handles:
                            if is_sticky.get(h, False) and sticky_values.get(h) is None:
                                sticky_values.pop(h, None)
            else:
                current: dict[str, Any] = {}
                pending_handles: set[str] = set(handles)
                initial_fired: bool = False

                async for handle, item in self.inbox.iter_any():
                    if handle not in handles:
                        continue
                    current[handle] = item
                    if not initial_fired:
                        pending_handles.discard(handle)
                        if pending_handles:
                            continue
                        await processor(dict(current))
                        initial_fired = True
                    else:
                        await processor(dict(current))

                    if not handle_streaming.get(handle, False):
                        self.inbox.mark_source_done(handle)
        await node.handle_eos()
        await self._mark_downstream_eos()

    async def _run_output_node(self) -> None:
        """Run an OutputNode by forwarding each arriving input to runner outputs.

        Output nodes should capture every incoming value. We iterate over any
        arriving inputs across all handles in arrival order and call the
        runner's dedicated `process_output_node` for each item.
        """
        node = self.node
        ctx = self.context

        # If fed only by non-routable upstreams, nothing to capture
        if self._only_nonroutable_upstreams():
            await self._mark_downstream_eos()
            return

        # Drain inputs in arrival order and capture via runner
        async for handle, item in self.inbox.iter_any():
            await self.runner.process_output_node(ctx, node, {handle: item})  # type: ignore[arg-type]
            # Note: drained updates are sent at end-of-stream in _mark_downstream_eos, not here

        # Upstream completed - mark downstream EOS
        await self._mark_downstream_eos()

    async def _run_streaming_input_node(self) -> None:
        node = self.node
        ctx = self.context

        # Assign initial properties as needed before starting the generator (not pre-gathered)
        await node.pre_process(ctx)
        await node.send_update(ctx, "running", properties=[])
        requires_gpu = node.requires_gpu()
        # Note: drained updates are sent at end-of-stream, not per item
        # node._on_input_item callback is not needed for drained notifications

        if requires_gpu and self.runner.device == "cpu":
            error_msg = f"Node {node.get_title()} ({node._id}) requires a GPU, but no GPU is available."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
            if requires_gpu and self.runner.device != "cpu":
                from nodetool.workflows.workflow_runner import (
                    acquire_gpu_lock,
                    release_gpu_lock,
                )

                await acquire_gpu_lock(node, ctx)
                try:
                    if is_cuda_available():
                        ModelManager.free_vram_if_needed(
                            reason=f"Preloading model for node {node._id}",
                            required_free_gb=1.0,
                        )
                    self.runner.log_vram_usage(f"Node {node.get_title()} ({node._id}) VRAM before GPU processing")
                    await node.preload_model(ctx)
                    self.runner.log_vram_usage(f"Node {node.get_title()} ({node._id}) VRAM after preload_model")

                    outputs = NodeOutputs(self.runner, node, ctx)
                    await node.run(ctx, NodeInputs(self.inbox), outputs)
                    self.runner.log_vram_usage(f"Node {node.get_title()} ({node._id}) VRAM after run completion")
                except Exception as e:
                    self.logger.error(f"Error running node {node.get_title()} ({node._id}): {e}")
                    ctx.post_message(
                        NodeUpdate(
                            node_id=node.id,
                            node_name=node.get_title(),
                            node_type=node.get_node_type(),
                            status="error",
                            error=str(e),
                        )
                    )
                    raise e
                finally:
                    release_gpu_lock()
            else:
                await node.preload_model(ctx)
                outputs = NodeOutputs(self.runner, node, ctx)
                await node.run(ctx, NodeInputs(self.inbox), outputs)

            # Send the actual collected results, filtering out chunk data
            await node.send_update(ctx, "completed", result=self._filter_result(outputs.collected()))
        finally:
            await node.handle_eos()
        await self._mark_downstream_eos()

    async def run(self) -> None:
        """Entry point: choose streaming vs non-streaming path and execute."""
        node = self.node
        ctx = self.context
        self.logger.info(
            "Executing node: %s (%s) [%s]",
            node.get_title(),
            node._id,
            node.get_node_type(),
        )

        # Record node_id for state updates via StateManager
        node_id = node._id

        # Queue state update: scheduled (non-blocking via StateManager)
        if self.runner.state_manager:
            try:
                await self.runner.state_manager.update_node_state(
                    node_id=node_id,
                    status="scheduled",
                    attempt=1,  # TODO: track attempts properly
                    scheduled_at=datetime.now(),
                )
                self.logger.debug(f"Queued state update (scheduled) for node {node_id}")
            except Exception as e:
                self.logger.error(f"Failed to queue state update: {e}")
                # Continue execution - state tracking failure shouldn't block workflow

        # Log NodeScheduled event (audit-only, non-fatal)
        if self.runner.event_logger:
            try:
                await self.runner.event_logger.log_node_scheduled(
                    node_id=node._id,
                    node_type=node.get_node_type(),
                    attempt=1,  # TODO: track attempts properly
                )
            except Exception as e:
                self.logger.warning(f"Failed to log NodeScheduled event (non-fatal): {e}")

        # Record start time for duration calculation
        start_time = asyncio.get_event_loop().time()

        try:
            # Queue state update: running (non-blocking via StateManager)
            if self.runner.state_manager:
                try:
                    await self.runner.state_manager.update_node_state(
                        node_id=node_id,
                        status="running",
                        started_at=datetime.now(),
                    )
                    self.logger.debug(f"Queued state update (running) for node {node_id}")
                except Exception as e:
                    self.logger.error(f"Failed to queue state update: {e}")

            streaming_input = node.__class__.is_streaming_input()
            streaming_output = node.__class__.is_streaming_output()

            if streaming_input:
                await self._run_streaming_input_node()
            elif streaming_output:
                await self._run_streaming_output_batched_node()
            else:
                await self._run_buffered_node()

            # Calculate duration
            duration_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

            # Queue state update: completed (non-blocking via StateManager)
            if self.runner.state_manager:
                try:
                    await self.runner.state_manager.update_node_state(
                        node_id=node_id,
                        status="completed",
                        completed_at=datetime.now(),
                        outputs_json={},  # TODO: track actual outputs
                    )
                    self.logger.debug(f"Queued state update (completed) for node {node_id}")
                except Exception as e:
                    self.logger.error(f"Failed to queue state update: {e}")

            # Log NodeCompleted event (audit-only, non-fatal)
            if self.runner.event_logger:
                try:
                    await self.runner.event_logger.log_node_completed(
                        node_id=node._id,
                        attempt=1,  # TODO: track attempts properly
                        outputs={},  # Outputs are tracked separately in send_messages
                        duration_ms=duration_ms,
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to log NodeCompleted event (non-fatal): {e}")

        except asyncio.CancelledError:
            self.logger.info(
                "Node execution cancelled: %s (%s) [%s]",
                node.get_title(),
                node._id,
                node.get_node_type(),
            )
            # Ensure downstream EOS is marked on cooperative cancellation
            try:
                await self._mark_downstream_eos()
            finally:
                raise
        except Exception as e:
            import traceback

            self.logger.error(traceback.format_exc())
            self.logger.error(
                "Node execution failed: %s (%s) [%s] - %s",
                node.get_title(),
                node._id,
                node.get_node_type(),
                e,
            )

            # Queue state update: failed (non-blocking via StateManager)
            if self.runner.state_manager:
                try:
                    await self.runner.state_manager.update_node_state(
                        node_id=node_id,
                        status="failed",
                        failed_at=datetime.now(),
                        last_error=str(e)[:1000],
                        retryable=False,  # TODO: determine retryability
                    )
                    self.logger.debug(f"Queued state update (failed) for node {node_id}")
                except Exception as e2:
                    self.logger.error(f"Failed to queue state update: {e2}")

            # Log NodeFailed event (audit-only, non-fatal)
            if self.runner.event_logger:
                try:
                    await self.runner.event_logger.log_node_failed(
                        node_id=node._id,
                        attempt=1,  # TODO: track attempts properly
                        error=str(e)[:1000],
                        retryable=False,  # TODO: determine retryability
                    )
                except Exception as e2:
                    self.logger.warning(f"Failed to log NodeFailed event (non-fatal): {e2}")

            # Post error update and propagate; ensure EOS downstream
            try:
                ctx.post_message(
                    NodeUpdate(
                        node_id=node.id,
                        node_name=node.get_title(),
                        node_type=node.get_node_type(),
                        status="error",
                        error=str(e)[:1000],
                    )
                )
            finally:
                await self._mark_downstream_eos()
            raise
        else:
            self.logger.info(
                "Node execution completed: %s (%s) [%s]",
                node.get_title(),
                node._id,
                node.get_node_type(),
            )
