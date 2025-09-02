"""Per-node async actor driving node execution.

This module implements the actor responsible for running a single node in an
async task. It integrates with ``NodeInbox`` for inputs and the
``WorkflowRunner`` for routing outputs.

Overview:
  - One actor per node: owns the node's lifecycle while running
  - Message-driven I/O: pulls inputs from ``NodeInbox`` and forwards outputs via
    ``WorkflowRunner.send_messages``
  - Streaming-first design: calls ``BaseNode.run``; streaming nodes yield items,
    non-streaming nodes are invoked once unless the actor fans out per arrival
  - Failure isolation: reports errors via ``NodeUpdate(status="error")`` and
    marks downstream EOS

Lifecycle:
  1. Gather initial inputs and assign properties (only valid names)
  2. Call ``pre_process`` and send a "running" update
  3. Drive processing: streaming via ``node.run`` with input/output wrappers; or
     non-streaming via ``runner.process_node_with_inputs`` per arrival (fan-out)
  4. On completion or error, mark downstream EOS

Notes:
  - Ordering: preserves per-handle order; no cross-handle guarantees beyond
    arrival order
  - Cancellation: runs in an ``asyncio.Task`` and cooperates with cancellation
  - Dynamic outputs: supported; undeclared yields raise errors early
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator

from nodetool.workflows.base_node import BaseNode, OutputNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.inbox import NodeInbox
from nodetool.workflows.types import NodeUpdate
from nodetool.workflows.workflow_runner import WorkflowRunner
from nodetool.workflows.io import NodeInputs, NodeOutputs


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
        self.logger = logging.getLogger(__name__)
        # Log level is controlled by env (DEBUG/NODETOOL_LOG_LEVEL)

    def _inbound_handles(self) -> set[str]:
        """Return the set of inbound input handles for this node."""
        return {
            e.targetHandle
            for e in self.context.graph.edges
            if e.target == self.node._id
        }

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

    async def _gather_initial_inputs(
        self, handles: set[str] | None = None
    ) -> dict[str, Any]:
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
        for h, val in zip(tasks.keys(), results):
            if val is not None:
                values[h] = val
        return values

    async def _mark_downstream_eos(self) -> None:
        """Mark end-of-stream on all outbound handles to unblock consumers."""
        for edge in self._outbound_edges():
            inbox = self.runner.node_inboxes.get(edge.target)
            if inbox is not None:
                inbox.mark_source_done(edge.targetHandle)
            # Notify listeners that this edge has been drained (EOS signaled)
            try:
                from .types import EdgeUpdate

                self.context.post_message(
                    EdgeUpdate(edge_id=edge.id or "", status="drained")
                )
            except Exception:
                # Best-effort notification; never block EOS on update errors
                pass

    async def _run_streaming(self) -> None:
        """Run a node that exposes streaming outputs.

        Manages initial property assignment, updates, and drives ``node.run``
        with ``NodeInputs`` and ``NodeOutputs`` wrappers.
        """
        node = self.node
        ctx = self.context
        self.logger.debug(f"Running streaming node {node.get_title()} ({node._id})")

        # If this node is only fed by non-routable upstreams, skip running
        if self._only_nonroutable_upstreams():
            await self._mark_downstream_eos()
            return

        # Assign initial properties as needed before starting the generator
        initial_inputs: dict[str, Any] = {}
        if node.__class__.is_streaming_input():
            # Do not pre-gather for streaming-input nodes; they consume exclusively via inbox.
            initial_inputs = {}
        else:
            # For non-streaming-input nodes, gather one per inbound handle
            initial_inputs = await self._gather_initial_inputs()

        self.logger.debug(f"Initial inputs: {initial_inputs}")

        for name, value in initial_inputs.items():
            node.assign_property(name, value)

        # Pre-process and send running update
        await node.pre_process(ctx)
        # Only include valid property names in the running update (regression expectation)
        safe_properties = [
            name
            for name in initial_inputs.keys()
            if node.find_property(name) is not None
        ]
        await node.send_update(ctx, "running", properties=safe_properties)

        # Drive the unified run() method (bridges to gen_process by default)
        try:
            await node.run(
                ctx, NodeInputs(self.inbox), NodeOutputs(self.runner, node, ctx)
            )
            await node.send_update(ctx, "completed", result={"status": "completed"})
        finally:
            await self._mark_downstream_eos()

    async def _run_non_streaming(self) -> None:
        """Run a non-streaming node with fan-out per arriving input message."""
        node = self.node
        ctx = self.context

        self.logger.debug(f"Running non-streaming node {node.get_title()} ({node._id})")
        # If this node is only fed by non-routable upstreams, skip running
        if self._only_nonroutable_upstreams():
            await self._mark_downstream_eos()
            return
        # Fanout for any inbound messages across handles: run once per arriving item.
        handles = self._effective_inbound_handles()
        if not handles:
            # No inbound inputs – single run
            await self.runner.process_node_with_inputs(ctx, node, {})
        else:
            # Determine per-handle upstream streaming characteristics
            upstream_all_non_streaming: dict[str, bool] = {}
            for e in self.context.graph.edges:
                if e.target != node._id:
                    continue
                handle = e.targetHandle
                src = self.context.graph.find_node(e.source)
                if src is None:
                    continue
                prev = upstream_all_non_streaming.get(handle, True)
                upstream_all_non_streaming[handle] = prev and (
                    not src.is_streaming_output()
                )

            sync_mode = getattr(node, "get_sync_mode", None)
            sync_mode_value = "on_any"
            if callable(sync_mode):
                try:
                    sync_mode_value = sync_mode()
                except Exception:
                    sync_mode_value = "on_any"

            if sync_mode_value == "zip_all":
                # Align across all inbound handles: emit only when we have at least
                # one value buffered for each handle; consume one from each per call.
                from collections import deque

                buffers: dict[str, deque[Any]] = {h: deque() for h in handles}

                async for handle, item in self.inbox.iter_any():
                    if handle not in buffers:
                        continue
                    buffers[handle].append(item)
                    # While a full set is available, pop and run
                    while all(len(q) > 0 for q in buffers.values()):
                        batch = {h: buffers[h].popleft() for h in buffers.keys()}
                        await self.runner.process_node_with_inputs(ctx, node, batch)
                        # For non-streaming upstreams we still mark one as done to
                        # keep tests deterministic in absence of EOS senders
                        for h in list(batch.keys()):
                            if upstream_all_non_streaming.get(h, False):
                                self.inbox.mark_source_done(h)
            else:
                # Default behavior: fire on any arrival with latest values from others
                current: dict[str, Any] = {}
                async for handle, item in self.inbox.iter_any():
                    current[handle] = item
                    await self.runner.process_node_with_inputs(ctx, node, dict(current))
                    # If all upstreams for this handle are non-streaming producers,
                    # they produce at most one item each. Mark one source as done to
                    # prevent indefinite waiting for EOS in isolated actor tests.
                    if upstream_all_non_streaming.get(handle, False):
                        self.inbox.mark_source_done(handle)
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

        # Upstream completed – mark downstream EOS
        await self._mark_downstream_eos()

    async def run(self) -> None:
        """Entry point: choose streaming vs non-streaming path and execute."""
        node = self.node
        ctx = self.context
        try:
            # Select the appropriate runner coroutine for this node
            if isinstance(node, OutputNode):
                run_coro = self._run_output_node()
            elif node.is_streaming_output() or node.__class__.is_streaming_input():
                run_coro = self._run_streaming()
            else:
                run_coro = self._run_non_streaming()

            timeout = node.get_timeout_seconds() or 0.0
            if timeout > 0:
                try:
                    await asyncio.wait_for(run_coro, timeout=timeout)
                except asyncio.TimeoutError:
                    try:
                        ctx.post_message(
                            NodeUpdate(
                                node_id=node.id,
                                node_name=node.get_title(),
                                node_type=node.get_node_type(),
                                status="error",
                                error=f"Node timed out after {timeout:.2f}s",
                            )
                        )
                    finally:
                        await self._mark_downstream_eos()
                    raise
            else:
                await run_coro
        except asyncio.CancelledError:
            # Ensure downstream EOS is marked on cooperative cancellation
            try:
                await self._mark_downstream_eos()
            finally:
                raise
        except Exception as e:
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
