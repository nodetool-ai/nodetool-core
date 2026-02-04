"""
Unified Node I/O wrappers used by BaseNode.run and NodeActor.

NodeInputs: convenience helpers over NodeInbox for reading inputs.
NodeOutputs: convenience helpers that validate and route outputs via WorkflowRunner.
"""

from __future__ import annotations

import inspect
from contextlib import suppress
from typing import TYPE_CHECKING, Any, AsyncIterator

from nodetool.config.logging_config import get_logger

from .types import EdgeUpdate

log = get_logger(__name__)

if TYPE_CHECKING:
    from .inbox import MessageEnvelope, NodeInbox


class NodeInputs:
    """Convenience wrapper for consuming node inputs.

    Wraps a ``NodeInbox`` to provide simple, intention-revealing methods for
    reading inputs in node implementations (e.g., inside ``BaseNode.run``).

    For backward compatibility, standard methods (first, stream, any) return
    unwrapped data. For nodes needing access to message metadata, timestamp,
    or event_id, use the _with_envelope variants (first_with_envelope,
    stream_with_envelope, any_with_envelope).
    """

    def __init__(self, inbox: NodeInbox) -> None:
        """Initialize a NodeInputs view over a NodeInbox.

        Args:
            inbox: The underlying per-node inbox used by the runner to deliver inputs.
        """
        self._inbox = inbox

    async def first(self, name: str, default: Any | None = None) -> Any | None:
        """Return the first available item for a handle, or a default.

        Args:
            name: Input handle name to read from.
            default: Value to return if the stream reaches EOS without any item.

        Returns:
            The first item (unwrapped data) from the specified handle, or ``default`` if none arrived.
        """
        async for item in self._inbox.iter_input(name):
            return item
        return default

    async def first_with_envelope(self, name: str, default: Any | None = None) -> Any | None:
        """Return the first available envelope for a handle, or a default.

        Use this when you need access to message metadata, timestamp, or event_id.

        Args:
            name: Input handle name to read from.
            default: Value to return if the stream reaches EOS without any item.

        Returns:
            The first MessageEnvelope from the specified handle, or ``default`` if none arrived.
        """
        async for envelope in self._inbox.iter_input_with_envelope(name):
            return envelope
        return default

    async def stream(self, name: str) -> AsyncIterator[Any]:
        """Iterate items for a specific handle until end-of-stream (EOS).

        Args:
            name: Input handle name to read from.

        Yields:
            Each item (unwrapped data) arriving on the specified handle in FIFO order.
        """
        async for item in self._inbox.iter_input(name):
            yield item

    async def stream_with_envelope(self, name: str) -> AsyncIterator[MessageEnvelope]:
        """Iterate envelopes for a specific handle until end-of-stream (EOS).

        Use this when you need access to message metadata, timestamp, or event_id.

        Args:
            name: Input handle name to read from.

        Yields:
            Each MessageEnvelope arriving on the specified handle in FIFO order.
        """
        async for envelope in self._inbox.iter_input_with_envelope(name):
            yield envelope

    async def any(self) -> AsyncIterator[tuple[str, Any]]:
        """Iterate across all handles in cross-handle arrival order.

        Yields:
            Tuples of ``(handle, item)`` as items (unwrapped data) arrive from any handle.
        """
        async for handle, item in self._inbox.iter_any():
            yield handle, item

    async def any_with_envelope(self) -> AsyncIterator[tuple[str, MessageEnvelope]]:
        """Iterate across all handles in cross-handle arrival order with envelopes.

        Use this when you need access to message metadata, timestamp, or event_id.

        Yields:
            Tuples of ``(handle, MessageEnvelope)`` as envelopes arrive from any handle.
        """
        async for handle, envelope in self._inbox.iter_any_with_envelope():
            yield handle, envelope

    def has_buffered(self, name: str) -> bool:
        """Return True if the handle currently has any buffered items.

        Args:
            name: Input handle name to check.

        Returns:
            True if at least one item is buffered for ``name``.
        """
        return self._inbox.has_buffered(name)

    def has_stream(self, name: str) -> bool:
        """Return True if the handle has open upstream producers.

        Args:
            name: Input handle name to check.

        Returns:
            True if at least one upstream source is still open for ``name``.
        """
        return self._inbox.is_open(name)


class NodeOutputs:
    """Emitter wrapper for routing node outputs.

    Provides validation and routing for outputs produced by a node. In
    ``capture_only`` mode, outputs are collected but not routed downstream.

    Supports attaching metadata to outgoing messages, which will be propagated
    through the node graph via MessageEnvelopes.
    """

    def __init__(self, runner, node, context, capture_only: bool = False) -> None:
        """Initialize a NodeOutputs view.

        Args:
            runner: The active ``WorkflowRunner`` instance.
            node: The current ``BaseNode`` instance producing outputs.
            context: The workflow ``ProcessingContext``.
            capture_only: If True, collect outputs without routing them downstream.
        """
        # Lazy imports to avoid cycles
        from .base_node import BaseNode
        from .processing_context import ProcessingContext
        from .workflow_runner import WorkflowRunner

        assert isinstance(runner, WorkflowRunner)
        assert isinstance(node, BaseNode)
        assert isinstance(context, ProcessingContext)

        self.runner = runner
        self.node = node
        self.context = context
        self.capture_only = capture_only
        self._collected: dict[str, Any] = {}

    async def emit(
        self, slot: str, value: Any, metadata: dict[str, Any] | None = None
    ) -> None:
        """Emit a value to a specific output slot with optional metadata.

        Validates the slot name against the node's declared/dynamic outputs and
        routes the value via the runner unless in capture-only mode.

        Args:
            slot: Output slot name (uses "output" if empty/None).
            value: Value to emit.
            metadata: Optional metadata to attach to the outgoing message.
                     This metadata will be available to downstream nodes
                     that consume messages with envelope access.

        Raises:
            ValueError: If the slot does not exist on the node instance.
        """
        if slot == "" or slot is None:
            slot = "output"

        from .base_node import BaseNode, OutputNode

        assert isinstance(self.node, BaseNode)

        # Allow node to suppress routing for this slot
        if not self.node.should_route_output(slot):
            log.debug(
                "NodeOutputs.emit suppressed: node=%s (%s) slot=%s",
                self.node.get_title(),
                self.node._id,
                slot,
            )
            return
        if self.node.find_output_instance(slot) is None:
            raise ValueError(f"Node {self.node.get_title()} ({self.node._id}) tried to emit to unknown output '{slot}'")
        # Always collect the last value per slot
        self._collected[slot] = value
        log.debug(
            "NodeOutputs.emit: node=%s (%s) slot=%s capture_only=%s metadata=%s",
            self.node.get_title(),
            self.node._id,
            slot,
            self.capture_only,
            metadata is not None,
        )

        # Auto-save assets before routing to downstream nodes (for streaming outputs)
        # This ensures asset_id is set before downstream nodes receive the value
        if not self.capture_only and self.node.__class__.auto_save_asset():
            from nodetool.workflows.asset_storage import auto_save_assets

            # Wrap the single value in a dict for auto_save_assets
            await auto_save_assets(self.node, {slot: value}, self.context)

        # Capture outputs from OutputNode instances into runner.outputs
        if isinstance(self.node, OutputNode):
            node_name = self.node.name
            if node_name in self.runner.outputs:
                if not self.runner.outputs[node_name] or self.runner.outputs[node_name][-1] != value:
                    self.runner.outputs[node_name].append(value)
            else:
                self.runner.outputs[node_name] = [value]

        if not self.capture_only:
            send_messages = self.runner.send_messages
            if inspect.iscoroutinefunction(send_messages):
                await send_messages(self.node, {slot: value}, self.context, metadata)
            else:
                send_messages(self.node, {slot: value}, self.context, metadata)

    def complete(self, slot: str) -> None:
        """Mark early end-of-stream for a specific output slot.

        No-op in capture-only mode.

        Args:
            slot: Output slot name to close.
        """
        from .base_node import BaseNode
        from .processing_context import ProcessingContext
        from .workflow_runner import WorkflowRunner

        assert isinstance(self.node, BaseNode)
        assert isinstance(self.context, ProcessingContext)
        assert isinstance(self.runner, WorkflowRunner)

        if self.capture_only:
            return
        graph = self.context.graph
        for edge in graph.edges:
            if edge.source == self.node._id and edge.sourceHandle == slot:
                log.debug(
                    "NodeOutputs.complete: node=%s (%s) slot=%s edge=%s target=%s handle=%s",
                    self.node.get_title(),
                    self.node._id,
                    slot,
                    edge.id,
                    edge.target,
                    edge.targetHandle,
                )
                inbox = self.runner.node_inboxes.get(edge.target)
                if inbox is not None:
                    inbox.mark_source_done(edge.targetHandle)
                # Notify that this specific edge has been drained
                with suppress(Exception):
                    self.context.post_message(
                        EdgeUpdate(workflow_id=self.context.workflow_id, edge_id=edge.id or "", status="drained")
                    )

    async def default(self, value: Any, metadata: dict[str, Any] | None = None) -> None:
        """Convenience for emitting to the default slot with optional metadata.

        If an "output" slot exists, use it; otherwise if there is exactly one
        declared output, use its name; else fall back to "output".

        Args:
            value: Value to emit.
            metadata: Optional metadata to attach to the outgoing message.
        """
        from .base_node import BaseNode

        assert isinstance(self.node, BaseNode)

        outputs = self.node.outputs_for_instance()
        slot = "output"
        if any(o.name == "output" for o in outputs):
            slot = "output"
        elif len(outputs) == 1:
            slot = outputs[0].name
        await self.emit(slot, value, metadata)

    def collected(self) -> dict[str, Any]:
        """Return the map of collected outputs (slot -> last value).

        Returns:
            A shallow copy of the collected outputs for this node.
        """
        return dict(self._collected)
