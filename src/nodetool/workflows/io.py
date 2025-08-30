"""
Unified Node I/O wrappers used by BaseNode.run and NodeActor.

NodeInputs: convenience helpers over NodeInbox for reading inputs.
NodeOutputs: convenience helpers that validate and route outputs via WorkflowRunner.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from .inbox import NodeInbox


class NodeInputs:
    """Convenience wrapper for consuming node inputs.

    Wraps a ``NodeInbox`` to provide simple, intention-revealing methods for
    reading inputs in node implementations (e.g., inside ``BaseNode.run``).
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
            The first item from the specified handle, or ``default`` if none arrived.
        """
        async for item in self._inbox.iter_input(name):
            return item
        return default

    async def stream(self, name: str) -> AsyncIterator[Any]:
        """Iterate items for a specific handle until end-of-stream (EOS).

        Args:
            name: Input handle name to read from.

        Yields:
            Each item arriving on the specified handle in FIFO order.
        """
        async for item in self._inbox.iter_input(name):
            yield item

    async def any(self) -> AsyncIterator[tuple[str, Any]]:
        """Iterate across all handles in cross-handle arrival order.

        Yields:
            Tuples of ``(handle, item)`` as items arrive from any handle.
        """
        async for handle, item in self._inbox.iter_any():
            yield handle, item

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
        from .workflow_runner import WorkflowRunner  # noqa: F401
        from .base_node import BaseNode  # noqa: F401

        self.runner = runner
        self.node = node
        self.context = context
        self.capture_only = capture_only
        self._collected: dict[str, Any] = {}

    async def emit(self, slot: str, value: Any) -> None:
        """Emit a value to a specific output slot.

        Validates the slot name against the node's declared/dynamic outputs and
        routes the value via the runner unless in capture-only mode.

        Args:
            slot: Output slot name (uses "output" if empty/None).
            value: Value to emit.

        Raises:
            ValueError: If the slot does not exist on the node instance.
        """
        if slot == "" or slot is None:
            slot = "output"
        # Allow node to suppress routing for this slot
        if not self.node.should_route_output(slot):
            return
        if self.node.find_output_instance(slot) is None:
            raise ValueError(
                f"Node {self.node.get_title()} ({self.node._id}) tried to emit to unknown output '{slot}'"
            )
        # Always collect the last value per slot
        self._collected[slot] = value
        if not self.capture_only:
            self.runner.send_messages(self.node, {slot: value}, self.context)

    def complete(self, slot: str) -> None:
        """Mark early end-of-stream for a specific output slot.

        No-op in capture-only mode.

        Args:
            slot: Output slot name to close.
        """
        if self.capture_only:
            return
        graph = self.context.graph
        for edge in graph.edges:
            if edge.source == self.node._id and edge.sourceHandle == slot:
                inbox = self.runner.node_inboxes.get(edge.target)
                if inbox is not None:
                    inbox.mark_source_done(edge.targetHandle)

    async def default(self, value: Any) -> None:
        """Convenience for emitting to the default slot.

        If an "output" slot exists, use it; otherwise if there is exactly one
        declared output, use its name; else fall back to "output".

        Args:
            value: Value to emit.
        """
        outputs = self.node.outputs_for_instance()
        slot = "output"
        if any(o.name == "output" for o in outputs):
            slot = "output"
        elif len(outputs) == 1:
            slot = outputs[0].name
        await self.emit(slot, value)

    def collected(self) -> dict[str, Any]:
        """Return the map of collected outputs (slot -> last value).

        Returns:
            A shallow copy of the collected outputs for this node.
        """
        return dict(self._collected)
