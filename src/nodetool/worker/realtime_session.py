"""Realtime session lifecycle for the Python worker.

Holds a single warm ``BaseNode`` instance for the duration of a realtime
session. Sits between the stdio bridge protocol verbs (``start_session``,
``update_parameter``, ``push_input_frame``, ``stop_session``) and the
node's own lifecycle hooks (``pre_process``, ``on_session_start``,
``gen_process``, ``on_session_stop``).

See ``PLAN-REALTIME.md`` item 6c for the architectural rationale: this is
the Python end of the substrate that lets the TypeScript ``RealtimeRunner``
drive Python-implemented model nodes (LongLive, Self-Forcing, etc.)
without paying a per-frame round-trip cost outside of the bridge frame
itself.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from nodetool.config.logging_config import get_logger
from nodetool.worker.context_stub import WorkerContext
from nodetool.workflows.base_node import NODE_BY_TYPE, BaseNode
from nodetool.workflows.inbox import NodeInbox
from nodetool.workflows.realtime import RealtimeSessionInfo

log = get_logger(__name__)

#: Default per-handle buffer size for realtime input frames. Mirrors the
#: TS-side ``LATEST_FRAME_WINS_CAPACITY`` (drop-oldest, capacity 2) used by
#: media-source nodes so dropped-frame behaviour is symmetrical across the
#: bridge boundary.
DEFAULT_INPUT_BUFFER_SIZE = 2

#: Wall-clock seconds to wait for the runner task to drain after the inbox
#: is closed during ``stop()``. Past this, the task is cancelled.
DEFAULT_STOP_TIMEOUT = 5.0

#: Callback for emitting an output frame back to the bridge.
#:
#: The realtime worker invokes this every time the node yields an item from
#: ``gen_process``. Arguments are ``(handle, payload, metadata)`` —
#: implementations wrap them in the bridge's ``realtime_output_frame``
#: envelope and send the message via stdio. The ``metadata`` argument is
#: optional so node code can call ``await self.emit_frame(slot, value)``
#: when no per-frame metadata is needed.
FrameEmitter = Callable[..., Awaitable[None]]


class RealtimeSessionError(Exception):
    """Raised when a realtime session cannot be started or driven."""


class RealtimeNodeInstance:
    """A single warm ``BaseNode`` subscribed to a live realtime session.

    Lifecycle:

    1. :meth:`start` instantiates the node, applies fields, awaits
       ``pre_process`` and ``on_session_start``, then spawns a background
       task that drives ``gen_process`` (or, for non-streaming nodes,
       calls ``process`` once). Output items are forwarded to
       ``emit_frame`` as they arrive.
    2. :meth:`push_input_frame` enqueues a frame on the named handle with
       drop-oldest backpressure and returns the dropped-frame count.
    3. :meth:`update_parameter` mutates a node attribute live without a
       restart.
    4. :meth:`stop` closes the inbox so ``gen_process`` finishes,
       awaits the background task, then awaits ``on_session_stop`` so
       warm-state cleanup runs deterministically on every termination
       path (normal stop, error, cancellation).

    Instances are intentionally single-threaded with respect to the
    asyncio loop: ``push_input_frame`` and ``update_parameter`` are
    non-blocking and safe to call from the bridge dispatch coroutine.
    """

    def __init__(
        self,
        node: BaseNode,
        context: WorkerContext,
        session: RealtimeSessionInfo,
        inbox: NodeInbox,
        emit_frame: FrameEmitter,
        input_buffer_size: int = DEFAULT_INPUT_BUFFER_SIZE,
    ) -> None:
        self.node = node
        self.context = context
        self.session = session
        self.inbox = inbox
        self.emit_frame = emit_frame
        self.input_buffer_size = input_buffer_size
        self._task: asyncio.Task[None] | None = None
        self._stopped = False
        self._error: BaseException | None = None
        # Track which handles we've registered with the inbox so we don't
        # call add_upstream() repeatedly (each call increments the open
        # count).
        self._known_handles: set[str] = set()

    @property
    def session_id(self) -> str:
        return self.session.session_id

    @property
    def error(self) -> BaseException | None:
        """Exception captured by the runner task, if any."""
        return self._error

    @classmethod
    async def start(
        cls,
        session: RealtimeSessionInfo,
        node_type: str,
        fields: dict[str, Any] | None,
        secrets: dict[str, str] | None,
        emit_frame: FrameEmitter,
        input_buffer_size: int = DEFAULT_INPUT_BUFFER_SIZE,
    ) -> "RealtimeNodeInstance":
        """Build a warm session, await lifecycle hooks, start the runner task.

        Raises:
            RealtimeSessionError: If the node type is unknown, the node is
                not realtime-capable, or one of the lifecycle hooks fails.
        """
        node_class = NODE_BY_TYPE.get(node_type)
        if node_class is None:
            raise RealtimeSessionError(f"Unknown node type: {node_type}")
        if not node_class.is_realtime_capable():
            raise RealtimeSessionError(
                f"Node {node_type} is not realtime-capable "
                "(set _is_realtime_capable = True to opt in)"
            )

        node = node_class()

        # Fields apply first (workflow-graph-time configuration), then
        # session.parameters as a live overlay. Both go through
        # assign_property so type coercion / validation behaves identically.
        for key, value in (fields or {}).items():
            error = node.assign_property(key, value)
            if error:
                raise RealtimeSessionError(
                    f"assign_property({key!r}) failed: {error}"
                )
        for key, value in (session.parameters or {}).items():
            error = node.assign_property(key, value)
            if error:
                raise RealtimeSessionError(
                    f"assign_property({key!r}) from session.parameters failed: "
                    f"{error}"
                )

        ctx = WorkerContext(secrets=secrets or {}, cancel_event=asyncio.Event())
        # Drop-oldest backpressure is applied per-handle via
        # put_nowait_drop_oldest; the inbox itself stays unbounded so
        # internal control messages never block.
        inbox = NodeInbox(buffer_limit=None)
        node.attach_inbox(inbox)

        try:
            await node.pre_process(ctx)
            await node.on_session_start(ctx, session)
        except Exception as exc:
            # Make sure on_session_stop runs even if start failed mid-way,
            # so partial warm state is freed.
            try:
                await node.on_session_stop(ctx, session)
            except Exception:
                log.exception(
                    "on_session_stop failed during failed start cleanup "
                    "for session %s",
                    session.session_id,
                )
            raise RealtimeSessionError(
                f"Failed to start realtime session: {exc}"
            ) from exc

        instance = cls(
            node=node,
            context=ctx,
            session=session,
            inbox=inbox,
            emit_frame=emit_frame,
            input_buffer_size=input_buffer_size,
        )
        instance._spawn_runner()
        return instance

    def _spawn_runner(self) -> None:
        """Start the background task that drives the node's processing loop."""

        async def _run() -> None:
            try:
                if self.node.is_streaming_output():
                    async for item in self.node.gen_process(self.context):
                        if not isinstance(item, dict):
                            log.warning(
                                "Realtime node %s yielded non-dict item; "
                                "skipping.",
                                self.node.__class__.__name__,
                            )
                            continue
                        for slot, value in item.items():
                            if value is None:
                                continue
                            await self.emit_frame(slot, value)
                else:
                    # Buffered nodes are unusual for realtime but we run them
                    # once for completeness so authors get observable
                    # behaviour rather than a silent no-op.
                    result = await self.node.process(self.context)
                    if result is not None:
                        await self.emit_frame("output", result)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception(
                    "Realtime session %s runner failed", self.session_id
                )
                self._error = exc

        self._task = asyncio.create_task(_run())

    def push_input_frame(
        self,
        handle: str,
        payload: Any,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Enqueue a frame with drop-oldest semantics.

        Returns the number of items dropped to make room (``0`` when there
        was space). Non-blocking and safe to call from the bridge dispatch
        coroutine.

        ``metadata`` is attached to the message envelope so downstream
        ``iter_input_with_metadata`` consumers can read it; ignored by the
        plain ``iter_input`` consumer that most realtime nodes use.
        """
        if self._stopped:
            return 0
        if handle not in self._known_handles:
            self.inbox.add_upstream(handle, 1)
            self._known_handles.add(handle)
        return self.inbox.put_nowait_drop_oldest(
            handle, payload, self.input_buffer_size, metadata=metadata
        )

    def update_parameter(self, name: str, value: Any) -> bool:
        """Live parameter update — assigns to the node's pydantic field.

        Returns True if the parameter was routed (resolved to a known
        field on the node), False if the name is unknown so the bridge
        can surface a useful warning to the workflow author.
        """
        if name not in self.node.__class__.model_fields:
            return False
        error = self.node.assign_property(name, value)
        if error:
            log.warning(
                "Realtime parameter update %s=%r on %s failed: %s",
                name,
                value,
                self.node.__class__.__name__,
                error,
            )
            return False
        return True

    async def stop(self, timeout: float = DEFAULT_STOP_TIMEOUT) -> None:
        """Close the session: drain runner, then await ``on_session_stop``.

        Idempotent. Safe to call from any code path (normal stop, error,
        or cancellation) — ``on_session_stop`` always runs so warm state
        gets released deterministically.
        """
        if self._stopped:
            return
        self._stopped = True

        # Mark all known handles as done so iter_input loops exit cleanly,
        # then close the inbox to wake any blocked iterators.
        for handle in list(self._known_handles):
            self.inbox.mark_source_done(handle)
        await self.inbox.close_all()

        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=timeout)
            except asyncio.TimeoutError:
                log.warning(
                    "Realtime session %s runner did not finish within %.1fs; "
                    "cancelling.",
                    self.session_id,
                    timeout,
                )
                self._task.cancel()
                try:
                    await self._task
                except (asyncio.CancelledError, Exception):
                    pass
            self._task = None

        try:
            await self.node.on_session_stop(self.context, self.session)
        except Exception:
            log.exception(
                "on_session_stop failed for session %s", self.session_id
            )
