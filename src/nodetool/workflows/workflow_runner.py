"""
Workflow execution engine using per-node actors for DAG graphs.

This module implements the actor-based execution model for workflow graphs:
- One lightweight async task (NodeActor) per node drives that node to completion.
- Actors consume inputs from per-node inboxes, run the node (`process` once or
  `gen_process` for streaming outputs), and deliver outputs downstream.
- End-of-stream (EOS) is tracked per input handle; actors mark downstream EOS on
  completion or error to unblock consumers.

Unified model:
- Everything is a stream; single values are delivered as one-item streams.
- Nodes either consume once (buffered `process`) or iteratively (`gen_process`
  with `is_streaming_input() = True`).
- For repeating a subgraph per item, prefer an explicit ForEach/Map group node
  that feeds the subgraph N times and streams/collects outputs.

Core features:
- Validation and initialization of nodes based on a `Graph`.
- Unified execution for streaming and non-streaming nodes without a central scheduler loop.
- GPU coordination via a global FIFO lock when a node requires GPU.
- Caching for cacheable nodes, progress reporting, and output updates for `OutputNode`s.
- Dynamic device selection (cpu, cuda, mps) and torch context hooks (optional Comfy).

Example:
    from nodetool.workflows.run_job_request import RunJobRequest
    from nodetool.workflows.processing_context import ProcessingContext

    runner = WorkflowRunner(job_id="unique_job_id")
    await runner.run(req, context)
"""

import asyncio
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Any, Optional

from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.models.run_node_state import RunNodeState
from nodetool.models.run_state import RunState
from nodetool.observability.tracing import (
    get_or_create_tracer,
    remove_tracer,
    trace_workflow,
)
from nodetool.types.api_graph import Edge
from nodetool.types.job import JobUpdate
from nodetool.workflows.base_node import (
    BaseNode,
    InputNode,
    OutputNode,
)
from nodetool.workflows.event_logger import WorkflowEventLogger
from nodetool.workflows.graph import Graph
from nodetool.workflows.inbox import NodeInbox
from nodetool.workflows.memory_utils import (
    clear_memory_uri_cache,
    log_memory,
    log_memory_summary,
    run_gc,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.state_manager import StateManager
from nodetool.workflows.suspendable_node import WorkflowSuspendedException
from nodetool.workflows.torch_support import (
    TORCH_AVAILABLE,
    BaseTorchSupport,
    build_torch_support,
    is_cuda_available,
    torch,
)
from nodetool.workflows.types import EdgeUpdate, NodeUpdate, OutputUpdate

log = get_logger(__name__)
# Log level is controlled by env (DEBUG/NODETOOL_LOG_LEVEL)

MAX_RETRIES = 2
BASE_DELAY = 1  # seconds
MAX_DELAY = 60  # seconds

# Brief delay (in seconds) for completion detection race condition handling.
# After all tasks complete, we wait briefly and re-check for pending inbox work
# to handle race conditions where EOS signals are still being processed.
COMPLETION_CHECK_DELAY = 0.01


# Define a process-wide GPU lock that is safe across event loops/threads
gpu_lock = threading.Lock()
# Track which node/thread holds the lock for debugging
_gpu_lock_holder: str | None = None
_gpu_lock_holder_time: float = 0.0

# Maximum time to wait for GPU lock before giving up (seconds)
GPU_LOCK_TIMEOUT = 300  # 5 minutes


async def acquire_gpu_lock(node: BaseNode, context: ProcessingContext):
    """
    Asynchronously acquires the global GPU lock for a given node.

    If the lock is currently held, this function will send a "waiting"
    status update for the node before attempting to acquire the lock.
    This function wraps the `gpu_lock.acquire()` call using a background thread
    so the asyncio event loop remains non-blocking.

    Args:
        node (BaseNode): The node attempting to acquire the GPU lock.
        context (ProcessingContext): The processing context, used for sending updates.

    Raises:
        RuntimeError: If lock cannot be acquired within GPU_LOCK_TIMEOUT seconds.
        asyncio.CancelledError: If the task is cancelled while waiting.
    """
    global _gpu_lock_holder, _gpu_lock_holder_time

    if gpu_lock.locked():
        holder_info = f" (held by: {_gpu_lock_holder})" if _gpu_lock_holder else ""
        hold_time = time.time() - _gpu_lock_holder_time if _gpu_lock_holder_time else 0
        log.warning(f"Node {node.get_title()} is waiting for GPU lock{holder_info}, held for {hold_time:.1f}s")
        await node.send_update(context, status="waiting")

    # Acquire the threading lock without blocking the event loop.
    # Use short timeouts so task cancellation does not leak a held lock.
    loop = asyncio.get_running_loop()
    start_time = time.time()
    attempts = 0

    try:
        while True:
            # Check for cancellation before trying to acquire
            # This allows clean shutdown when workflow is cancelled
            try:
                acquired = await loop.run_in_executor(None, lambda: gpu_lock.acquire(timeout=0.2))
            except asyncio.CancelledError:
                log.info(f"Node {node.get_title()} cancelled while waiting for GPU lock")
                raise

            if acquired:
                _gpu_lock_holder = f"{node.get_title()} ({node.id})"
                _gpu_lock_holder_time = time.time()
                break

            attempts += 1
            elapsed = time.time() - start_time

            # Log progress every 10 seconds
            if attempts % 40 == 0:  # 40 * 0.25s = 10s
                holder_info = f" (held by: {_gpu_lock_holder})" if _gpu_lock_holder else ""
                log.warning(f"Node {node.get_title()} still waiting for GPU lock after {elapsed:.1f}s{holder_info}")

            # Timeout after GPU_LOCK_TIMEOUT seconds
            if elapsed > GPU_LOCK_TIMEOUT:
                holder_info = f" (held by: {_gpu_lock_holder})" if _gpu_lock_holder else ""
                error_msg = (
                    f"GPU lock acquisition timed out after {GPU_LOCK_TIMEOUT}s for node "
                    f"{node.get_title()}{holder_info}. This may indicate a stuck previous run. "
                    f"Try restarting the nodetool server."
                )
                log.error(error_msg)
                raise RuntimeError(error_msg)

            # Yield briefly before retrying - also check for cancellation here
            try:
                await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                log.info(f"Node {node.get_title()} cancelled while waiting for GPU lock")
                raise

    except asyncio.CancelledError:
        # Make sure we don't hold the lock if we were cancelled after acquiring
        # This shouldn't happen given the structure, but be safe
        log.debug(f"GPU lock acquisition cancelled for {node.get_title()}")
        raise

    log.debug(f"Node {node.get_title()} acquired GPU lock")


def release_gpu_lock():
    """
    Releases the global GPU lock.
    """
    global _gpu_lock_holder, _gpu_lock_holder_time
    log.debug(f"Releasing GPU lock (was held by: {_gpu_lock_holder})")
    _gpu_lock_holder = None
    _gpu_lock_holder_time = 0.0
    gpu_lock.release()


def force_release_gpu_lock():
    """
    Force-release the GPU lock if it appears stuck.

    This should only be called as a last resort when debugging stuck workflows.
    Returns True if the lock was released, False if it wasn't held.
    """
    global _gpu_lock_holder, _gpu_lock_holder_time
    if gpu_lock.locked():
        log.warning(f"Force-releasing GPU lock (was held by: {_gpu_lock_holder})")
        try:
            gpu_lock.release()
            _gpu_lock_holder = None
            _gpu_lock_holder_time = 0.0
            return True
        except RuntimeError:
            # Lock wasn't held by this thread
            log.error("Cannot force-release GPU lock - not held by current thread")
            return False
    else:
        log.info("GPU lock is not currently held")
        return False


def get_gpu_lock_status() -> dict:
    """
    Get the current status of the GPU lock for debugging.

    Returns:
        Dict with lock status information.
    """
    return {
        "locked": gpu_lock.locked(),
        "holder": _gpu_lock_holder,
        "held_since": _gpu_lock_holder_time,
        "held_for_seconds": time.time() - _gpu_lock_holder_time if _gpu_lock_holder_time else 0,
    }


class WorkflowRunner:
    """
    Actor-based DAG execution engine for computational nodes.

    WorkflowRunner orchestrates execution by:
    1. Validating the graph and initializing nodes.
    2. Building per-node `NodeInbox` instances and attaching them to nodes.
    3. Starting one `NodeActor` per node; each actor consumes inputs, executes the node
       (`process` once or `gen_process` for streaming), and emits outputs downstream.
    4. Managing GPU access via a global FIFO lock when needed.
    5. Handling caching, output updates, and cleanup.

    Attributes:
        job_id: Unique identifier for this workflow run.
        status: "running", "completed", or "error".
        current_node: ID of the node currently being processed (for progress hooks).
        context: The active `ProcessingContext` during a run.
        outputs: Final outputs collected from `OutputNode`s (list per output name).
        device: Selected device ("cpu", "cuda", "mps").
        active_processing_node_ids: Node IDs currently running in async tasks.
        node_inboxes: Per-node inboxes for input delivery and EOS tracking.
    """

    def __init__(
        self,
        job_id: str,
        device: str | None = None,
        disable_caching: bool = False,
        buffer_limit: int | None = 3,
        enable_event_logging: bool = True,
    ):
        """
        Initializes a new WorkflowRunner instance.

        Args:
            job_id (str): Unique identifier for this workflow execution.
            device (Optional[str]): The specific device ("cpu", "cuda", "mps") to run the workflow on.
                                    If None, it auto-detects based on Torch availability (CUDA, then MPS, then CPU).
            disable_caching (bool): Whether to disable result caching for cacheable nodes.
            buffer_limit (Optional[int]): Maximum number of items allowed in each per-handle inbox buffer.
                                         When a buffer reaches this limit, producers will block until consumers
                                         drain the buffer, implementing backpressure. None means unlimited.
            enable_event_logging (bool): Whether to enable event logging for resumability.
        """
        self.job_id = job_id
        self.status = "running"
        self.current_node: Optional[str] = None
        self.context: Optional[ProcessingContext] = None
        self.outputs: dict[str, Any] = {}
        self.active_processing_node_ids: set[str] = set()  # Track nodes currently in an async task
        self.node_inboxes: dict[str, NodeInbox] = {}
        self.disable_caching = disable_caching
        self.buffer_limit = buffer_limit
        self._torch_support: BaseTorchSupport = build_torch_support(
            base_delay=BASE_DELAY,
            max_delay=MAX_DELAY,
            max_retries=MAX_RETRIES,
        )
        if device:
            self.device = device
        else:
            self.device = "cpu"
            if TORCH_AVAILABLE:
                if is_cuda_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "mps"

            log.info(f"Workflow runs on device: {self.device}")
            log.debug(f"WorkflowRunner initialized for job_id: {self.job_id} with device: {self.device}")
        # Streaming input queue and dispatcher task (created during run())
        self._input_queue: asyncio.Queue | None = None
        self._input_task: asyncio.Task | None = None
        # Event loop where the runner is executing; used for thread-safe enqueues
        self._runner_loop: asyncio.AbstractEventLoop | None = None
        self._streaming_edges: dict[str, bool] = {}
        self._edge_counters: dict[str, int] = defaultdict(int)

        # Event logging for resumability (audit-only, not source of truth)
        self.enable_event_logging = enable_event_logging
        self.event_logger: WorkflowEventLogger | None = None
        if enable_event_logging:
            self.event_logger = WorkflowEventLogger(run_id=job_id)

        # State table tracking (source of truth for correctness)
        self.run_state: RunState | None = None

        # State manager for queue-based updates (eliminates DB write contention)
        self.state_manager: StateManager | None = None

        # Multi-edge list inputs: {node_id: {handle_names requiring aggregation}}
        # Populated during graph analysis to track list-typed properties with multiple incoming edges
        self.multi_edge_list_inputs: dict[str, set[str]] = {}

    def _edge_key(self, edge: Edge) -> str:
        return edge.id or (f"{edge.source}:{edge.sourceHandle}->{edge.target}:{edge.targetHandle}")

    def edge_streams(self, edge: Edge) -> bool:
        """Return True if the given edge is marked as streaming."""

        return self._streaming_edges.get(self._edge_key(edge), False)

    def _analyze_streaming(self, graph: Graph) -> None:
        """Populate ``_streaming_edges`` with streaming propagation info.

        Any node that reports streaming outputs (``is_streaming_output()``) seeds a
        simple breadth-first walk across the graph. Every edge reachable from those
        sources is marked as streaming, and the streaming flag propagates through
        intermediate nodes regardless of their own streaming capabilities. Edges not
        reachable from a streaming source are marked as non-streaming.
        """

        self._streaming_edges.clear()

        if not graph.edges:
            return

        adjacency: dict[str, list[Edge]] = defaultdict(list)
        for edge in graph.edges:
            adjacency[edge.source].append(edge)
            self._streaming_edges[self._edge_key(edge)] = False

        queue: deque[str] = deque()
        visited: set[str] = set()

        for node in graph.nodes:
            try:
                node_id = getattr(node, "_id", getattr(node, "id", None))
                if node_id is None:
                    continue
                if node.is_streaming_output():
                    queue.append(node_id)
                    visited.add(node_id)
            except Exception:
                # Best-effort streaming detection; ignore misbehaving nodes
                continue

        while queue:
            current = queue.popleft()
            for edge in adjacency.get(current, []):
                self._streaming_edges[self._edge_key(edge)] = True
                target_id = edge.target
                if target_id not in visited:
                    visited.add(target_id)
                    queue.append(target_id)

    def _classify_list_inputs(self, graph: Graph) -> None:
        """Identify properties that require list aggregation.

        Populates ``self.multi_edge_list_inputs`` with a mapping from node IDs to
        the set of handle names that:
        1. Have type ``list[T]``
        2. Have one or more incoming edges on the same targetHandle

        For these handles, the actor will collect all incoming values into a list
        before invoking the node's process method, rather than using the default
        behavior of taking the first/latest value.
        """
        self.multi_edge_list_inputs.clear()

        if not graph.edges:
            return

        # Group edges by (target_node_id, targetHandle)
        edges_by_target_handle: dict[tuple[str, str], list[Edge]] = defaultdict(list)
        for edge in graph.edges:
            key = (edge.target, edge.targetHandle)
            edges_by_target_handle[key].append(edge)

        # Check each target handle
        for (node_id, handle), edges in edges_by_target_handle.items():
            if len(edges) == 0:
                continue

            node = graph.find_node(node_id)
            if node is None:
                continue

            prop = node.find_property(handle)
            if prop is None:
                continue

            # Check if the property type is a list type
            if not prop.type.is_list_type():
                # Multiple edges to non-list property will be caught during validation
                continue

            # Mark this handle for list aggregation (single or multiple edges)
            if node_id not in self.multi_edge_list_inputs:
                self.multi_edge_list_inputs[node_id] = set()
            self.multi_edge_list_inputs[node_id].add(handle)

        if self.multi_edge_list_inputs:
            log.debug(f"Multi-edge list inputs detected: {self.multi_edge_list_inputs}")

    # --- Streaming Input support for InputNode(is_streaming=True) ---
    def _enqueue_input_event(self, event: dict[str, Any]) -> None:
        """
        Thread-safe enqueue of input events onto the runner's asyncio.Queue.

        If called from another thread (e.g., WebSocket server loop), schedules
        the put_nowait on the runner loop via call_soon_threadsafe.
        """
        if self._input_queue is None:
            raise RuntimeError("Input queue is not initialized")
        loop = self._runner_loop
        op = event.get("op")
        inp = event.get("input")
        handle = event.get("handle")
        if op is None or inp is None or handle is None:
            raise ValueError("Invalid event")
        if loop is not None:
            log.debug(
                f"Enqueue (thread-safe) input event: op={op} input={inp} handle={handle} current_thread={threading.get_ident()} loop_id={id(loop)}"
            )
            loop.call_soon_threadsafe(self._input_queue.put_nowait, event)
            return
        self._input_queue.put_nowait(event)

    def _find_input_node_id(self, context: ProcessingContext, input_name: str) -> str:
        assert context.graph is not None, "Graph not set in context"
        for node in context.graph.inputs():
            if getattr(node, "name", None) == input_name:
                return node._id
        raise ValueError(f"Input node not found for input name: {input_name}")

    def push_input_value(self, *, input_name: str, value: Any, source_handle: str | None = None) -> None:
        """
        Enqueue a streaming input event to be dispatched on the runner loop.
        """
        if self._input_queue is None:
            raise RuntimeError("Input queue is not initialized")
        # Default to the standard InputNode output handle when none is provided
        event = {
            "op": "push",
            "input": input_name,
            "value": value,
            "handle": source_handle or "output",
        }
        log.debug(f"Enqueue input push: op:{event['op']}, input:{event['input']}, handle:{event['handle']}")
        self._enqueue_input_event(event)

    def finish_input_stream(self, *, input_name: str, source_handle: str | None = None) -> None:
        """
        Signal end-of-stream for a streaming input. This marks downstream inboxes as
        done for the corresponding target handles so consumers can complete.
        """
        if self._input_queue is None:
            raise RuntimeError("Input queue is not initialized")
        # Default to the standard InputNode output handle when none is provided
        event = {"op": "end", "input": input_name, "handle": source_handle or "output"}
        log.debug(f"Enqueue input end: op:{event['op']}, input:{event['input']}, handle:{event['handle']}")
        self._enqueue_input_event(event)

    async def _dispatch_inputs(self, context: ProcessingContext) -> None:
        assert self._input_queue is not None
        assert context.graph is not None
        graph = context.graph
        try:
            loop = asyncio.get_running_loop()
            log.debug(f"Input dispatcher started: loop_id={id(loop)} queue_id={id(self._input_queue)}")
        except Exception:
            log.debug("Input dispatcher started (loop id unavailable)")
        while True:
            ev = await self._input_queue.get()
            log.debug(f"Dispatch input event: op:{ev.get('op')}, input:{ev.get('input')}, handle:{ev.get('handle')}")
            if ev.get("op") == "shutdown":
                log.debug("Input dispatcher received shutdown; exiting")
                return
            try:
                input_name: str = ev.get("input")
                node_id = self._find_input_node_id(context, input_name)
                node = graph.find_node(node_id)
                if node is None:
                    log.warning(f"Dispatch event dropped: input node not found for {input_name}")
                    continue
                # Determine output handle from event or InputNode defaults
                # Default to the standard InputNode output handle when none is provided
                handle = ev.get("handle") or "output"
                if ev.get("op") == "push":
                    value = ev.get("value")

                    for edge in graph.find_edges(node_id, handle):
                        inbox = self.node_inboxes.get(edge.target)
                        if inbox is not None:
                            await inbox.put(edge.targetHandle, value)
                            edge_id = edge.id or ""
                            self._edge_counters[edge_id] += 1
                            context.post_message(
                                EdgeUpdate(
                                    workflow_id=context.workflow_id,
                                    edge_id=edge_id,
                                    status="message_sent",
                                    counter=self._edge_counters[edge_id],
                                )
                            )
                        else:
                            log.debug(f"No inbox for target {edge.target} on edge {edge.id}")
                elif ev.get("op") == "end":
                    for edge in graph.find_edges(node_id, handle):
                        inbox = self.node_inboxes.get(edge.target)
                        if inbox is not None:
                            inbox.mark_source_done(edge.targetHandle)
                            context.post_message(
                                EdgeUpdate(workflow_id=context.workflow_id, edge_id=edge.id or "", status="drained")
                            )
                        else:
                            log.debug(f"No inbox for target {edge.target} on edge {edge.id}")
            except Exception as e:
                log.error(f"Error dispatching input event: {e}")
                pass

    def is_running(self) -> bool:
        """
        Checks if the workflow is currently in the "running" state.

        Returns:
            bool: True if the workflow status is "running", False otherwise.
        """
        return self.status == "running"

    def _filter_invalid_edges(self, graph: Graph) -> None:
        """Remove edges that reference non-existent nodes/slots or feed into a node's
        *output* socket.

        This is meant as a last-resort safety net so that a single mis-wired edge
        does not stall the whole workflow.  Any edge that fulfils one of the
        following conditions will be dropped:

        1. Source node cannot be found.
        2. Target node cannot be found.
        3. Source handle is not a declared output of the source node.
        4. Target handle is not a declared property of the target node *and* the
           target node is *not* dynamic.

        The method mutates ``graph.edges`` in-place.
        """
        valid_edges = []
        removed: list[str] = []  # store edge ids for logging

        for edge in graph.edges:
            # 1 / 2 - both nodes must exist
            source_node = graph.find_node(edge.source)
            target_node = graph.find_node(edge.target)
            if source_node is None or target_node is None:
                log.warning(f"Edge {edge.id} has a source or target node that does not exist")
                removed.append(edge.id or "<unknown>")
                continue

            target_cls = target_node.__class__

            # 3 - source handle must be an output on the *source* node (instance-aware)
            if source_node.find_output_instance(edge.sourceHandle) is None:  # type: ignore
                log.warning(f"Edge {edge.id} has a source handle that does not exist")
                removed.append(edge.id or "<unknown>")
                continue

            # 4 - target property must exist unless node is dynamic
            if not target_cls.is_dynamic() and target_node.find_property(edge.targetHandle) is None:
                log.warning(f"Edge {edge.id} has a target handle that does not exist")
                removed.append(edge.id or "<unknown>")
                continue

            # Edge passed all checks - keep it
            valid_edges.append(edge)

        # Save removed edge IDs for potential teardown notifications
        self._removed_edge_ids = removed

        # Replace edges in the graph with the validated list
        graph.edges = valid_edges

    async def run(
        self,
        request: RunJobRequest,
        context: ProcessingContext,
        send_job_updates: bool = True,
        initialize_graph: bool = True,
        validate_graph: bool = True,
    ):
        """
        Executes the entire workflow based on the provided request and context.

        This is the main entry point for running a workflow. It handles:
        - Setting up the graph and context.
        - Processing input parameters and messages.
        - Validating and initializing the graph.
        - Orchestrating the graph processing loop.
        - Finalizing nodes and cleaning up resources (e.g., CUDA cache).
        - Posting job status updates (running, completed, error).

        Args:
            request (RunJobRequest): Contains the workflow graph, input parameters, and initial messages.
            context (ProcessingContext): Manages the execution state, inter-node communication,
                                     and provides services like caching.
            send_job_updates (bool): Whether to send job updates to the client.
            initialize_graph (bool): Whether to initialize the graph.
            validate_graph (bool): Whether to validate the graph.

        Raises:
            ValueError: If the graph is missing from the request or if there's a mismatch
                        between input parameters and graph input nodes
            Exception: Propagates exceptions from graph processing, including CUDA OOM errors
                       if they persist after retries.

        Post-conditions:
            - Updates workflow status to "completed", "cancelled", or "error".
            - Posts a final JobUpdate message with results or error information.
        """
        # Create tracer for this workflow run
        tracer = get_or_create_tracer(self.job_id)

        try:
            async with trace_workflow(
                job_id=self.job_id,
                workflow_id=request.workflow_id,
                user_id=request.user_id,
                tracer=tracer,
            ) as span:
                await self._run_workflow(
                    request=request,
                    context=context,
                    send_job_updates=send_job_updates,
                    initialize_graph=initialize_graph,
                    validate_graph=validate_graph,
                    span=span,
                )
        finally:
            # Clean up tracer - must be AFTER trace_workflow exits to ensure span is ended
            remove_tracer(self.job_id)

    async def _run_workflow(
        self,
        request: RunJobRequest,
        context: ProcessingContext,
        send_job_updates: bool,
        initialize_graph: bool,
        validate_graph: bool,
        span,
    ):
        """Internal method containing the actual workflow execution logic."""
        log.info("Starting workflow run: job_id=%s", self.job_id)
        log_memory(f"WorkflowRunner.run START job_id={self.job_id}")
        self._edge_counters.clear()
        self.status = "running"
        log.debug("Run parameters: params=%s messages=%s", request.params, request.messages)
        log.debug(f"WorkflowRunner.run called for job_id: {self.job_id} with req: {request}, context: {context}")

        Environment.load_settings()

        assert request.graph is not None, "Graph is required"

        # Load graph with skip_errors=False to ensure node loading failures are raised
        # rather than silently dropping nodes
        graph = Graph.from_dict(
            request.graph.model_dump(),
            skip_errors=False,
            allow_undefined_properties=True,
        )
        log.info(
            "Graph loaded: %d nodes, %d edges",
            len(graph.nodes),
            len(graph.edges),
        )
        self._filter_invalid_edges(graph)

        log.info(
            "Graph prepared: %d nodes, %d valid edges after filtering",
            len(graph.nodes),
            len(graph.edges),
        )

        context.graph = graph
        self._analyze_streaming(graph)
        self._classify_list_inputs(graph)
        self._initialize_inboxes(context, graph)
        self.context = context
        context.device = self.device
        log.debug(f"Context and device set. Device: {self.device}")

        # Validate that all InputNodes have non-empty names
        invalid_inputs: list[str] = []
        for node in graph.inputs():
            try:
                if not getattr(node, "name", None):
                    invalid_inputs.append(node._id)
            except Exception:
                pass
        if invalid_inputs:
            raise ValueError(f"All InputNode(s) must have a non-empty name. Invalid: {', '.join(invalid_inputs)}")

        input_nodes = {node.name: node for node in graph.inputs()}
        duplicate_names = []
        seen_names = set()
        for node in graph.inputs():
            name = getattr(node, "name", None)
            if name:
                if name in seen_names:
                    duplicate_names.append(name)
                seen_names.add(name)
        if duplicate_names:
            raise ValueError(
                f"Multiple InputNode(s) have the same name. Duplicate names: {', '.join(set(duplicate_names))}. Please use unique names for each input node."
            )

        start_time = time.time()
        if send_job_updates:
            log.debug(f"Posting 'running' job update for job {self.job_id}")
            context.post_message(JobUpdate(job_id=self.job_id, workflow_id=context.workflow_id, status="running"))

        # Create run_state (source of truth) - creates if not exists for direct runner usage
        try:
            self.run_state = await RunState.get(self.job_id)
            if self.run_state is None:
                self.run_state = await RunState.create_run(
                    run_id=self.job_id,
                    execution_strategy=request.execution_strategy.value if request.execution_strategy else None,
                )
                log.info(f"Created run_state for {self.job_id} with status={self.run_state.status}")
            else:
                log.info(f"Loaded run_state for {self.job_id} with status={self.run_state.status}")
        except Exception as e:
            log.error(f"Failed to load/create run_state: {e}")
            raise

        # Initialize and start StateManager (single writer for node states)
        self.state_manager = StateManager(run_id=self.job_id)
        await self.state_manager.start()
        log.info(f"Started StateManager for run {self.job_id}")

        # Start event logger background flush task (for non-blocking event logging)
        if self.event_logger:
            try:
                await self.event_logger.start()
                log.info(f"Started EventLogger for run {self.job_id}")
            except Exception as e:
                log.warning(f"Failed to start EventLogger (non-fatal): {e}")

        # Log RunCreated event (audit-only, non-fatal)
        if self.event_logger:
            try:
                await self.event_logger.log_run_created(
                    graph=request.graph.model_dump() if request.graph else {},
                    params=request.params or {},
                    user_id=getattr(context, "user_id", ""),
                )
            except Exception as e:
                log.warning(f"Failed to log RunCreated event (non-fatal): {e}")

        with self.torch_context(context):
            try:
                if request.params:
                    log.info(f"Processing params: {request.params}")
                    log.info(f"Available input nodes: {list(input_nodes.keys())}")
                    for key, value in request.params.items():
                        log.info(f"Setting input node {key} to {value}")
                        if key not in input_nodes:
                            log.warning(f"input params {key} not found as input node")
                        else:
                            node = input_nodes[key]
                            log.info(f"Assigning property 'value'={value} to node {node.id} ({node.name})")
                            node.assign_property("value", value)

                if validate_graph:
                    await self.validate_graph(context, graph)
                if initialize_graph:
                    await self.initialize_graph(context, graph)
                # Start input dispatcher to consume queued input events and route to inboxes
                # Capture the runner's event loop for thread-safe enqueues
                try:
                    self._runner_loop = asyncio.get_running_loop()
                except RuntimeError:
                    self._runner_loop = None
                self._input_queue = asyncio.Queue()
                self._input_task = asyncio.create_task(self._dispatch_inputs(context))
                try:
                    log.debug(
                        f"Runner loop captured: loop_id={id(self._runner_loop) if self._runner_loop else None} thread_id={threading.get_ident()}"
                    )
                    log.debug(
                        f"Input queue and dispatcher started: queue_id={id(self._input_queue)} task_id={id(self._input_task)}"
                    )
                except Exception:
                    pass
                # Enqueue initial params/messages via unified input queue
                if request.params:
                    for key, value in request.params.items():
                        if key not in input_nodes:
                            log.warning(f"input params {key} not found as input node")
                        else:
                            node = input_nodes[key]
                            # Determine the correct output handle name for this InputNode
                            outputs = node.outputs_for_instance()
                            handle_name = outputs[0].name if outputs else "output"

                            # push value on the node's declared output handle; end stream if not streaming
                            log.info(f"Pushing input value for {key}: {value} handle={handle_name}")
                            self.push_input_value(
                                input_name=getattr(node, "name", key),
                                value=value,
                                source_handle=handle_name,
                            )
                            if not node.is_streaming_output():
                                # default: treat as non-streaming and end
                                self.finish_input_stream(
                                    input_name=getattr(node, "name", key),
                                    source_handle=handle_name,
                                )

                # Also enqueue default values configured directly on graph InputNodes
                # (e.g., NumberInput.value) when not provided via request.params.
                for node in graph.inputs():
                    name = getattr(node, "name", None)
                    if not name:
                        continue
                    # Skip if provided via params
                    if request.params and name in request.params:
                        continue
                    default_value = getattr(node, "value", None)
                    if default_value is None:
                        continue
                    # Skip empty values (e.g., default-initialized Message objects)
                    # This prevents pushing empty messages that would cause downstream
                    # nodes to execute with empty inputs before real input arrives.
                    # We use duck-typing here to support any type that implements is_empty(),
                    # such as Message, ImageRef, AudioRef, etc. from nodetool.metadata.types.
                    if hasattr(default_value, "is_empty") and callable(default_value.is_empty):
                        if default_value.is_empty():
                            continue
                    try:
                        outputs = node.outputs_for_instance()
                        handle_name = outputs[0].name if outputs else "output"
                    except Exception:
                        handle_name = "output"
                    log.info(f"Pushing default value for {name}: {default_value} handle={handle_name}")
                    self.push_input_value(input_name=name, value=default_value, source_handle=handle_name)
                    if not node.is_streaming_output():
                        self.finish_input_stream(input_name=name, source_handle=handle_name)

                await self.process_graph(context, graph)

                # If we reach here, no exceptions from the main processing stages
                if self.status == "running":  # Check if it wasn't set to error by some internal logic
                    self.status = "completed"

                    # Update run_state (source of truth)
                    if self.run_state:
                        try:
                            await self.run_state.mark_completed()
                            log.info(f"Marked run_state as completed for {self.job_id}")
                        except Exception as e:
                            log.error(f"Failed to mark run_state as completed: {e}")

                    # Send completion JobUpdate BEFORE finally block
                    # The WebSocket processor may close during finally, so send this early
                    if send_job_updates:
                        context.post_message(
                            JobUpdate(
                                job_id=self.job_id,
                                status="completed",
                                workflow_id=context.workflow_id,
                                result=self.outputs,
                                message=f"Workflow {self.job_id} completed",
                            )
                        )

            except asyncio.CancelledError:
                # Gracefully handle external cancellation.
                # We do not emit synthetic per-edge "drained" UI messages.
                self.status = "cancelled"

                # Update run_state (source of truth)
                if self.run_state:
                    try:
                        await self.run_state.mark_cancelled()
                        log.info(f"Marked run_state as cancelled for {self.job_id}")
                    except Exception as e:
                        log.error(f"Failed to mark run_state as cancelled: {e}")

                # Log RunCancelled event (audit-only, non-fatal)
                if self.event_logger:
                    try:
                        await self.event_logger.log_run_cancelled(reason="Workflow execution cancelled")
                    except Exception as e:
                        log.warning(f"Failed to log RunCancelled event (non-fatal): {e}")

                if send_job_updates:
                    context.post_message(JobUpdate(job_id=self.job_id, workflow_id=context.workflow_id, status="cancelled"))

            except WorkflowSuspendedException as e:
                # Handle workflow suspension from suspendable node
                self.status = "suspended"

                log.info(f"Workflow {self.job_id} suspended at node {e.node_id}: {e.reason}")

                # Update run_state (source of truth)
                if self.run_state:
                    try:
                        await self.run_state.mark_suspended(
                            node_id=e.node_id,
                            reason=e.reason,
                            state=e.state,
                            metadata=e.metadata,
                        )
                        log.info(f"Marked run_state as suspended for {self.job_id} at node {e.node_id}")
                    except Exception as e2:
                        log.error(f"Failed to mark run_state as suspended: {e2}")
                        raise

                # Update node_state to suspended (source of truth)
                try:
                    node_state = await RunNodeState.get_or_create(
                        run_id=self.job_id,
                        node_id=e.node_id,
                    )
                    await node_state.mark_suspended(
                        reason=e.reason,
                        state=e.state,
                    )
                    log.info(f"Marked node_state as suspended for node {e.node_id}")
                except Exception as e2:
                    log.error(f"Failed to mark node_state as suspended: {e2}")
                    raise

                # Flush pending state updates from all nodes before suspending
                if self.state_manager:
                    try:
                        await self.state_manager.stop(timeout=5.0)
                        log.info(f"Flushed and stopped StateManager for run {self.job_id}")
                        self.state_manager = None  # Prevent finally block from stopping again
                    except Exception as e2:
                        log.warning(f"Failed to stop StateManager: {e2}")

                # Stop EventLogger for suspension
                if self.event_logger:
                    try:
                        await self.event_logger.stop()
                        log.info(f"EventLogger stopped for suspension")
                        self.event_logger = None  # Prevent finally block from stopping again
                    except Exception as e2:
                        log.warning(f"Failed to stop EventLogger: {e2}")

                # Log suspension events (audit-only, non-fatal)
                if self.event_logger:
                    try:
                        # Log NodeSuspended event
                        await self.event_logger.log_node_suspended(
                            node_id=e.node_id,
                            reason=e.reason,
                            state=e.state,
                            metadata=e.metadata,
                        )

                        # Log RunSuspended event
                        await self.event_logger.log_run_suspended(
                            reason=e.reason,
                            suspended_node_id=e.node_id,
                        )

                        # Check if this is a trigger node suspension
                        if e.metadata.get("trigger_node"):
                            # Register with trigger wakeup service
                            from nodetool.workflows.trigger_node import TriggerWakeupService

                            wakeup_service = TriggerWakeupService.get_instance()
                            wakeup_service.register_suspended_trigger(
                                workflow_id=self.job_id,
                                node_id=e.node_id,
                                trigger_metadata=e.metadata,
                            )
                            log.info(f"Registered trigger node {e.node_id} for wake-up in workflow {self.job_id}")

                    except Exception as e2:
                        log.warning(f"Failed to log suspension events (non-fatal): {e2}")

                if send_job_updates:
                    context.post_message(
                        JobUpdate(
                            job_id=self.job_id,
                            status="suspended",
                            workflow_id=context.workflow_id,
                            message=f"Workflow suspended at node {e.node_id}: {e.reason}",
                        )
                    )

                # Do not re-raise - suspension is a clean exit
                return

            except Exception as e:
                error_message_for_job_update = str(e)
                log.error(f"Error during graph execution for job {self.job_id}: {error_message_for_job_update}")
                log.debug(f"Exception caught in WorkflowRunner.run: {e}", exc_info=True)

                # Specific handling for OOM error message, but status is always error
                if self._torch_support.is_cuda_oom_exception(e):
                    error_message_for_job_update = (
                        f"VRAM OOM error: {str(e)}. No additional VRAM available after retries."
                    )
                    # log.error already done by generic message

                self.status = "error"

                # Update run_state (source of truth)
                if self.run_state:
                    try:
                        await self.run_state.mark_failed(error=error_message_for_job_update[:1000])
                        log.info(f"Marked run_state as failed for {self.job_id}")
                    except Exception as e2:
                        log.error(f"Failed to mark run_state as failed: {e2}")

                # Log RunFailed event (audit-only, non-fatal)
                if self.event_logger:
                    try:
                        await self.event_logger.log_run_failed(
                            error=error_message_for_job_update[:1000],
                        )
                    except Exception as e2:
                        log.warning(f"Failed to log RunFailed event (non-fatal): {e2}")

                # Always post the error JobUpdate
                if send_job_updates:
                    context.post_message(
                        JobUpdate(
                            job_id=self.job_id,
                            status="error",
                            workflow_id=context.workflow_id,
                            error=error_message_for_job_update[:1000],
                        )
                    )
                raise  # Re-raise the exception to be caught by the caller (e.g., pytest.raises)
            finally:
                # This block executes whether an exception occurred or not.
                log.info(f"Finalizing nodes for job {self.job_id} in finally block")

                # Stop StateManager and flush pending updates
                if self.state_manager:
                    try:
                        await self.state_manager.stop(timeout=10.0)
                        log.info(f"StateManager stopped for run {self.job_id}")
                    except Exception as e:
                        log.error(f"Error stopping StateManager: {e}")

                # Stop EventLogger and flush remaining events
                if self.event_logger:
                    try:
                        await self.event_logger.stop()
                        log.info(f"EventLogger stopped for run {self.job_id}")
                    except Exception as e:
                        log.warning(f"Error stopping EventLogger (non-fatal): {e}")

                # Stop input dispatcher if running
                try:
                    if self._input_queue is not None:
                        await self._input_queue.put({"op": "shutdown"})
                    if self._input_task is not None:
                        await self._input_task
                except Exception as e:
                    log.debug(f"Error stopping input dispatcher: {e}")
                if graph and graph.nodes:  # graph is the internal Graph instance from the start of run
                    for node in graph.nodes:
                        try:
                            await node.finalize(context)
                        except Exception as e:
                            log.debug(
                                "Finalize failed for node %s (%s): %s",
                                node.get_title(),
                                node._id,
                                e,
                                exc_info=True,
                            )
                        inbox = self.node_inboxes.get(node._id)
                        if inbox is not None:
                            try:
                                await inbox.close_all()
                            except Exception as e:
                                log.debug(f"Error closing inbox for node {node._id}: {e}")
                log.debug("Nodes finalized in finally block.")

                # Ensure downstream consumers mark all edges as drained as part of teardown
                self.drain_active_edges(context, graph)
                self._torch_support.empty_cuda_cache()
                log.debug("CUDA cache emptied if available.")

                # Clear memory URI cache to free up RAM from images/audio stored during workflow
                log_memory(f"WorkflowRunner.run cleanup START job_id={self.job_id}")
                cache_cleared = clear_memory_uri_cache(log_stats=True)
                log.info(f"Cleared {cache_cleared} items from memory URI cache")

                # Run garbage collection to free unreferenced objects
                run_gc(f"WorkflowRunner.run cleanup job_id={self.job_id}", log_before_after=True)

                # Log final memory state
                log_memory_summary(f"WorkflowRunner.run END job_id={self.job_id}")

                # No legacy generator state to clear in actor mode

            # This part is reached ONLY IF no exception propagated from the try-except block.
            # If an exception was raised and re-thrown by the 'except' block, execution does not reach here.
            if self.status == "completed":
                total_time = time.time() - start_time
                log.info(f"Job {self.job_id} completed successfully")
                log.info(f"Finished job {self.job_id} - Total time: {total_time:.2f} seconds")

                # Log RunCompleted event (audit-only, non-fatal)
                if self.event_logger:
                    try:
                        await self.event_logger.log_run_completed(
                            outputs=self.outputs,
                            duration_ms=int(total_time * 1000),
                        )
                    except Exception as e:
                        log.warning(f"Failed to log RunCompleted event (non-fatal): {e}")

                # Note: JobUpdate(status="completed") is sent in the try block before finally
                # to ensure it's received before the WebSocket closes
            # If self.status became "error" and the exception was re-raised, we don't reach here.

    async def validate_graph(self, context: ProcessingContext, graph: Graph):
        """
        Validates all nodes and their connections within the graph.

        Each node is validated for its required inputs based on its defined input slots
        and incoming edges. Errors found during validation are posted as `NodeUpdate` messages.

        Args:
            context (ProcessingContext): The processing context, used for posting error messages.
            graph (Graph): The directed acyclic graph of nodes to be validated.

        Raises:
            ValueError: If the graph contains validation errors. The error message will
                        summarize the issues found.
        """
        log.info("Validating graph - %d nodes, %d edges", len(graph.nodes), len(graph.edges))
        is_valid = True
        all_errors = []

        # First validate node inputs
        for node in graph.nodes:
            input_edges = [edge for edge in graph.edges if edge.target == node.id]
            log.debug("Validating node %s", node.get_title())
            errors = node.validate_inputs(input_edges)
            if len(errors) > 0:
                is_valid = False
                for e in errors:
                    log.debug(f"Node error: {e}")
                    all_errors.append(e)
                    context.post_message(
                        NodeUpdate(
                            node_id=node.id,
                            node_name=node.get_title(),
                            node_type=node.get_node_type(),
                            status="error",
                            error=str(e),
                        )
                    )

        # Now validate edge type compatibility
        edge_errors = graph.validate_edge_types()
        if edge_errors:
            is_valid = False
            for error in edge_errors:
                log.debug(f"Edge error: {error}")
                all_errors.append(error)
                # Extract node_id from error message if possible
                node_id = error.split(":")[0] if ":" in error else None
                if node_id and graph.find_node(node_id):
                    node = graph.find_node(node_id)
                    context.post_message(
                        NodeUpdate(
                            node_id=node_id,
                            node_name=node.get_title() if node else node_id,
                            node_type=node.get_node_type() if node else node_id,
                            status="error",
                            error=error,
                        )
                    )

        if not is_valid:
            log.debug("Graph validation failed")
            raise ValueError("Graph contains errors: " + "\n".join(all_errors))
        log.info("Graph validation successful")

    async def initialize_graph(self, context: ProcessingContext, graph: Graph):
        """
        Initializes all nodes in the graph by calling their `initialize` method.

        Node initialization can involve setting up internal state, loading resources,
        or other preparatory steps before processing begins.

        Args:
            context (ProcessingContext): The processing context, passed to each node's
                                     `initialize` method and used for posting errors.
            graph (Graph): The directed acyclic graph of nodes to be initialized.

        Raises:
            Exception: Any exception raised during a node's `initialize` method is caught,
                       logged, reported via a `NodeUpdate`, and then re-raised to halt
                       graph processing.
        """
        log.debug("Initializing graph with %d nodes", len(graph.nodes))
        for node in graph.nodes:
            try:
                log.debug(f"Initializing node: {node.get_title()} ({node.id})")
                await node.initialize(context)
            except Exception as e:
                log.error(f"Error initializing node {node.get_title()} ({node.id}): {str(e)}")
                context.post_message(
                    NodeUpdate(
                        node_id=node.id,
                        node_name=node.get_title(),
                        node_type=node.get_node_type(),
                        status="error",
                        error=str(e)[:1000],
                    )
                )
                raise
        log.debug(f"Edges: {graph.edges}")
        log.debug("Graph initialization completed")

    async def send_messages(self, node: BaseNode, result: dict[str, Any], context: ProcessingContext):
        """
        Sends messages from a completed node or streaming node to connected target nodes.

        Args:
            node (BaseNode): The source node that has produced the results.
            result (dict[str, Any]): A dictionary where keys are output slot names
                                     (handles) and values are the data to be sent.
            context (ProcessingContext): The processing context, containing the graph
                                     to find target nodes and edges.
        """
        for key, value_to_send in result.items():
            if not node.should_route_output(key):
                log.debug(
                    "Routing suppressed by node hook for output '%s' on node %s",
                    key,
                    node.id,
                )
                continue
            # find edges from node.id and this specific output slot (key)
            outgoing_edges = context.graph.find_edges(node.id, key)
            for edge in outgoing_edges:
                # Deliver to inboxes for streaming-input consumers
                inbox = self.node_inboxes.get(edge.target)
                if inbox is not None:
                    await inbox.put(edge.targetHandle, value_to_send)
                edge_id = edge.id or ""
                self._edge_counters[edge_id] += 1
                context.post_message(
                    EdgeUpdate(
                        workflow_id=context.workflow_id,
                        edge_id=edge_id,
                        status="message_sent",
                        counter=self._edge_counters[edge_id],
                    )
                )

    def _initialize_inboxes(self, context: ProcessingContext, graph: Graph) -> None:
        """Build and attach `NodeInbox` instances for each node based on graph topology."""
        self.node_inboxes.clear()
        # Pre-compute upstream counts per (node_id, handle)
        upstream_counts: dict[tuple[str, str], int] = {}
        for edge in graph.edges:
            key = (edge.target, edge.targetHandle)
            upstream_counts[key] = upstream_counts.get(key, 0) + 1

        for node in graph.nodes:
            inbox = NodeInbox(buffer_limit=self.buffer_limit)
            # Attach per-handle upstream counts
            # Only handles with at least one upstream are registered; others remain implicit
            for (target_id, handle), count in upstream_counts.items():
                if target_id == node._id:
                    inbox.add_upstream(handle, count)
            self.node_inboxes[node._id] = inbox
            node.attach_inbox(inbox)

    def drain_active_edges(self, context: ProcessingContext, graph: Graph) -> None:
        """Post a drained update for any edge with pending or open input.

        Inspects each edge's target node inbox and posts an ``EdgeUpdate`` with
        status ``"drained"`` for edges whose target handle either still has
        buffered items or open upstream sources. This is intended for workflow
        teardown (completed, cancelled, or error) to ensure front‑end consumers
        stop listening to streams and clear any spinners.

        Args:
            context: The processing context used to post messages.
            graph: The workflow graph containing edges to inspect.

        Returns:
            None
        """
        if not graph or not graph.edges:
            return
        for edge in graph.edges:
            try:
                inbox = self.node_inboxes.get(edge.target)
                if inbox is None:
                    continue
                if (inbox.has_buffered(edge.targetHandle) or inbox.is_open(edge.targetHandle)) and edge.id:
                    context.post_message(EdgeUpdate(workflow_id=context.workflow_id, edge_id=edge.id, status="drained"))
            except Exception:
                # Best effort - ignore errors during draining
                pass

    async def process_graph(self, context: ProcessingContext, graph: Graph, parent_id: str | None = None) -> None:
        """Actor-based processing: start one actor per node and await completion.

        OutputNodes are not driven by actors (outputs are captured in send_messages).

        Completion Detection:
        Due to the streaming nature of the workflow, completion is determined by:
        1. All actor tasks have finished (either successfully or with exceptions)
        2. All node inboxes are fully drained (no pending messages, no open sources)
        3. The message queue has been processed

        This ensures that workflows don't hang due to race conditions between task
        completion and message delivery.
        """
        from nodetool.models.condition_builder import Field
        from nodetool.models.run_node_state import RunNodeState
        from nodetool.workflows.actor import NodeActor

        log.info(
            "Processing graph (%d nodes, %d edges)",
            len(graph.nodes),
            len(graph.edges),
        )

        # Load existing node states for resumption
        node_states = {}
        log.info(
            f"Checking for existing node states for run {self.job_id} (status={self.run_state.status if self.run_state else 'None'})"
        )
        if self.run_state and self.run_state.status in ["suspended", "running"]:
            try:
                # Query all node states for this run
                condition = Field("run_id") == self.job_id
                states, _ = await RunNodeState.query(condition=condition)
                for state in states:
                    node_states[state.node_id] = state
                log.info(f"Loaded {len(node_states)} existing node states for resumption: {list(node_states.keys())}")
                for node_id, state in node_states.items():
                    log.info(f"  Node {node_id}: status={state.status}")
            except Exception as e:
                log.warning(f"Failed to load node states for resumption: {e}")

        tasks = []
        task_to_node: dict[asyncio.Task, str] = {}  # Map tasks to node IDs for debugging
        for node in graph.nodes:
            inbox = self.node_inboxes.get(node._id)
            assert inbox is not None, f"No inbox found for node {node._id}"

            # Skip InputNodes - driven externally
            try:
                if isinstance(node, InputNode):
                    continue
            except Exception:
                pass

            # Check if node was already completed
            node_state = node_states.get(node._id)
            log.info(
                f"Processing node {node._id}: type={node.get_node_type()}, state={node_state.status if node_state else 'not found'}"
            )
            if node_state and node_state.status == "completed":
                log.info(f"Skipping already completed node: {node._id} ({node.get_node_type()})")
                continue

            # Restore resuming state for suspended nodes
            if node_state and node_state.status == "suspended":
                if hasattr(node, "_set_resuming_state") and node_state.resume_state_json:
                    try:
                        node._set_resuming_state(node_state.resume_state_json, 0)  # type: ignore[call-non-callable]
                        log.info(
                            f"Restored resuming state for suspended node: {node._id} "
                            f"(state_keys={list(node_state.resume_state_json.keys())})"
                        )
                    except Exception as e:
                        log.error(f"Failed to restore resuming state for node {node._id}: {e}")

            actor = NodeActor(self, node, context, inbox)
            task = asyncio.create_task(actor.run())
            tasks.append(task)
            task_to_node[task] = node._id

        # Smart wait loop:
        # - If WorkflowSuspendedException occurs, cancel all other tasks and exit immediately (Fast Suspend).
        # - If other exceptions occur, wait for all tasks to finish (standard Gather behavior).
        pending = set(tasks)
        log.info(f"Starting process_graph wait loop with {len(pending)} tasks")
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_EXCEPTION)
            log.info(f"Wait loop iteration: {len(done)} done, {len(pending)} pending")
            for t in done:
                node_id = task_to_node.get(t, "unknown")
                # Check for suspension
                if t.exception():
                    log.info(
                        f"Task for node {node_id} finished with exception: {type(t.exception())} - {t.exception()}"
                    )
                    if isinstance(t.exception(), WorkflowSuspendedException):
                        exc = t.exception()
                        log.info(
                            f"Detected WorkflowSuspendedException in task for node {node_id}. Cancelling {len(pending)} pending tasks."
                        )
                        # Cancel remaining tasks
                        for p in pending:
                            p.cancel()
                        # Wait for clean cancellation
                        if pending:
                            log.info("Waiting for cancellations to complete...")
                            await asyncio.gather(*pending, return_exceptions=True)
                            log.info("Cancellations complete.")
                        raise exc
                else:
                    log.debug(f"Task for node {node_id} completed successfully")

        log.info("All actor tasks completed. Verifying completion state...")

        # Verify all inboxes are fully drained
        # This is critical for correct completion detection in streaming workflows
        inboxes_with_pending = self._check_pending_inbox_work(graph)
        if inboxes_with_pending:
            log.warning(
                f"Detected {len(inboxes_with_pending)} inboxes with pending work after tasks completed: {inboxes_with_pending}"
            )
            # Give a brief moment for any in-flight messages to settle
            # This handles race conditions where EOS signals are being processed
            await asyncio.sleep(COMPLETION_CHECK_DELAY)
            # Re-check after brief delay
            inboxes_with_pending = self._check_pending_inbox_work(graph)
            if inboxes_with_pending:
                log.warning(
                    f"Still have {len(inboxes_with_pending)} inboxes with pending work after delay: {inboxes_with_pending}"
                )

        # Propagate first error if any (preserving original priority)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        first_error: Exception | None = None
        for r in results:
            if isinstance(r, Exception) and first_error is None:
                first_error = r
        if first_error is not None:
            log.error(f"Propagating first error from results: {first_error}")
            raise first_error

        log.info("process_graph finished successfully (no errors/suspensions).")

    def _check_pending_inbox_work(self, graph: Graph) -> list[str]:
        """Check all inboxes for pending work and return list of node IDs with pending messages.

        This method is used to verify that all streaming work has completed before
        considering the workflow done. It helps detect race conditions where tasks
        complete but messages are still being processed.

        Returns:
            List of node IDs that have inboxes with pending work (buffered items or open sources).
        """
        pending_nodes = []
        for node in graph.nodes:
            inbox = self.node_inboxes.get(node._id)
            if inbox is not None and inbox.has_pending_work():
                pending_nodes.append(node._id)
                # Log detailed state for debugging
                log.debug(
                    f"Inbox for node {node._id} has pending work: "
                    f"has_any={inbox.has_any()}, "
                    f"buffers={[(h, len(b)) for h, b in inbox._buffers.items() if len(b) > 0]}, "
                    f"open_counts={[(h, c) for h, c in inbox._open_counts.items() if c > 0]}"
                )
        return pending_nodes

    def log_vram_usage(self, message=""):
        """
        Logs the current VRAM usage on the primary CUDA device, if available.

        The VRAM usage is reported in Gigabytes (GB).
        If Torch is not available or no CUDA device is present, this method does nothing.

        Args:
            message (str, optional): A prefix message to include in the log output.
                                     Defaults to "".
        """
        self._torch_support.log_vram_usage(self, message)

    @contextmanager
    def torch_context(self, context: ProcessingContext):
        """
        A context manager for setting up and tearing down the PyTorch/GPU environment
        for a workflow run.

        Args:
            context (ProcessingContext): The processing context, used by the ComfyUI
                                     progress hook if active.

        Yields:
            None: The context manager yields control to the enclosed block of code.
        """
        with self._torch_support.torch_context(self, context):
            yield

    async def process_output_node(
        self,
        context: ProcessingContext,
        node: OutputNode,
        inputs: dict[str, Any],
    ):
        """
        Processes an `OutputNode` in the workflow graph.

        This method handles the specific processing logic for output nodes, which
        implement the `process` method.
        """
        log.debug("Processing OutputNode inputs: %s", list(inputs.keys()))
        value = None
        if "value" in inputs:
            value = inputs["value"]
        elif inputs:
            # Fallback: use the first provided handle value (common for auto-generated output nodes)
            value = next(iter(inputs.values()))

        if value is not None:
            # Emit a running update for OutputNode for consistency with other nodes
            await node.send_update(context, "running", properties=["name"])
            # Get the type of the output for metadata purposes
            output_type = node.__class__.__name__.replace("Output", "").lower()
            value = await context.normalize_output_value(value)

            if node.name in self.outputs:
                if self.outputs[node.name] and self.outputs[node.name][-1] == value:
                    pass
                else:
                    self.outputs[node.name].append(value)
            else:
                self.outputs[node.name] = [value]

            # Send the new OutputUpdate message
            context.post_message(
                OutputUpdate(
                    node_id=node.id,
                    node_name=node.get_title(),
                    output_name=node.name,
                    value=value,
                    output_type=output_type,
                )
            )
            # Emit a completed NodeUpdate including the value in the result
            await node.send_update(
                context,
                "completed",
                result={"value": value},
                properties=["name"],
            )
        else:
            # This case should ideally not happen if graph is validated.
            log.warning(f"OutputNode {node.name} ({node._id}) received no 'value' in inputs.")
            # Still send a completed update, but with no result value for this path.
            await node.send_update(context, "completed", result={}, properties=["name"])

    async def process_with_gpu(self, context: ProcessingContext, node: BaseNode, retries: int = 0):
        """
        Processes a node with GPU, with retry logic for CUDA OOM errors.
        """
        log.debug(f"process_with_gpu called for node: {node.get_title()} ({node._id}), retries: {retries}")
        return await self._torch_support.process_with_gpu(self, context, node, retries)
