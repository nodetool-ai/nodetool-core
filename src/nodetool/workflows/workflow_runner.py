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
import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager, suppress
from typing import Any, Optional

from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.types.graph import Edge
from nodetool.types.job import JobUpdate
from nodetool.workflows.base_node import (
    BaseNode,
    InputNode,
    OutputNode,
)
from nodetool.workflows.graph import Graph
from nodetool.workflows.inbox import NodeInbox
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
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


# Define a process-wide GPU lock that is safe across event loops/threads
gpu_lock = threading.Lock()


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
    """
    if gpu_lock.locked():  # Check if the lock is currently held by another task
        log.debug(f"Node {node.get_title()} is waiting for GPU lock as it is currently held.")
        await node.send_update(context, status="waiting")
    # Acquire the threading lock without blocking the event loop.
    # Use short timeouts so task cancellation does not leak a held lock.
    loop = asyncio.get_running_loop()
    while True:
        acquired = await loop.run_in_executor(None, lambda: gpu_lock.acquire(timeout=0.2))
        if acquired:
            break
        # Yield briefly before retrying to avoid busy-waiting
        await asyncio.sleep(0.05)
    log.debug(f"Node {node.get_title()} acquired GPU lock")


def release_gpu_lock():
    """
    Releases the global GPU lock.

    This function is a simple wrapper around `gpu_lock.release()`.
    """
    log.debug("Releasing GPU lock from node")
    gpu_lock.release()


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
                            context.post_message(EdgeUpdate(edge_id=edge.id or "", status="drained"))
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
        log.info("Starting workflow run: job_id=%s", self.job_id)
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

        start_time = time.time()
        if send_job_updates:
            context.post_message(JobUpdate(job_id=self.job_id, status="running"))

        with self.torch_context(context):
            try:
                if request.params:
                    log.info(f"Processing params: {request.params}")
                    log.info(f"Available input nodes: {list(input_nodes.keys())}")
                    for key, value in request.params.items():
                        log.info(f"Setting input node {key} to {value}")
                        if key not in input_nodes:
                            # Log available nodes again just in case
                            log.error(f"Input node '{key}' not found. Available: {list(input_nodes.keys())}")
                            raise ValueError(f"No input node found for param: {key}")

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
                            raise ValueError(f"No input node found for param: {key}")
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

            except asyncio.CancelledError:
                # Gracefully handle external cancellation.
                # We do not emit synthetic per-edge "drained" UI messages.
                self.status = "cancelled"
                if send_job_updates:
                    context.post_message(JobUpdate(job_id=self.job_id, status="cancelled"))
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
                # Always post the error JobUpdate
                if send_job_updates:
                    context.post_message(
                        JobUpdate(
                            job_id=self.job_id,
                            status="error",
                            error=error_message_for_job_update[:1000],
                        )
                    )
                raise  # Re-raise the exception to be caught by the caller (e.g., pytest.raises)
            finally:
                # This block executes whether an exception occurred or not.
                log.info(f"Finalizing nodes for job {self.job_id} in finally block")
                # Stop input dispatcher if running
                try:
                    if self._input_queue is not None:
                        await self._input_queue.put({"op": "shutdown"})
                    if self._input_task is not None:
                        await self._input_task
                except Exception:
                    pass
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
                            with suppress(Exception):
                                await inbox.close_all()
                log.debug("Nodes finalized in finally block.")

                # Ensure downstream consumers mark all edges as drained as part of teardown
                self.drain_active_edges(context, graph)
                self._torch_support.empty_cuda_cache()
                log.debug("CUDA cache emptied if available.")

                # No legacy generator state to clear in actor mode

            # This part is reached ONLY IF no exception propagated from the try-except block.
            # If an exception was raised and re-thrown by the 'except' block, execution does not reach here.
            if self.status == "completed":
                total_time = time.time() - start_time
                log.info(f"Job {self.job_id} completed successfully (post-try-finally processing)")
                log.info(f"Finished job {self.job_id} - Total time: {total_time:.2f} seconds")
                if send_job_updates:
                    context.post_message(
                        JobUpdate(
                            job_id=self.job_id,
                            status="completed",
                            result=self.outputs,
                            message=f"Workflow {self.job_id} completed in {total_time:.2f} seconds",
                        )
                    )
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
                    context.post_message(EdgeUpdate(edge_id=edge.id, status="drained"))
            except Exception:
                # Best effort - ignore errors during draining
                pass

    async def process_graph(self, context: ProcessingContext, graph: Graph, parent_id: str | None = None) -> None:
        """Actor-based processing: start one actor per node and await completion.

        OutputNodes are not driven by actors (outputs are captured in send_messages).
        """
        from nodetool.workflows.actor import NodeActor

        log.info(
            "Processing graph (%d nodes, %d edges)",
            len(graph.nodes),
            len(graph.edges),
        )
        tasks: list[asyncio.Task] = []
        for node in graph.nodes:
            inbox = self.node_inboxes.get(node._id)
            assert inbox is not None, f"No inbox found for node {node._id}"
            # Skip starting actors for InputNodes; they are driven externally via input queue
            try:
                if isinstance(node, InputNode):
                    continue
            except Exception:
                pass
            actor = NodeActor(self, node, context, inbox)
            tasks.append(asyncio.create_task(actor.run()))
        # Wait for all, propagate first error if any
        results = await asyncio.gather(*tasks, return_exceptions=True)
        first_error: Exception | None = None
        for r in results:
            if isinstance(r, Exception) and first_error is None:
                first_error = r
        if first_error is not None:
            raise first_error

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
