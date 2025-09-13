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
from contextlib import contextmanager
import gc
from nodetool.config.logging_config import get_logger
import random
import time
import threading
from typing import Any, AsyncGenerator, Optional
from collections import deque

from nodetool.ml.core.model_manager import ModelManager
from nodetool.types.job import JobUpdate
from nodetool.workflows.base_node import (
    BaseNode,
    InputNode,
    OutputNode,
)
from nodetool.workflows.types import EdgeUpdate, NodeProgress, NodeUpdate, OutputUpdate
from nodetool.metadata.types import MessageTextContent
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.config.environment import Environment
from nodetool.workflows.graph import Graph
from nodetool.workflows.inbox import NodeInbox

# Optional dependencies check
TORCH_AVAILABLE = False
COMFY_AVAILABLE = False
try:
    import torch

    TORCH_AVAILABLE = True
    try:
        import comfy  # type: ignore
        import comfy.utils  # type: ignore
        import comfy.model_management  # type: ignore

        COMFY_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    pass

log = get_logger(__name__)
# Log level is controlled by env (DEBUG/NODETOOL_LOG_LEVEL)

MAX_RETRIES = 2
BASE_DELAY = 1  # seconds
MAX_DELAY = 60  # seconds


# Define a global GPU lock
gpu_lock = asyncio.Lock()


async def acquire_gpu_lock(node: BaseNode, context: ProcessingContext):
    """
    Asynchronously acquires the global GPU lock for a given node.

    If the lock is currently held, this function will send a "waiting"
    status update for the node before attempting to acquire the lock.
    This function wraps the `gpu_lock.acquire()` call.

    Args:
        node (BaseNode): The node attempting to acquire the GPU lock.
        context (ProcessingContext): The processing context, used for sending updates.
    """
    if gpu_lock.locked():  # Check if the lock is currently held by another coroutine
        log.debug(
            f"Node {node.get_title()} is waiting for GPU lock as it is currently held."
        )
        await node.send_update(context, status="waiting")
    # The acquire call itself will ensure FIFO waiting if the lock is contended.
    await gpu_lock.acquire()
    log.debug(f"Node {node.get_title()} acquired GPU lock")


def release_gpu_lock():
    """
    Releases the global GPU lock.

    This function is a simple wrapper around `gpu_lock.release()`.
    """
    log.debug("Releasing GPU lock from node")
    gpu_lock.release()


def get_available_vram():
    """
    Gets the available VRAM on the primary CUDA device.

    Returns:
        int: The available VRAM in bytes. Returns 0 if Torch is not available
             or no CUDA device is present.
    """
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.cuda.get_device_properties(
            0
        ).total_memory - torch.cuda.memory_allocated(0)
    return 0


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
        node_inboxes: Per-node inboxes for input delivery and EOS tracking.
    """

    def __init__(
        self,
        job_id: str,
        device: str | None = None,
        disable_caching: bool = False,
    ):
        """
        Initializes a new WorkflowRunner instance.

        Args:
            job_id (str): Unique identifier for this workflow execution.
            device (Optional[str]): The specific device ("cpu", "cuda", "mps") to run the workflow on.
                                    If None, it auto-detects based on Torch availability (CUDA, then MPS, then CPU).
        """
        self.job_id = job_id
        self.status = "running"
        self.current_node: Optional[str] = None
        self.context: Optional[ProcessingContext] = None
        self.outputs: dict[str, Any] = {}
        self.active_processing_node_ids: set[str] = (
            set()
        )  # Track nodes currently in an async task
        self.node_inboxes: dict[str, NodeInbox] = {}
        self.disable_caching = disable_caching
        if device:
            self.device = device
        else:
            self.device = "cpu"
            if TORCH_AVAILABLE:
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    self.device = "mps"

            log.info(f"Workflow runs on device: {self.device}")
            log.debug(
                f"WorkflowRunner initialized for job_id: {self.job_id} with device: {self.device}"
            )
        # Streaming input queue and dispatcher task (created during run())
        self._input_queue: asyncio.Queue | None = None
        self._input_task: asyncio.Task | None = None
        # Event loop where the runner is executing; used for thread-safe enqueues
        self._runner_loop: asyncio.AbstractEventLoop | None = None

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
        if loop is not None:
            try:
                try:
                    op = event.get("op")
                    inp = event.get("input")
                    handle = event.get("handle")
                    log.debug(
                        f"Enqueue (thread-safe) input event: op={op} input={inp} handle={handle} current_thread={threading.get_ident()} loop_id={id(loop)}"
                    )
                except Exception:
                    pass
                loop.call_soon_threadsafe(self._input_queue.put_nowait, event)
                return
            except Exception:
                # Fallback to direct put; best-effort if loop reference is stale
                log.debug(
                    "call_soon_threadsafe failed; falling back to direct queue put",
                    exc_info=True,
                )
        try:
            op = event.get("op")
            inp = event.get("input")
            handle = event.get("handle")
            log.debug(
                f"Enqueue (direct) input event: op={op} input={inp} handle={handle} (no runner loop)"
            )
        except Exception:
            pass
        self._input_queue.put_nowait(event)

    def _find_input_node_id(self, context: ProcessingContext, input_name: str) -> str:
        assert context.graph is not None, "Graph not set in context"
        for node in context.graph.inputs():
            if getattr(node, "name", None) == input_name:
                return node._id
        raise ValueError(f"Input node not found for input name: {input_name}")

    def push_input_value(
        self, *, input_name: str, value: Any, source_handle: str | None = None
    ) -> None:
        """
        Enqueue a streaming input event to be dispatched on the runner loop.
        """
        if self._input_queue is None:
            raise RuntimeError("Input queue is not initialized")
        event = {
            "op": "push",
            "input": input_name,
            "value": value,
            "handle": source_handle,
        }
        log.debug(
            f"Enqueue input push: op:{event['op']}, input:{event['input']}, handle:{event['handle']}"
        )
        self._enqueue_input_event(event)

    def finish_input_stream(
        self, *, input_name: str, source_handle: str | None = None
    ) -> None:
        """
        Signal end-of-stream for a streaming input. This marks downstream inboxes as
        done for the corresponding target handles so consumers can complete.
        """
        if self._input_queue is None:
            raise RuntimeError("Input queue is not initialized")
        event = {"op": "end", "input": input_name, "handle": source_handle}
        log.debug(
            f"Enqueue input end: op:{event['op']}, input:{event['input']}, handle:{event['handle']}"
        )
        self._enqueue_input_event(event)

    async def _dispatch_inputs(self, context: ProcessingContext) -> None:
        assert self._input_queue is not None
        assert context.graph is not None
        graph = context.graph
        try:
            loop = asyncio.get_running_loop()
            log.debug(
                f"Input dispatcher started: loop_id={id(loop)} queue_id={id(self._input_queue)}"
            )
        except Exception:
            log.debug("Input dispatcher started (loop id unavailable)")
        while True:
            ev = await self._input_queue.get()
            log.debug(
                f"Dispatch input event: op:{ev.get('op')}, input:{ev.get('input')}, handle:{ev.get('handle')}"
            )
            if ev.get("op") == "shutdown":
                log.debug("Input dispatcher received shutdown; exiting")
                return
            try:
                input_name: str = ev.get("input")
                node_id = self._find_input_node_id(context, input_name)
                node = graph.find_node(node_id)
                if node is None:
                    log.warning(
                        f"Dispatch event dropped: input node not found for {input_name}"
                    )
                    continue
                # Determine output handle from event or InputNode defaults
                handle = ev.get("handle")
                if ev.get("op") == "push":
                    value = ev.get("value")

                    for edge in graph.find_edges(node_id, handle):
                        inbox = self.node_inboxes.get(edge.target)
                        if inbox is not None:
                            inbox.put(edge.targetHandle, value)
                            context.post_message(
                                EdgeUpdate(edge_id=edge.id or "", status="message_sent")
                            )
                        else:
                            log.debug(
                                f"No inbox for target {edge.target} on edge {edge.id}"
                            )
                elif ev.get("op") == "end":
                    for edge in graph.find_edges(node_id, handle):
                        inbox = self.node_inboxes.get(edge.target)
                        if inbox is not None:
                            inbox.mark_source_done(edge.targetHandle)
                            context.post_message(
                                EdgeUpdate(edge_id=edge.id or "", status="drained")
                            )
                        else:
                            log.debug(
                                f"No inbox for target {edge.target} on edge {edge.id}"
                            )
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
            # 1 / 2 – both nodes must exist
            source_node = graph.find_node(edge.source)
            target_node = graph.find_node(edge.target)
            if source_node is None or target_node is None:
                removed.append(edge.id or "<unknown>")
                continue

            source_cls = source_node.__class__
            target_cls = target_node.__class__

            # 3 – source handle must be an output on the *source* node (instance-aware)
            if source_node.find_output_instance(edge.sourceHandle) is None:  # type: ignore
                removed.append(edge.id or "<unknown>")
                continue

            # 4 – target property must exist unless node is dynamic
            if (
                not target_cls.is_dynamic()
                and target_node.find_property(edge.targetHandle) is None
            ):
                removed.append(edge.id or "<unknown>")
                continue

            # Edge passed all checks – keep it
            valid_edges.append(edge)

        # Save removed edge IDs for potential teardown notifications
        try:
            self._removed_edge_ids = removed
        except Exception:
            pass

        if removed:
            log.warning(
                "Filtering %d invalid edge(s) from workflow: %s",
                len(removed),
                ", ".join(removed),
            )

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
                        between input parameters and graph input nodes, or if a ChatInput node
                        is required but not found for `req.messages`.
            Exception: Propagates exceptions from graph processing, including CUDA OOM errors
                       if they persist after retries.

        Post-conditions:
            - Updates workflow status to "completed", "cancelled", or "error".
            - Posts a final JobUpdate message with results or error information.
        """
        log.info(f"Starting workflow execution for job_id: {self.job_id}")
        log.debug(
            "Run parameters: params=%s messages=%s", request.params, request.messages
        )
        log.debug(
            f"WorkflowRunner.run called for job_id: {self.job_id} with req: {request}, context: {context}"
        )

        Environment.load_settings()

        assert request.graph is not None, "Graph is required"

        graph = Graph.from_dict(request.graph.model_dump())
        self._filter_invalid_edges(graph)

        log.info(
            "Graph prepared: %d nodes, %d valid edges after filtering",
            len(graph.nodes),
            len(graph.edges),
        )

        context.graph = graph
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
            raise ValueError(
                f"All InputNode(s) must have a non-empty name. Invalid: {', '.join(invalid_inputs)}"
            )

        input_nodes = {node.name: node for node in graph.inputs()}

        start_time = time.time()
        if send_job_updates:
            context.post_message(JobUpdate(job_id=self.job_id, status="running"))

        with self.torch_context(context):
            try:
                if request.params:
                    for key, value in request.params.items():
                        if key not in input_nodes:
                            raise ValueError(f"No input node found for param: {key}")

                        node = input_nodes[key]
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
                        # push value; end stream immediately if not streaming
                        self.push_input_value(
                            input_name=getattr(node, "name", key), value=value
                        )
                        if not node.is_streaming_output():
                            # default: treat as non-streaming and end
                            self.finish_input_stream(
                                input_name=getattr(node, "name", key)
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
                    self.push_input_value(input_name=name, value=default_value)
                    if not node.is_streaming_output():
                        self.finish_input_stream(input_name=name)

                await self.process_graph(context, graph)

                # If we reach here, no exceptions from the main processing stages
                if (
                    self.status == "running"
                ):  # Check if it wasn't set to error by some internal logic
                    self.status = "completed"

            except asyncio.CancelledError:
                # Gracefully handle external cancellation.
                # We do not emit synthetic per-edge "drained" UI messages.
                self.status = "cancelled"
                if send_job_updates:
                    context.post_message(
                        JobUpdate(job_id=self.job_id, status="cancelled")
                    )
            except Exception as e:
                error_message_for_job_update = str(e)
                log.error(
                    f"Error during graph execution for job {self.job_id}: {error_message_for_job_update}"
                )
                log.debug(f"Exception caught in WorkflowRunner.run: {e}", exc_info=True)

                # Specific handling for OOM error message, but status is always error
                if TORCH_AVAILABLE and isinstance(e, torch.cuda.OutOfMemoryError):
                    error_message_for_job_update = f"VRAM OOM error: {str(e)}. No additional VRAM available after retries."
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
                if (
                    graph and graph.nodes
                ):  # graph is the internal Graph instance from the start of run
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
                            except Exception:
                                pass
                log.debug("Nodes finalized in finally block.")

                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                log.debug("CUDA cache emptied if available.")

                # No legacy generator state to clear in actor mode

            # This part is reached ONLY IF no exception propagated from the try-except block.
            # If an exception was raised and re-thrown by the 'except' block, execution does not reach here.
            if self.status == "completed":
                total_time = time.time() - start_time
                log.info(
                    f"Job {self.job_id} completed successfully (post-try-finally processing)"
                )
                log.info(
                    f"Finished job {self.job_id} - Total time: {total_time:.2f} seconds"
                )
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
        log.info(
            "Validating graph – %d nodes, %d edges", len(graph.nodes), len(graph.edges)
        )
        log.debug(f"validate_graph called with graph: {graph}")
        is_valid = True
        all_errors = []

        # First validate node inputs
        for node in graph.nodes:
            input_edges = [edge for edge in graph.edges if edge.target == node.id]
            log.debug("Validating node %s", node.get_title())
            errors = node.validate(input_edges)
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
                log.error(
                    f"Error initializing node {node.get_title()} ({node.id}): {str(e)}"
                )
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

    def send_messages(
        self, node: BaseNode, result: dict[str, Any], context: ProcessingContext
    ):
        """
        Sends messages from a completed node or streaming node to connected target nodes.

        Args:
            node (BaseNode): The source node that has produced the results.
            result (dict[str, Any]): A dictionary where keys are output slot names
                                     (handles) and values are the data to be sent.
            context (ProcessingContext): The processing context, containing the graph
                                     to find target nodes and edges.
        """
        log.debug(f"Sending messages from {node.get_title()} ({node.id})")
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
                log.debug(
                    f"Sent message from {node.get_title()} ({node.id}) output '{key}' "
                    f"to {edge.target} input '{edge.targetHandle}'. Value: {str(value_to_send)[:50]}"
                )
                # Deliver to inboxes for streaming-input consumers
                inbox = self.node_inboxes.get(edge.target)
                if inbox is not None:
                    inbox.put(edge.targetHandle, value_to_send)
                context.post_message(
                    EdgeUpdate(
                        edge_id=edge.id or "",
                        status="message_sent",
                    )
                )
        log.debug(f"send_messages finished for node: {node.get_title()} ({node.id})")

    def _initialize_inboxes(self, context: ProcessingContext, graph: Graph) -> None:
        """Build and attach `NodeInbox` instances for each node based on graph topology."""
        self.node_inboxes.clear()
        # Pre-compute upstream counts per (node_id, handle)
        upstream_counts: dict[tuple[str, str], int] = {}
        for edge in graph.edges:
            key = (edge.target, edge.targetHandle)
            upstream_counts[key] = upstream_counts.get(key, 0) + 1

        for node in graph.nodes:
            inbox = NodeInbox()
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
                if inbox.has_buffered(edge.targetHandle) or inbox.is_open(
                    edge.targetHandle
                ):
                    if edge.id:
                        context.post_message(
                            EdgeUpdate(edge_id=edge.id, status="drained")
                        )
            except Exception:
                # Best effort – ignore errors during draining
                pass

    async def process_graph(
        self, context: ProcessingContext, graph: Graph, parent_id: str | None = None
    ) -> None:
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

    async def process_node_with_inputs(
        self, context: ProcessingContext, node: BaseNode, inputs: dict[str, Any]
    ):
        """
        Processes a regular `BaseNode` (i.e., not an `OutputNode`)
        with its resolved inputs.

        This method handles the core execution lifecycle for a standard node:
        1. Assigns input values to the node's properties.
        2. Calls `node.pre_process(context)`.
        3. Checks for cached results if `node.is_cacheable()`.
        4. Determines if GPU is required (`node.requires_gpu()`).
        5. Caches the result if `node.is_cacheable()`.
        6. Sends a "completed" `NodeUpdate` with the result.
        7. Calls `self.send_messages` to propagate the result to downstream nodes.

        Args:
            context (ProcessingContext): The processing context.
            node (BaseNode): The node to process.
            inputs (dict[str, Any]): The input values for the node, keyed by input slot names.

        Raises:
            ValueError: If there's an error assigning an input property to the node.
            RuntimeError: If the node requires GPU but no GPU is available on the runner.
            Exception: Any other exception from the node's processing methods, which is
                       logged and re-raised after posting a `NodeUpdate` with error status.
        """
        log.debug(
            f"process_node_with_inputs called for node: {node.get_title()} ({node._id}), inputs: {list(inputs.keys())}"
        )

        # Assign input values to node properties
        for name, value in inputs.items():
            try:
                error = node.assign_property(name, value)
                if error:
                    log.error(
                        f"Error assigning property {name} to node {node.id}: {error}"
                    )
            except Exception as e:
                log.error(f"Error assigning property {name} to node {node.id}")
                raise ValueError(f"Error assigning property {name}: {str(e)}")

        # Preprocess the node
        log.debug(f"Pre-processing node: {node.get_title()} ({node._id})")
        await node.pre_process(context)

        # Check if the node is cacheable
        if node.is_cacheable() and not self.disable_caching:
            log.debug(f"Checking cache for node: {node.get_title()} ({node._id})")
            cached_result = context.get_cached_result(node)
        else:
            cached_result = None

        if cached_result is not None:
            log.info(f"Using cached result for node: {node.get_title()} ({node._id})")
            result = cached_result
        else:
            # Determine if the node requires GPU processing
            requires_gpu = node.requires_gpu()

            # Dynamic streaming: if upstream is streaming, treat this node as stream-driven
            # and avoid caching side-effects during the run path.
            driven_by_stream = context.graph.has_streaming_upstream(node._id)

            if requires_gpu and self.device == "cpu":
                error_msg = f"Node {node.get_title()} ({node._id}) requires a GPU, but no GPU is available."
                log.error(error_msg)
                raise RuntimeError(error_msg)

            await node.send_update(
                context,
                "running",
                result=None,
            )

            # Prepare unified I/O wrappers
            inbox = self.node_inboxes.get(node._id)
            from nodetool.workflows.io import NodeInputs, NodeOutputs

            outputs_collector = NodeOutputs(self, node, context, capture_only=True)
            node_inputs = NodeInputs(inbox) if inbox is not None else None

            if requires_gpu and self.device != "cpu":
                await acquire_gpu_lock(node, context)
                try:
                    self.log_vram_usage(
                        f"Node {node.get_title()} ({node._id}) VRAM before GPU processing"
                    )
                    await node.preload_model(context)
                    self.log_vram_usage(
                        f"Node {node.get_title()} ({node._id}) VRAM after preload_model"
                    )
                    await node.move_to_device(self.device)
                    self.log_vram_usage(
                        f"Node {node.get_title()} ({node._id}) VRAM after move to {self.device}"
                    )

                    await node.run(context, node_inputs, outputs_collector)  # type: ignore[arg-type]
                    self.log_vram_usage(
                        f"Node {node.get_title()} ({node._id}) VRAM after run completion"
                    )
                finally:
                    await node.move_to_device("cpu")
                    self.log_vram_usage(
                        f"Node {node.get_title()} ({node._id}) VRAM after move to cpu"
                    )
                    release_gpu_lock()
            else:
                await node.preload_model(context)
                await node.run(context, node_inputs, outputs_collector)  # type: ignore[arg-type]

            result = outputs_collector.collected()

            # Cache the result if the node is cacheable
            if (
                node.is_cacheable()
                and not self.disable_caching
                and not driven_by_stream
            ):
                log.debug(f"Caching result for node: {node.get_title()} ({node._id})")
                context.cache_result(node, result)

        # Send completion update and route collected outputs downstream
        await node.send_update(context, "completed", result=result)
        self.send_messages(node, result, context)
        # log.info(
        #     f"{node.get_title()} ({node._id}) processing time: {datetime.now() - started_at}"
        # )

    def log_vram_usage(self, message=""):
        """
        Logs the current VRAM usage on the primary CUDA device, if available.

        The VRAM usage is reported in Gigabytes (GB).
        If Torch is not available or no CUDA device is present, this method does nothing.

        Args:
            message (str, optional): A prefix message to include in the log output.
                                     Defaults to "".
        """
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()
            vram = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024
            log.info(f"{message} VRAM: {vram:.2f} GB")

    @contextmanager
    def torch_context(self, context: ProcessingContext):
        """
        A context manager for setting up and tearing down the PyTorch/GPU environment
        for a workflow run.

        If `COMFY_AVAILABLE` (ComfyUI) is True:
        - Sets a global progress bar hook for ComfyUI operations, which posts
          `NodeProgress` messages using `self.current_node` and the provided `context`.

        If `TORCH_AVAILABLE` and CUDA is available:
        - Logs VRAM usage before entering the `try` block (`"Before workflow"`).
        - In the `finally` block (after the `yield`):
          - Logs VRAM usage again (`"After workflow"`).
          This helps in monitoring VRAM consumption throughout the workflow.

        Args:
            context (ProcessingContext): The processing context, used by the ComfyUI
                                     progress hook if active.

        Yields:
            None: The context manager yields control to the enclosed block of code.
        """
        if COMFY_AVAILABLE:

            def hook(value, total, preview_image):
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    comfy.model_management.throw_exception_if_processing_interrupted()
                context.post_message(
                    NodeProgress(
                        node_id=self.current_node or "",
                        progress=value,
                        total=total,
                    )
                )

            comfy.utils.set_progress_bar_global_hook(hook)

        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.log_vram_usage("Before workflow")

        try:
            yield
        finally:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.log_vram_usage("After workflow")

        log.info("Exiting torch context")

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
        if "value" in inputs:
            value = inputs["value"]
            # Emit a running update for OutputNode for consistency with other nodes
            await node.send_update(context, "running", properties=["name"])
            if node.name in self.outputs:
                if self.outputs[node.name] and self.outputs[node.name][-1] == value:
                    # Skip duplicate
                    pass
                else:
                    self.outputs[node.name].append(value)
            else:
                self.outputs[node.name] = [value]

            # Get the type of the output for metadata purposes
            output_type = node.__class__.__name__.replace("Output", "").lower()
            value = await context.embed_assets_in_data(value)

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
            log.warning(
                f"OutputNode {node.name} ({node._id}) received no 'value' in inputs."
            )
            # Still send a completed update, but with no result value for this path.
            await node.send_update(context, "completed", result={}, properties=["name"])

    async def process_with_gpu(
        self, context: ProcessingContext, node: BaseNode, retries: int = 0
    ):
        """
        Processes a node with GPU, with retry logic for CUDA OOM errors.
        """
        log.debug(
            f"process_with_gpu called for node: {node.get_title()} ({node._id}), retries: {retries}"
        )
        if TORCH_AVAILABLE:
            try:
                if node._requires_grad:
                    return await node.process(context)
                with torch.no_grad():
                    return await node.process(context)
            except Exception as e:
                is_cuda_oom = TORCH_AVAILABLE and isinstance(
                    e, torch.cuda.OutOfMemoryError
                )

                if is_cuda_oom:
                    log.error(
                        f"VRAM OOM error for node {node.get_title()} ({node._id}): {str(e)}"
                    )
                    retries += 1

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        vram_before_cleanup = get_available_vram()
                        log.error(f"VRAM before cleanup: {vram_before_cleanup} GB")

                        ModelManager.clear()
                        gc.collect()

                        if COMFY_AVAILABLE:
                            for (
                                model_loaded
                            ) in comfy.model_management.current_loaded_models:
                                model_loaded.model_unload()

                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()  # Force release of CUDA IPC handles
                        torch.cuda.synchronize()
                        log.error(f"VRAM after cleanup: {get_available_vram()} GB")

                    if retries >= MAX_RETRIES:
                        log.error(
                            f"Max retries ({MAX_RETRIES}) reached for OOM error on node {node.get_title()}. Raising error."
                        )
                        raise  # Re-raise the OOM error if max retries are exhausted.

                    delay = min(
                        BASE_DELAY * (2 ** (retries - 1)) + random.uniform(0, 1),
                        MAX_DELAY,
                    )
                    log.warning(
                        f"VRAM OOM encountered for node {node._id}. Retrying in {delay:.2f} seconds. (Attempt {retries}/{MAX_RETRIES})"
                    )
                    await asyncio.sleep(delay)
                    return await self.process_with_gpu(
                        context, node, retries + 1
                    )  # Recursive call
                else:
                    log.debug(
                        f"Non-OOM error in process_with_gpu for node {node.get_title()}: {e}",
                        exc_info=True,
                    )
                    # For non-OOM errors in non-streaming nodes, process_node_with_inputs handles logging and NodeUpdate.
                    # It then re-raises, so we just re-raise here to exit the retry loop and propagate.
                    raise
        else:
            # This case implies TORCH_AVAILABLE is False.
            # Fallback to regular processing if no GPU capability or torch is not there.
            log.debug(
                f"Torch not available, falling back to regular process for node: {node.get_title()}"
            )
            return await node.process(context)
