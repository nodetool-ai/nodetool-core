"""
Workflow execution engine for processing directed acyclic graphs (DAGs) of computational nodes.

This module provides the core workflow execution functionality, handling parallel processing,
resource management, and orchestration of computational nodes. It supports both CPU and
GPU-based computations with automatic device selection and memory management.

The primary class, `WorkflowRunner`, manages the execution of a workflow graph defined by
`RunJobRequest` and `ProcessingContext`. It orchestrates node execution, message passing
between nodes, and handles GPU resources via an `OrderedLock`.

Key Components:
    - WorkflowRunner: Main execution engine that processes DAGs of nodes.
    - OrderedLock: A FIFO lock mechanism primarily used for managing access to GPU resources,
                   ensuring that nodes acquire GPU access in a sequential, ordered manner.
    - Message: A Pydantic model representing data passed between nodes.

Core Functionality:
    - Graph Processing: Parses and validates the workflow graph, initializing nodes.
    - Node Execution: Processes individual nodes, handling inputs, outputs, and execution logic.
                       Supports regular nodes, iterator nodes, and output nodes.
    - Parallelism: Executes independent nodes in parallel using `asyncio`.
    - GPU Management:
        - Uses `OrderedLock` to serialize GPU access.
        - Attempts to move models to and from GPU, and to CPU to free VRAM.
        - Implements retry logic for CUDA OutOfMemory errors, including VRAM cleanup attempts.
    - Message Passing: Manages a message queue for inter-node communication.
    - Result Caching: Supports caching of results for cacheable nodes.
    - Error Handling: Captures and reports errors at both node and job levels.
    - Progress Tracking: Sends updates on node and job status.
    - Dynamic Device Selection: Chooses between "cpu", "cuda", or "mps" based on availability.
    - Context Management: Utilizes `torch_context` for managing PyTorch-specific operations,
                          including ComfyUI progress hooks if available.

Example:
    ```python
    from nodetool.workflows.run_job_request import RunJobRequest
    from nodetool.workflows.processing_context import ProcessingContext

    # Assuming req and context are properly initialized
    runner = WorkflowRunner(job_id="unique_job_id")
    await runner.run(req, context)
    ```

Dependencies:
    - Optional: `torch`, `comfy` (for GPU operations and ComfyUI integration)
    - Required: `asyncio`, `pydantic`, `logging`
"""

import asyncio
from contextlib import contextmanager
import gc
import time
from typing import Any, AsyncGenerator, Optional
from collections import deque
import random

# Import anext for Python 3.10+ async generator support
try:
    anext
except NameError:
    # For Python < 3.10, define anext
    async def anext(async_gen):
        return await async_gen.__anext__()


from nodetool.common.model_manager import ModelManager
from nodetool.types.job import JobUpdate
from nodetool.workflows.base_node import (
    BaseNode,
    InputNode,
    OutputNode,
)
from nodetool.workflows.types import NodeProgress, NodeUpdate, OutputUpdate
from nodetool.metadata.types import Event
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.common.environment import Environment
from nodetool.workflows.graph import Graph
from nodetool.types.graph import (
    Graph as APIGraph,
)
from nodetool.types.graph import Node, Edge

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

log = Environment.get_logger()

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
        node.send_update(context, status="waiting")
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
    A workflow execution engine that processes directed acyclic graphs (DAGs) of computational nodes.

    The WorkflowRunner handles the execution of complex workflows by managing node dependencies,
    parallel processing, GPU resource allocation using an `OrderedLock`, and result caching.
    It supports both CPU and GPU-based computations, with automatic device selection
    (CPU, CUDA, MPS) based on availability and node requirements.

    The engine processes a graph by:
    1. Validating the graph structure and node inputs.
    2. Initializing all nodes.
    3. Processing initial nodes (those with no dependencies).
    4. Entering a main loop to manage message passing and execute ready nodes in batches.
    5. Handling node-specific logic, including GPU operations, caching, and retries for OOM errors.

    Attributes:
        job_id (str): Unique identifier for the workflow execution.
        status (str): Current status of the workflow (e.g., "running", "completed", "error").
        current_node (Optional[str]): ID of the node currently being processed (primarily for ComfyUI progress).
        context (Optional[ProcessingContext]): The processing context for the current job.
        outputs (dict[str, Any]): A dictionary to store the final outputs of the workflow,
                                populated by `OutputNode`s.
        device (str): The primary computing device ("cpu", "cuda", "mps") selected for the workflow.
        active_processing_node_ids (set[str]): A set of node IDs currently being processed in an
                                             async task, to prevent re-processing before completion.
        edge_queues (dict[tuple[str, str, str, str], deque[Any]]): A dictionary to store edge queues.
        active_generators (dict[str, tuple[AsyncGenerator, dict[str, Any]]]): Stores (generator_iterator, initial_config_properties)
    """

    def __init__(self, job_id: str, device: str | None = None):
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
        self.edge_queues: dict[tuple[str, str, str, str], deque[Any]] = {}
        self.active_generators: dict[str, tuple[AsyncGenerator, dict[str, Any]]] = (
            {}
        )  # Stores (generator_iterator, initial_config_properties)
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

    def is_running(self) -> bool:
        """
        Checks if the workflow is currently in the "running" state.

        Returns:
            bool: True if the workflow status is "running", False otherwise.
        """
        return self.status == "running"

    def _initialize_edge_queues(self, graph: Graph):
        log.debug("Initializing edge queues for graph with %d edges", len(graph.edges))
        for edge in graph.edges:
            edge_key = (
                edge.source,
                edge.sourceHandle,
                edge.target,
                edge.targetHandle,
            )
            self.edge_queues[edge_key] = deque()
            log.debug("Initialized queue for edge %s", edge_key)
        log.debug("Edge queues initialized: %s", list(self.edge_queues.keys()))

    async def run(
        self,
        req: RunJobRequest,
        context: ProcessingContext,
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
            req (RunJobRequest): Contains the workflow graph, input parameters, and initial messages.
            context (ProcessingContext): Manages the execution state, inter-node communication,
                                     and provides services like caching.

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
        log.debug("Run parameters: params=%s messages=%s", req.params, req.messages)

        Environment.load_settings()

        assert req.graph is not None, "Graph is required"

        self.edge_queues.clear()

        # Load node instances using the context
        loaded_node_instances = context.load_nodes(req.graph.nodes)
        log.debug("Loaded %d node instances", len(loaded_node_instances))

        # Create the internal Graph object with these loaded instances
        graph = Graph(
            nodes=loaded_node_instances,
            edges=req.graph.edges,  # Edges from the original request graph
        )
        context.graph = graph
        self._initialize_edge_queues(graph)
        log.debug("Edge queues after initialization: %s", self.edge_queues)
        self.context = context
        context.device = self.device

        input_nodes = {node.name: node for node in graph.inputs()}

        start_time = time.time()
        context.post_message(JobUpdate(job_id=self.job_id, status="running"))

        if req.params:
            for key, value in req.params.items():
                if key not in input_nodes:
                    raise ValueError(f"No input node found for param: {key}")

                node = input_nodes[key]
                node.assign_property("value", value)

        if req.messages:
            # find chat input node
            chat_input_node = next(
                (
                    node
                    for node in context.graph.nodes
                    if node.get_node_type() == "nodetool.input.ChatInput"
                ),
                None,
            )
            if chat_input_node is None:
                raise ValueError(
                    "Chat input node not found. Make sure you have a ChatInput node in your graph."
                )
            chat_input_node.assign_property("value", req.messages)

        with self.torch_context(context):
            try:
                await self.validate_graph(context, graph)
                await self.initialize_graph(context, graph)
                await self.process_graph(context, graph)

                # If we reach here, no exceptions from the main processing stages
                if (
                    self.status == "running"
                ):  # Check if it wasn't set to error by some internal logic
                    self.status = "completed"

            except Exception as e:
                error_message_for_job_update = str(e)
                log.error(
                    f"Error during graph execution for job {self.job_id}: {error_message_for_job_update}"
                )

                # Specific handling for OOM error message, but status is always error
                if TORCH_AVAILABLE and isinstance(e, torch.cuda.OutOfMemoryError):
                    error_message_for_job_update = f"VRAM OOM error: {str(e)}. No additional VRAM available after retries."
                    # log.error already done by generic message

                self.status = "error"
                # Always post the error JobUpdate
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
                if (
                    graph and graph.nodes
                ):  # graph is the internal Graph instance from the start of run
                    for node in graph.nodes:
                        await node.finalize(context)

                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                self.active_generators.clear()
                log.info(
                    f"Cleared active_generators for job {self.job_id} in finally block"
                )
                log.debug("Final edge queue state: %s", self.edge_queues)

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
                context.post_message(
                    JobUpdate(
                        job_id=self.job_id,
                        status="completed",
                        result=self.outputs,
                        message=f"Workflow {self.job_id} completed in {total_time:.2f} seconds",
                    )
                )
            # If self.status became "error" and the exception was re-raised, we don't reach here.
            # If self.status became "error" due to some internal logic but no exception was re-raised (not current design),
            # then we might reach here with status "error", and no "completed" message would be sent.

        # This log helps understand the ultimate exit status of the run() method itself.
        log.info(
            f"WorkflowRunner.run for job_id: {self.job_id} method ending with status: {self.status}"
        )
        log.debug("Workflow outputs: %s", self.outputs)

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
        log.debug("Validating graph with %d nodes", len(graph.nodes))
        is_valid = True

        for node in graph.nodes:
            input_edges = [edge for edge in graph.edges if edge.target == node.id]
            log.debug("Validating node %s", node.get_title())
            errors = node.validate(input_edges)
            if len(errors) > 0:
                is_valid = False
                for e in errors:
                    context.post_message(
                        NodeUpdate(
                            node_id=node.id,
                            node_name=node.get_title(),
                            status="error",
                            error=str(e),
                        )
                    )
        if not is_valid:
            log.debug("Graph validation failed")
            raise ValueError("Graph contains errors: " + "\n".join(errors))
        log.debug("Graph validation successful")

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
                        status="error",
                        error=str(e)[:1000],
                    )
                )
                raise
        log.debug("Graph initialization completed")

    def send_messages(
        self, node: BaseNode, result: dict[str, Any], context: ProcessingContext
    ):
        """
        Sends messages from a completed node's output slots to connected target nodes.

        For each key-value pair in the `result` dictionary (representing an output slot
        and its value), this method finds all outgoing edges from that slot. If the value
        is an Event object, it is handled specially to trigger immediate processing.
        Otherwise, the value is appended to the `deque` in `self.edge_queues`.

        Args:
            node (BaseNode): The source node that has produced the results.
            result (dict[str, Any]): A dictionary where keys are output slot names
                                     (handles) and values are the data to be sent.
            context (ProcessingContext): The processing context, containing the graph
                                     to find target nodes and edges.
        """
        log.debug(f"Sending messages from {node.get_title()} ({node.id})")
        for key, value_to_send in result.items():
            # find edges from node.id and this specific output slot (key)
            outgoing_edges = context.graph.find_edges(node.id, key)
            for edge in outgoing_edges:
                edge_key = (
                    edge.source,
                    edge.sourceHandle,
                    edge.target,
                    edge.targetHandle,
                )

                if edge_key not in self.edge_queues:
                    log.warning(
                        f"Edge key {edge_key} not found in self.edge_queues. "
                        f"Message from {node.get_title()} ({node.id}) for slot '{key}' not sent."
                    )
                    continue

                self.edge_queues[edge_key].append(value_to_send)
                log.debug(
                    f"Sent message from {node.get_title()} ({node.id}) output '{key}' "
                    f"to {edge.target} input '{edge.targetHandle}' via edge_queue. Value: {str(value_to_send)[:50]}"
                )
        log.debug("Edge queue state after sending messages: %s", self.edge_queues)

    async def _process_trigger_nodes(
        self,
        context: ProcessingContext,
        graph: Graph,
    ):
        """
        Processes trigger nodes in the graph (those with no incoming edges).

        These nodes are typically input nodes.

        Args:
            context (ProcessingContext): The execution context for the workflow,
                                     used for posting updates and errors.
            graph (Graph): The graph of nodes to be processed.

        Raises:
            Exception: If any of the initial nodes raise an exception during their
                       `process_node` call. The first such exception encountered is
                       re-raised after logging and posting error messages.
                       The job status is also set to "error".
        """
        log.debug("Processing trigger nodes")
        initial_processing_tasks = []
        initial_nodes_for_tasks = []

        for node in graph.nodes:
            # A node is a trigger if it has no incoming edges to any of its input handles.
            # These nodes derive their initial values from configuration (e.g. node.data or req.params).
            has_incoming_edges = any(edge.target == node._id for edge in graph.edges)
            if not has_incoming_edges:
                log.debug(
                    f"Queueing initial/trigger node for processing: {node.get_title()} ({node._id})"
                )
                # Inputs for these nodes are typically part of their config, not passed dynamically here.
                initial_processing_tasks.append(self.process_node(context, node, {}))
                initial_nodes_for_tasks.append(node)

        if initial_processing_tasks:
            results = await asyncio.gather(
                *initial_processing_tasks, return_exceptions=True
            )
            for i, (node_obj, result_or_exc) in enumerate(
                zip(initial_nodes_for_tasks, results)
            ):
                if isinstance(result_or_exc, Exception):
                    log.error(
                        f"Error processing initial node {node_obj.get_title()}: {result_or_exc}"
                    )
                    context.post_message(
                        NodeUpdate(
                            node_id=node_obj.id,
                            node_name=node_obj.get_title(),
                            status="error",
                            error=str(result_or_exc)[:1000],
                        )
                    )
                    raise result_or_exc
        log.debug("Trigger node processing complete")

    def _get_ready_nodes_and_prepare_tasks(
        self,
        context: ProcessingContext,
        graph: Graph,
    ) -> tuple[list[tuple[BaseNode, dict[str, Any]]], list[asyncio.Task], bool]:
        """
        Identifies nodes ready for processing and creates `asyncio.Task`s for them.

        A node is considered ready if:
        1. It is not currently in `self.active_processing_node_ids`.
        2. EITHER:
           a. It's a streaming node and `node._id` is in `self.active_generators` (already initialized).
           b. All its required input slots (defined by incoming edges)
              have messages available in the `self.edge_queues`.
           c. An event message is available for any of its input slots (for event handling).

        Nodes identified as ready are added to `self.active_processing_node_ids`.
        For non-streaming nodes, inputs are consumed from `edge_queues`.
        For active streaming nodes, input dict is empty as they self-drive or use stored config.
        For event-driven processing, the event is passed specially.

        Args:
            context (ProcessingContext): The execution context for the workflow.
            graph (Graph): The graph of nodes.

        Returns:
            tuple[list[tuple[BaseNode, dict[str, Any]]], list[asyncio.Task], bool]:
                A tuple containing:
                - `ready_node_task_details_list`: List of `(BaseNode, dict_of_inputs_for_run)`
                  tuples. For active generator nodes, dict is empty. For event processing,
                  dict contains special event marker.
                - `tasks_to_run_this_iteration`: List of `asyncio.Task` objects.
                - `any_progress_potential`: Boolean indicating if any node was scheduled
                                          (either an active generator or a node with inputs).
        """
        log.debug("Scanning graph for ready nodes")
        tasks_to_run_this_iteration = []
        ready_node_task_details_list: list[tuple[BaseNode, dict[str, Any]]] = []
        any_progress_potential = False

        for node in graph.nodes:
            if node._id in self.active_processing_node_ids:
                log.debug(
                    f"Node {node.get_title()} ({node._id}) is already active. Skipping."
                )
                continue

            # Case 1: Node is an active streaming generator, ready to pull next item
            if node.is_streaming_output() and node._id in self.active_generators:
                log.debug(
                    f"Active streaming node {node.get_title()} ({node._id}) is ready to pull next item."
                )
                inputs_for_this_run: dict[str, Any] = (
                    {}
                )  # Inputs are internal to generator
                tasks_to_run_this_iteration.append(
                    self.process_node(context, node, inputs_for_this_run)
                )
                ready_node_task_details_list.append((node, inputs_for_this_run))
                self.active_processing_node_ids.add(node._id)
                any_progress_potential = True
                continue

            # --- NEW: Event Check and Processing ---
            # Check if any input slot has an Event. If so, process immediately with available inputs.
            node_input_handles = {
                edge.targetHandle for edge in graph.edges if edge.target == node._id
            }
            event_detected_on_handle: Optional[str] = None

            for handle_name in node_input_handles:
                for (
                    edge
                ) in (
                    graph.edges
                ):  # Iterate through edges to find the one for current handle_name
                    if edge.target == node._id and edge.targetHandle == handle_name:
                        edge_key = (
                            edge.source,
                            edge.sourceHandle,
                            edge.target,
                            edge.targetHandle,
                        )
                        if edge_key in self.edge_queues and self.edge_queues[edge_key]:
                            # Peek at the first item in the deque for this edge
                            if isinstance(self.edge_queues[edge_key][0], Event):  # type: ignore
                                event_detected_on_handle = handle_name
                                break  # Found an event on this handle for this edge
                if event_detected_on_handle:
                    break  # Found an event for the node (across all its handles)

            if event_detected_on_handle:
                log.info(
                    f"Event detected for node {node.get_title()} on input handle '{event_detected_on_handle}'. Preparing for immediate processing."
                )
                inputs_for_event_node_run: dict[str, Any] = {}

                # Consume the event and any other currently available messages for this node's input handles
                for handle_to_fill in node_input_handles:
                    for (
                        edge_iter
                    ) in (
                        graph.edges
                    ):  # Iterate edges to find those matching current handle_to_fill
                        if (
                            edge_iter.target == node._id
                            and edge_iter.targetHandle == handle_to_fill
                        ):
                            edge_key_consume = (
                                edge_iter.source,
                                edge_iter.sourceHandle,
                                edge_iter.target,
                                edge_iter.targetHandle,
                            )
                            if (
                                edge_key_consume in self.edge_queues
                                and self.edge_queues[edge_key_consume]
                            ):
                                # Message available, consume it
                                item = self.edge_queues[edge_key_consume].popleft()
                                inputs_for_event_node_run[handle_to_fill] = item
                                log.debug(
                                    f"Consumed message for event-triggered node {node.get_title()} input '{handle_to_fill}'. "
                                    f"Queue for edge {edge_key_consume} now has {len(self.edge_queues[edge_key_consume])} items."
                                )
                                break  # Consumed one message for this handle_to_fill, move to next handle_to_fill

                if not inputs_for_event_node_run:
                    # This case implies an event was peeked but not consumed, which shouldn't happen with this logic.
                    log.warning(
                        f"Event was detected for {node.get_title()} on handle '{event_detected_on_handle}', "
                        "but no inputs were consumed. This might indicate an internal logic issue. Skipping node for this cycle."
                    )
                    continue

                tasks_to_run_this_iteration.append(
                    self.process_node(context, node, inputs_for_event_node_run)
                )
                ready_node_task_details_list.append((node, inputs_for_event_node_run))
                self.active_processing_node_ids.add(node._id)
                any_progress_potential = True
                log.debug(
                    f"Node {node.get_title()} ({node._id}) scheduled for event-triggered processing. "
                    f"Inputs provided: {list(inputs_for_event_node_run.keys())}"
                )
                continue  # Crucial: Move to the next node, skipping regular input checks for this one

            # Case 3: Check for regular messages (if not an active streamer or event-triggered)
            required_input_slots = {
                edge.targetHandle for edge in graph.edges if edge.target == node._id
            }

            # If a node has no edge-defined input slots AND it's not an active generator,
            # it's likely a trigger node whose first run is managed by _process_trigger_nodes.
            # It shouldn't be picked up here unless it becomes an active generator.
            if not required_input_slots and not (
                node.is_streaming_output() and node._id in self.active_generators
            ):
                continue

            # --- Peek Phase: Check if all inputs are available ---

            # --- Consume Phase: If all inputs can be satisfied, now consume them ---
            inputs_for_this_run: dict[str, Any] = {}
            messages_consumed_for_this_node = False

            if required_input_slots:  # Only try to consume if there are slots to fill
                # First check if we can satisfy all required input slots in this run
                all_slots_can_be_satisfied = True

                for slot_name in required_input_slots:
                    slot_has_available_data = False
                    for edge in graph.edges:
                        if edge.target == node._id and edge.targetHandle == slot_name:
                            edge_key = (
                                edge.source,
                                edge.sourceHandle,
                                edge.target,
                                edge.targetHandle,
                            )
                            if (
                                edge_key in self.edge_queues
                                and self.edge_queues[edge_key]
                            ):
                                slot_has_available_data = True
                                break

                    if not slot_has_available_data:
                        all_slots_can_be_satisfied = False
                        break

                # Only consume if ALL required slots can be satisfied
                if not all_slots_can_be_satisfied:
                    continue

                # Now actually consume the inputs
                for slot_name in required_input_slots:
                    for edge in graph.edges:
                        if edge.target == node._id and edge.targetHandle == slot_name:
                            edge_key = (
                                edge.source,
                                edge.sourceHandle,
                                edge.target,
                                edge.targetHandle,
                            )
                            if (
                                edge_key in self.edge_queues
                                and self.edge_queues[edge_key]
                            ):
                                item = self.edge_queues[edge_key].popleft()
                                inputs_for_this_run[slot_name] = item
                                messages_consumed_for_this_node = True
                                log.debug(
                                    f"Consumed message for {node.get_title()} slot '{slot_name}' from edge {edge_key}. "
                                    f"Queue for edge {edge_key} now has {len(self.edge_queues[edge_key])} items."
                                )
                                break  # Found an edge and consumed for this slot_name, move to next slot_name

                if (
                    not messages_consumed_for_this_node and required_input_slots
                ):  # If consumption failed for a node that needs inputs
                    continue  # Skip to next node

            # Node is ready if:
            # 1. It's a non-active streaming node and its initial inputs are now consumed (or it's a trigger with no edge inputs).
            # 2. It's a non-streaming node and its inputs are consumed.
            # Active streaming nodes are handled by Case 1 at the start of the loop.

            # This log and task creation applies to:
            # - Non-streaming nodes with satisfied inputs.
            # - Streaming nodes for their *initialization run* if their inputs (if any) are satisfied.
            log.debug(
                f"Node {node.get_title()} ({node._id}) is ready for processing. Inputs provided: {list(inputs_for_this_run.keys())}"
            )
            tasks_to_run_this_iteration.append(
                self.process_node(context, node, inputs_for_this_run)
            )
            ready_node_task_details_list.append((node, inputs_for_this_run))
            self.active_processing_node_ids.add(node._id)
            any_progress_potential = True

        log.debug("Found %d ready nodes", len(ready_node_task_details_list))
        return (
            ready_node_task_details_list,
            tasks_to_run_this_iteration,
            any_progress_potential,
        )

    async def _execute_node_batch(
        self,
        context: ProcessingContext,
        ready_node_task_details_list: list[tuple[BaseNode, dict[str, Any]]],
        tasks_to_run: list[asyncio.Task],
    ) -> bool:
        """
        Executes a batch of node processing tasks concurrently and handles results.

        This method uses `asyncio.gather` to run the provided `tasks_to_run`.
        After completion (or error) of each task:
        - The corresponding node is removed from `self.active_processing_node_ids`.
        - If an error occurred, it's logged and a `NodeUpdate` is posted.
        - The node is added to `processed_nodes` (to mark it as attempted).
        - If successful, the node is added to `processed_nodes`.
        - CRITICALLY, for successful nodes, the inputs that were just consumed for this run
          (from `inputs_that_were_used`) are cleared from the `node_inputs_buffer`.
          If a node's entry in the buffer becomes empty, it's removed.

        Args:
            context (ProcessingContext): The execution context.
            ready_node_task_details_list (list[tuple[BaseNode, dict[str, Any]]]):
                List of `(BaseNode, dict_of_inputs_used)` tuples corresponding to the tasks.
                The `dict_of_inputs_used` is crucial for clearing the correct buffer entries.
            tasks_to_run (list[asyncio.Task]): The list of `asyncio.Task` objects to execute.
            node_inputs_buffer (dict[str, dict[str, Any]]): Buffer of inputs, which will be
                modified by clearing consumed inputs for successful nodes.

        Returns:
            bool: True if any tasks were dispatched and processed (i.e., `tasks_to_run` was not empty),
                  False otherwise.
        """
        log.debug("Executing node batch of size %d", len(tasks_to_run))
        if not tasks_to_run:
            return False

        results = await asyncio.gather(*tasks_to_run, return_exceptions=True)
        executed_something = False

        for i, (node_processed, inputs_that_were_used) in enumerate(
            ready_node_task_details_list
        ):
            executed_something = True
            self.active_processing_node_ids.remove(
                node_processed._id
            )  # Node task finished, remove from active set

            if isinstance(results[i], Exception):
                log.error(
                    f"Error processing node {node_processed.get_title()}: {results[i]}"
                )
                context.post_message(
                    NodeUpdate(
                        node_id=node_processed.id,
                        node_name=node_processed.get_title(),
                        status="error",
                        error=str(results[i])[:1000],
                    )
                )
                raise results[i]  # Propagate the error to halt graph processing

        log.debug("Batch execution completed. Success=%s", executed_something)
        return executed_something

    def _check_loop_termination_conditions(
        self,
        context: ProcessingContext,
        graph: Graph,
        iterations_without_progress: int,
        max_iterations_limit: int,
    ) -> bool:
        # Primary condition for normal termination:
        # If no new data was consumed into nodes AND no nodes actually ran for a few cycles.
        if iterations_without_progress > 2:
            log.info(
                f"System has been idle for {iterations_without_progress} iterations (no new tasks scheduled, "
                f"no tasks completed, no tasks in-flight). Checking for termination."
            )
            log.debug(f"Active processing node IDs: {self.active_processing_node_ids}")
            log.debug(f"Current edge queues: {self.edge_queues}")
            # Check if there's any pending data in any edge queue. If so, it might be a stall.
            # Otherwise, it's a clean completion.
            pending_data_in_queues = False
            for edge_key, queue in self.edge_queues.items():
                if queue:
                    log.warning(
                        f"Graph processing concluding, but edge queue {edge_key} still has {len(queue)} items."
                    )
                    pending_data_in_queues = True

            if pending_data_in_queues:
                log.warning(
                    "Graph processing finished due to inactivity, but some edge queues still contain data. Potential stall or graph logic issue."
                )
                # Optionally, trigger deadlock-like reporting here or set job status to error.
                # For now, let it terminate and the final job status will reflect outputs.
            return True

        return False  # Default: continue loop

    async def _main_processing_loop(
        self,
        context: ProcessingContext,
        graph: Graph,
        parent_id: str | None,  # Used for logging context
    ):
        """
        The main event loop for processing nodes within a graph (or subgraph).

        This loop iteratively performs the following steps until termination conditions are met:
        1. Identifies ready nodes and prepares tasks: Calls `_get_ready_nodes_and_prepare_tasks` to find nodes
           whose input dependencies are met and creates processing tasks for them.
        2. Executes node batch: Calls `_execute_node_batch` to run the tasks for ready nodes
           concurrently, clearing consumed inputs upon success.
        3. Updates progress counter: Resets `iterations_without_progress` if messages were
           consumed or nodes were processed; otherwise, increments it.
        4. Checks termination: Calls `_check_loop_termination_conditions` to see if the
           loop should exit (e.g., due to completion, stall, or max iterations).

        Args:
            context (ProcessingContext): The execution context for the workflow.
            graph (Graph): The graph (or subgraph) of nodes to be processed.
            parent_id (str | None): Optional ID of a parent group node, used primarily for
                                    contextual logging if this is a subgraph execution.
        """
        log.debug(
            "Starting main processing loop for graph with %d nodes", len(graph.nodes)
        )
        iterations_without_progress = 0
        # Heuristic limit: N*3 (3 passes per node for complex message patterns) + buffer
        max_iterations_limit = len(graph.nodes) * 3 + 10

        while True:
            # 1. Identify ready nodes and prepare tasks
            ready_node_task_details_list, tasks_to_run, any_progress_potential = (
                self._get_ready_nodes_and_prepare_tasks(context, graph)
            )

            # 2. Execute the batch of ready nodes
            nodes_were_processed_this_iteration = await self._execute_node_batch(
                context,
                ready_node_task_details_list,
                tasks_to_run,
            )

            # 3. Update progress counter
            # Progress is defined as:
            #   - New tasks were identified and scheduled in this iteration (any_progress_potential), OR
            #   - Tasks from this iteration's batch ran (nodes_were_processed_this_iteration), OR
            #   - There are still tasks from previous iterations that are in-flight (active_processing_node_ids is not empty).
            # If none of these are true, then the system is truly idle for this iteration.
            if (
                nodes_were_processed_this_iteration
                or any_progress_potential
                or self.active_processing_node_ids
            ):
                iterations_without_progress = 0
            else:
                iterations_without_progress += 1
                log.debug(
                    f"System idle this iteration. iterations_without_progress: {iterations_without_progress} for parent_id: {parent_id}"
                )

            # 4. Check termination conditions
            if self._check_loop_termination_conditions(
                context,
                graph,
                iterations_without_progress,
                max_iterations_limit,
            ):
                break

        log.debug("Main processing loop complete")

    async def process_graph(
        self, context: ProcessingContext, graph: Graph, parent_id: str | None = None
    ):
        """
        Orchestrates the processing of a given graph (or subgraph).

        This method performs the following high-level steps:
        1. Initializes a set for `processed_nodes`.
        2. Processes initial nodes: Calls `_process_initial_nodes` to execute nodes
           that have no input dependencies within the graph.
           If initial node processing fails, graph processing is halted.
        3. Starts the main processing loop: Calls `_main_processing_loop` to handle
           message passing, identify ready nodes, and execute them iteratively until
           the graph is complete or a termination condition (e.g., deadlock) is met.
        4. Logs a warning if not all nodes were processed upon loop completion, which
           might indicate an issue if termination wasn't due to full completion.

        Args:
            context (ProcessingContext): The execution context for the workflow.
            graph (Graph): The graph of nodes to be processed.
            parent_id (str | None): Optional ID of a parent group node, used if this graph
                                    is a subgraph being processed. This is primarily for
                                    logging and contextual information.
        """
        log.info(f"Processing graph (parent_id: {parent_id})")
        log.debug(
            "Graph has %d nodes and %d edges",
            len(graph.nodes),
            len(graph.edges),
        )

        await self._process_trigger_nodes(context, graph)

        # Start the main processing loop for the rest of the graph.
        # This loop handles message passing and sequential/parallel execution of nodes
        # based on their dependencies.
        await self._main_processing_loop(context, graph, parent_id)
        log.debug("Graph processing finished for parent_id: %s", parent_id)

    async def _init_streaming_node(
        self,
        context: ProcessingContext,
        node: BaseNode,
        initial_config_properties: dict[str, Any],
    ):
        """
        Initializes a streaming node for its first run.
        Sets up the generator, assigns initial properties, and sends a "running" update.
        """
        log.info(f"Initializing streaming node: {node.get_title()} ({node._id})")
        self.current_node = node._id  # Ensure current_node is set for ComfyUI hooks

        # Assign initial configuration properties to the node instance.
        # These might come from upstream nodes (via inputs_from_edges) or be intrinsic to the node (for trigger nodes).
        for name, value in initial_config_properties.items():
            try:
                node.assign_property(name, value)
            except Exception as e:
                log.error(
                    f"Error assigning property {name} to streaming node {node.id} during init: {str(e)}"
                )
                # Depending on node design, this might be critical. For now, log and continue.
                # The node's gen_process should be robust to missing/incorrect properties.

        await node.pre_process(context)

        # Send "running" update. Properties reflect the initial configuration.
        node.send_update(
            context, "running", properties=list(initial_config_properties.keys())
        )

        try:
            generator = node.gen_process(context)
            self.active_generators[node._id] = (generator, initial_config_properties)
            log.debug(
                f"Streaming node {node.get_title()} ({node._id}) initialized and generator stored."
            )
        except Exception as e:
            log.error(
                f"Error creating generator for streaming node {node.get_title()} ({node._id}): {str(e)}"
            )
            node.send_update(
                context,
                "error",
                result={"error": str(e)[:1000]},
                properties=list(initial_config_properties.keys()),
            )
            # Remove from active_processing_node_ids if it was added by the caller of process_node
            if node._id in self.active_processing_node_ids:
                self.active_processing_node_ids.remove(node._id)
            raise

    async def _pull_from_streaming_node(
        self, context: ProcessingContext, node: BaseNode
    ):
        """
        Pulls the next item from an active streaming node's generator.
        Handles item yielding, completion, and errors.
        Raises StopAsyncIteration if generator completes, or other exceptions on error.
        """
        log.debug(
            f"Pulling next item from streaming node: {node.get_title()} ({node._id})"
        )
        self.current_node = node._id  # Ensure current_node is set

        if node._id not in self.active_generators:
            log.error(
                f"Attempted to pull from streaming node {node.get_title()} ({node._id}) not in active_generators."
            )
            # This indicates a logic error. The node should have been removed if completed/errored.
            node.send_update(
                context,
                "error",
                result={
                    "error": "Streaming state error: generator not found for pull."
                },
            )
            # To prevent further issues, ensure it's marked as an error that halts the graph.
            raise RuntimeError(
                f"Streaming state error for node {node.get_title()}: generator not found for pull."
            )

        generator, initial_config_properties = self.active_generators[node._id]

        try:
            item = await anext(generator)

            # Validate the format of the yielded item from the generator
            # Based on `AsyncGenerator[tuple[str, Any], None]` type hint for gen_process.
            if not isinstance(item, tuple):
                slot_name = "output"
                value = item
            else:
                if not (len(item) == 2 and isinstance(item[0], str)):
                    error_message = (
                        f"Streaming node {node.get_title()} ({node._id}) yielded item with invalid format. "
                        f"Expected (str, Any) tuple, got {type(item)} with value resembling: {str(item)[:100]}."
                    )
                    log.error(error_message)  # Log here for immediate specific context
                    # Let the generic exception handler below handle send_update, cleanup, and re-raise.
                    raise ValueError(error_message)

                slot_name: str = item[0]
                value: Any = item[1]

            declared_outputs = [o.name for o in node.outputs()]
            if slot_name not in declared_outputs:
                error_message = (
                    f"Streaming node {node.get_title()} ({node._id}) yielded for undeclared output slot: '{slot_name}'. "
                    f"Declared outputs: {declared_outputs}. Input properties during init: {list(initial_config_properties.keys())}."
                )
                log.error(error_message)  # Log here for immediate specific context
                # Let the generic exception handler below handle send_update, cleanup, and re-raise.
                raise ValueError(error_message)

            self.send_messages(node, {slot_name: value}, context)
            log.debug(
                f"Streaming node {node.get_title()} ({node._id}) yielded item for slot: '{slot_name}'"
            )
        except StopAsyncIteration:
            log.info(
                f"Streaming node {node.get_title()} ({node._id}) completed generation."
            )
            node.send_update(
                context,
                "completed",
                result={"status": "completed"},
                properties=list(initial_config_properties.keys()),
            )
            del self.active_generators[node._id]
            raise  # Re-raise StopAsyncIteration to signal completion to process_node
        except Exception as e:
            log.error(
                f"Error during generation for streaming node {node.get_title()} ({node._id}): {str(e)}"
            )
            node.send_update(
                context,
                "error",
                result={"error": str(e)[:1000]},
                properties=list(initial_config_properties.keys()),
            )
            if (
                node._id in self.active_generators
            ):  # Ensure cleanup if error happened before StopAsyncIteration
                del self.active_generators[node._id]
            raise  # Re-raise to be caught by _execute_node_batch

    async def process_node(
        self,
        context: ProcessingContext,
        node: BaseNode,
        inputs_from_edges: dict[str, Any],
    ):
        """
        Processes a single node in the workflow graph.
        Orchestrates initialization and item pulling for streaming nodes,
        event handling for reactive nodes, or standard processing for other nodes.
        """
        log.debug(
            f"Processing node: {node.get_title()} ({node._id}) with inputs: {list(inputs_from_edges.keys())}"
        )
        self.current_node = node._id

        try:
            # Attempt to identify if this invocation is primarily for an event
            event_value_for_handling = None
            event_slot_for_handling = None

            if hasattr(node, "handle_event"):  # Only consider if node has handle_event
                for slot_name, value in inputs_from_edges.items():
                    if isinstance(value, Event):
                        event_value_for_handling = value
                        event_slot_for_handling = slot_name
                        # Found an event for a node that can handle events.
                        # Prioritize this path. Take the first event found.
                        break

            if event_value_for_handling and event_slot_for_handling:
                # An event is present and node is capable of handling it via handle_event.
                # We need to ensure ALL inputs_from_edges are assigned as properties before calling process_event_node,
                # because handle_event might rely on other properties being set.
                log.debug(
                    f"Node {node.get_title()} has an Event on slot '{event_slot_for_handling}'. Assigning all inputs and routing to event processing."
                )
                for key, val in inputs_from_edges.items():
                    try:
                        node.assign_property(key, val)
                    except Exception as e:
                        # Log and potentially raise, as this might be critical
                        log.error(
                            f"Error assigning property {key} to node {node.id} before event handling: {str(e)}"
                        )
                        raise ValueError(
                            f"Error assigning property {key} to node {node.id} for event context: {str(e)}"
                        ) from e

                # Now call process_event_node. It will re-assign the event to event_slot (harmless if already done)
                # and then call node.handle_event(context, event_value_for_handling).
                await self.process_event_node(
                    context, node, event_value_for_handling, event_slot_for_handling
                )

            # Existing logic for other node types
            elif node.is_streaming_output():
                # inputs_from_edges are used as initial_config_properties for _init_streaming_node
                # _init_streaming_node assigns these.
                if node._id not in self.active_generators:
                    # First time processing this streaming node instance in this run.
                    # `inputs_from_edges` contains its initial configuration if it's not a trigger node.
                    # `_init_streaming_node` will assign these or node uses intrinsic config.
                    await self._init_streaming_node(context, node, inputs_from_edges)
                    # After initialization, immediately try to pull.
                    # `_pull_from_streaming_node` will raise StopAsyncIteration if it completes,
                    # or another exception on error, or return normally if it yields an item.
                    await self._pull_from_streaming_node(context, node)
                else:
                    # Node is already an active generator, pull next item.
                    await self._pull_from_streaming_node(context, node)

            elif isinstance(node, OutputNode):
                # OutputNode processing relies on inputs being passed to it,
                # and its 'process' method might use properties set from these inputs.
                # Ensure properties are assigned for OutputNode here.
                log.debug(
                    f"Node {node.get_title()} is OutputNode. Assigning inputs as properties."
                )
                for key, val in inputs_from_edges.items():
                    try:
                        node.assign_property(key, val)
                    except Exception as e:
                        log.error(
                            f"Error assigning property {key} to OutputNode {node.id}: {str(e)}"
                        )
                        raise ValueError(
                            f"Error assigning property {key} to OutputNode {node.id}: {str(e)}"
                        ) from e
                await self.process_output_node(context, node, inputs_from_edges)
            else:
                # Regular, non-streaming, non-output node.
                # process_node_with_inputs handles assigning properties from inputs_from_edges.
                await self.process_node_with_inputs(context, node, inputs_from_edges)

        except StopAsyncIteration:
            # This occurs when a streaming node (either newly initialized or existing) finishes its generation.
            # _pull_from_streaming_node has already sent "completed" and cleaned up from active_generators.
            log.debug(
                f"Node {node.get_title()} ({node._id}) (streaming) finished generation (caught StopAsyncIteration in process_node)."
            )
            # The task for this node will complete successfully.
        except Exception as e:
            log.error(
                f"Exception during process_node for {node.get_title()} ({node._id}): {str(e)}"
            )
            # If it was a streaming node that errored, ensure it's cleaned up.
            # _pull_from_streaming_node should handle this, but this is a safeguard.
            if node.is_streaming_output() and node._id in self.active_generators:
                log.debug(
                    f"Ensuring cleanup of errored streaming node {node.get_title()} from active_generators in process_node exception handler."
                )
                del self.active_generators[node._id]
                # An error update should have been sent by _pull_from_streaming_node or other specific handlers.

            # Re-raise so _execute_node_batch can see the exception and halt graph if necessary.
            raise

    async def process_event_node(
        self, context: ProcessingContext, node: BaseNode, event: Event, event_slot: str
    ):
        """
        Processes a node in response to an event.

        This method calls the node's handle_event method, which is an async generator
        that can yield more outputs (including additional events).

        Args:
            context (ProcessingContext): The processing context.
            node (BaseNode): The node to process.
            event (Event): The event that triggered this processing.
            event_slot (str): The input slot that received the event.
        """
        log.info(
            f"Processing EVENT {event.name} for node {node.get_title()} ({node._id}) on slot '{event_slot}'"
        )

        # Assign event to the appropriate slot
        node.assign_property(event_slot, event)

        # Send running update
        node.send_update(context, "running", properties=[event_slot])

        try:
            # Call handle_event which returns an async generator
            async for slot_name, value in node.handle_event(context, event):
                # Each yielded value gets sent as a message
                self.send_messages(node, {slot_name: value}, context)
                log.debug(
                    f"Event handler for {node.get_title()} ({node._id}) yielded output for slot '{slot_name}'"
                )

            # Send completed update
            node.send_update(
                context,
                "completed",
                result={},
                properties=[event_slot],
            )
        except Exception as e:
            log.error(
                f"Error handling event for node {node.get_title()} ({node._id}): {str(e)}"
            )
            node.send_update(
                context,
                "error",
                result={"error": str(e)[:1000]},
                properties=[event_slot],
            )
            raise

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
           - If cached, uses the cached result.
           - Otherwise, proceeds to execution.
        4. Determines if GPU is required (`node.requires_gpu()`).
           - If GPU required and not available on `self.device`, raises RuntimeError.
           - Sends a "running" `NodeUpdate`.
           - If GPU required and available:
             - Acquires `gpu_lock`.
             - Calls `node.move_to_device(self.device)`.
             - Calls `node.process_with_gpu(context)`.
             - Converts output using `node.convert_output()`.
             - Releases `gpu_lock`.
             - Optionally moves node back to CPU (if in production environment).
           - Else (CPU processing):
             - Calls `node.process(context)`.
             - Converts output using `node.convert_output()`.
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
        log.debug(f"{node.get_title()} ({node._id}) inputs: {inputs}")

        # Assign input values to node properties
        for name, value in inputs.items():
            try:
                node.assign_property(name, value)
            except Exception as e:
                log.error(f"Error assigning property {name} to node {node.id}")
                raise ValueError(f"Error assigning property {name}: {str(e)}")

        # Preprocess the node
        log.debug(f"Pre-processing node: {node.get_title()} ({node._id})")
        await node.pre_process(context)

        # Check if the node is cacheable
        if node.is_cacheable():
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

            if requires_gpu and self.device == "cpu":
                error_msg = f"Node {node.get_title()} ({node._id}) requires a GPU, but no GPU is available."
                log.error(error_msg)
                raise RuntimeError(error_msg)

            node.send_update(
                context, "running", result=None, properties=list(inputs.keys())
            )

            if requires_gpu and self.device != "cpu":
                await acquire_gpu_lock(node, context)
                try:
                    await node.move_to_device(self.device)
                    self.log_vram_usage(
                        f"Node {node.get_title()} ({node._id}) VRAM after move to {self.device}"
                    )

                    result = await self.process_with_gpu(context, node, 0)
                    result = await node.convert_output(context, result)
                finally:
                    if Environment.is_production():
                        await node.move_to_device("cpu")
                        self.log_vram_usage(
                            f"Node {node.get_title()} ({node._id}) VRAM after move to cpu"
                        )
                    release_gpu_lock()
            else:
                result = await node.process(context)
                result = await node.convert_output(context, result)

            # Cache the result if the node is cacheable
            if node.is_cacheable():
                log.debug(f"Caching result for node: {node.get_title()} ({node._id})")
                context.cache_result(node, result)

        # Send completion update
        node.send_update(context, "completed", result=result)
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
            if node.name in self.outputs:
                self.outputs[node.name].append(value)
            else:
                self.outputs[node.name] = [value]

            # Get the type of the output for metadata purposes
            output_type = node.__class__.__name__.replace("Output", "").lower()

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
        else:
            # This case should ideally not happen if graph is validated.
            log.warning(
                f"OutputNode {node.name} ({node._id}) received no 'value' in inputs."
            )
            # Still send a completed update, but with no result value for this path.
            node.send_update(context, "completed", result={}, properties=["name"])

    async def process_with_gpu(
        self, context: ProcessingContext, node: BaseNode, retries: int = 0
    ):
        """
        Processes a node with GPU, with retry logic for CUDA OOM errors.
        """
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
                    # For non-OOM errors in non-streaming nodes, process_node_with_inputs handles logging and NodeUpdate.
                    # It then re-raises, so we just re-raise here to exit the retry loop and propagate.
                    raise
        else:
            # This case implies TORCH_AVAILABLE is False.
            # Fallback to regular processing if no GPU capability or torch is not there.
            return await node.process(context)


async def main():
    class Generator(BaseNode):
        first_name: str = ""
        last_name: str = ""

        async def gen_process(
            self, context: ProcessingContext
        ) -> AsyncGenerator[tuple[str, Any], None]:  # Ensure signature matches BaseNode
            yield "output", self.first_name
            await asyncio.sleep(0.01)  # Using a very short sleep for tests
            yield "output", self.last_name
            await asyncio.sleep(0.01)
            yield "output", "!"

    class Collector(BaseNode):
        value: str = ""

        async def process(self, context: ProcessingContext) -> str:
            return self.value

    class String(BaseNode):
        value: str = ""

        async def process(self, context: ProcessingContext) -> str:
            return self.value

    class StringOutput(OutputNode):
        value: str = ""

        async def process(self, context: ProcessingContext) -> str:
            return self.value

    class IntegerOutput(OutputNode):
        value: int = 0

        async def process(self, context: ProcessingContext) -> int:
            return self.value

    class IntegerInput(InputNode):
        value: int = 0

        async def process(self, context: ProcessingContext) -> int:
            return self.value

    class Add(BaseNode):
        a: float = 0.0
        b: float = 0.0

        async def process(self, context: ProcessingContext) -> float:
            return self.a + self.b

    # Test 3: Event-based Communication
    print("\n--- Starting Event-based Communication Test ---")

    class EventProducer(BaseNode):
        message_prefix: str = "Event"

        @classmethod
        def return_type(cls):
            return {"event_out": Event}

        async def gen_process(
            self, context: ProcessingContext
        ) -> AsyncGenerator[tuple[str, Any], None]:
            for i in range(3):
                event = Event(name=f"{self.message_prefix}_{i}", payload={"index": i})
                yield "event_out", event
                await asyncio.sleep(0.01)

    class EventConsumer(BaseNode):
        event_in: Event | None = None
        messages_received: list[str] = []

        @classmethod
        def return_type(cls):
            return {"response": str}

        async def handle_event(self, context: ProcessingContext, event: Event):
            self.messages_received.append(event.name)
            yield "response", f"Handled: {event.name}"

    event_producer_def = {
        "id": "event_producer",
        "type": EventProducer.get_node_type(),
        "data": {"message_prefix": "TestEvent"},
    }

    event_consumer_def = {
        "id": "event_consumer",
        "type": EventConsumer.get_node_type(),
    }

    event_output_def = {
        "id": "event_output",
        "type": StringOutput.get_node_type(),
        "data": {"name": "event_responses"},
    }

    event_nodes = [event_producer_def, event_consumer_def, event_output_def]

    event_edges = [
        {
            "id": "e1",
            "source": "event_producer",
            "sourceHandle": "event_out",
            "target": "event_consumer",
            "targetHandle": "event_in",
            "ui_properties": {},
        },
        {
            "id": "e2",
            "source": "event_consumer",
            "sourceHandle": "response",
            "target": "event_output",
            "targetHandle": "value",
            "ui_properties": {},
        },
    ]

    event_graph = APIGraph(
        nodes=[Node(**n) for n in event_nodes],
        edges=[Edge(**e) for e in event_edges],
    )

    event_req = RunJobRequest(
        user_id="user_event_test",
        workflow_id="wf_event_test",
        job_type="event_communication_test",
        params={},
        graph=event_graph,
    )

    event_context = ProcessingContext(
        user_id="user_event_test",
        auth_token="local_token_event",
    )

    event_workflow_runner = WorkflowRunner(job_id="event_job_1")

    await event_workflow_runner.run(event_req, event_context)

    print("--------------------------------")
    print("Outputs from Event Communication test:")
    print(event_workflow_runner.outputs)

    # Test 4: Direct Generator to Consumer (OutputNode)
    print("\n--- Starting Direct Generator to Consumer Test ---")

    # Node definitions
    direct_generator_def = {
        "id": "gen_direct_1",
        "type": Generator.get_node_type(),
        "data": {
            "first_name": "TestFirst",
            "last_name": "TestLast",
        },  # Properties set directly
    }
    consumer_output_def = {
        "id": "consumer_out_1",
        "type": StringOutput.get_node_type(),  # StringOutput will act as the consumer
        "data": {
            "name": "direct_gen_consumed_output"
        },  # This name will be the key in outputs
    }

    direct_gen_nodes_list = [
        direct_generator_def,
        consumer_output_def,
    ]

    direct_gen_edges_list = [
        {
            "id": "e_gen_direct_consumer",
            "source": "gen_direct_1",  # Source node ID
            "sourceHandle": "output",  # Output slot of Generator
            "target": "consumer_out_1",  # Target node ID
            "targetHandle": "value",  # Input slot of StringOutput
            "ui_properties": {},
        },
    ]

    direct_gen_graph = APIGraph(
        nodes=[Node(**n) for n in direct_gen_nodes_list],
        edges=[Edge(**e) for e in direct_gen_edges_list],
    )

    direct_gen_req = RunJobRequest(
        user_id="user_direct_gen_test",
        workflow_id="wf_direct_gen_test",
        job_type="direct_generator_consumer_test",
        params={},  # No external params, generator values are in its data
        graph=direct_gen_graph,
    )
    direct_gen_context = ProcessingContext(
        user_id="user_direct_gen_test",
        auth_token="local_token_direct_gen",
    )
    direct_gen_workflow_runner = WorkflowRunner(job_id="direct_gen_job_1")

    await direct_gen_workflow_runner.run(direct_gen_req, direct_gen_context)

    print("--------------------------------")
    print("Outputs from Direct Generator to Consumer test:")
    print(direct_gen_workflow_runner.outputs)


if __name__ == "__main__":
    asyncio.run(main())
