"""
Workflow execution engine for processing directed acyclic graphs (DAGs) of computational nodes.

This module provides the core workflow execution functionality, handling parallel processing,
resource management, and orchestration of computational nodes. It supports both CPU and
GPU-based computations with automatic device selection and memory management.

Key Components:
    - WorkflowRunner: Main execution engine that processes DAGs of nodes
    - OrderedLock: GPU resource management with FIFO queuing
    - Message: Inter-node communication model

Features:
    - Parallel execution of independent nodes
    - GPU resource management with ordered locking
    - Result caching for cacheable nodes
    - Error handling and retry logic for GPU OOM situations
    - Progress tracking and status updates
    - Support for regular nodes and group nodes (subgraphs)
    - Dynamic device selection (CPU/CUDA/MPS)
    - Automatic VRAM management and cleanup

Example:
    ```python
    runner = WorkflowRunner(job_id="123")
    await runner.run(request, context)
    ```

Dependencies:
    - Optional: torch, comfy (for GPU operations)
    - Required: asyncio, pydantic, logging
"""

import asyncio
from contextlib import contextmanager
import gc
import time
from typing import Any, Optional
from collections import deque
import random

from pydantic import BaseModel

from nodetool.common.model_manager import ModelManager
from nodetool.metadata.types import DataframeRef
from nodetool.types.job import JobUpdate
from nodetool.workflows.base_node import GroupNode, BaseNode, IteratorNode, OutputNode
from nodetool.workflows.types import NodeProgress, NodeUpdate
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.common.environment import Environment
from nodetool.workflows.graph import Graph
from nodetool.common.environment import Environment

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
class OrderedLock:
    def __init__(self):
        self._waiters = deque()
        self._locked = False

    async def acquire(self):
        log.debug("Attempting to acquire GPU lock")
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._waiters.append(fut)
        try:
            if self._locked or self._waiters[0] != fut:
                log.debug("GPU lock is held or others are waiting; waiting for lock")
                await fut
            self._locked = True
            log.debug("GPU lock acquired")
        except asyncio.CancelledError:
            self._waiters.remove(fut)
            # Notify next waiter if necessary
            if self._waiters and not self._locked:
                next_fut = self._waiters[0]
                if not next_fut.done():
                    next_fut.set_result(True)
                    log.debug("Notified next waiter for GPU lock after cancellation")
            raise

    def release(self):
        log.debug("Releasing GPU lock")
        if self._locked:
            self._locked = False
            self._waiters.popleft()
            if self._waiters:
                # Notify the next waiter
                next_fut = self._waiters[0]
                if not next_fut.done():
                    next_fut.set_result(True)
                    log.debug("Notified next waiter for GPU lock")


gpu_lock = OrderedLock()


async def acquire_gpu_lock(node: BaseNode, context: ProcessingContext):
    if gpu_lock._locked or gpu_lock._waiters:
        log.debug(f"Node {node.get_title()} is waiting for GPU lock")
        # Lock is held or others are waiting; send update message
        node.send_update(context, status="waiting")
    await gpu_lock.acquire()
    log.debug(f"Node {node.get_title()} acquired GPU lock")


def release_gpu_lock():
    log.debug("Releasing GPU lock from node")
    gpu_lock.release()


def get_available_vram():
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.cuda.get_device_properties(
            0
        ).total_memory - torch.cuda.memory_allocated(0)
    return 0


class Message(BaseModel):
    target: BaseNode
    slot: str
    value: Any


class WorkflowRunner:
    """
    A workflow execution engine that processes directed acyclic graphs (DAGs) of computational nodes.

    The WorkflowRunner handles the execution of complex workflows by managing node dependencies,
    parallel processing, GPU resource allocation, and result caching. It supports both CPU and
    GPU-based computations, with automatic device selection based on availability.

    Key Features:
        - Parallel execution of independent nodes
        - GPU resource management with ordered locking mechanism
        - Result caching for cacheable nodes
        - Error handling and retry logic for GPU out-of-memory situations
        - Progress tracking and status updates
        - Support for both regular nodes and group nodes (subgraphs)

    Attributes:
        job_id (str): Unique identifier for the workflow execution
        status (str): Current status of the workflow ("running", "completed", "cancelled", or "error")
        current_node (Optional[str]): ID of the node currently being processed
        context (Optional[ProcessingContext]): Execution context for managing state and communication
        messages (deque[Message]): Queue of messages for inter-node communication
        device (str): Computing device to use ("cpu", "cuda", or "mps")
        active_processing_node_ids (set[str]): Track nodes currently in an async task

    Example:
        ```python
        runner = WorkflowRunner(job_id="123")
        await runner.run(request, context)
        ```
    """

    def __init__(self, job_id: str, device: str | None = None):
        """
        Initializes a new WorkflowRunner instance.

        Args:
            job_id (str): Unique identifier for this workflow execution.
            device (str): The device to run the workflow on.
        """
        self.job_id = job_id
        self.status = "running"
        self.current_node: Optional[str] = None
        self.context: Optional[ProcessingContext] = None
        self.outputs: dict[str, Any] = {}
        self.messages: deque[Message] = deque()
        self.active_processing_node_ids: set[str] = (
            set()
        )  # Track nodes currently in an async task
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
        Checks if the workflow is currently in the running state.

        Returns:
            bool: True if the workflow status is "running", False otherwise.
        """
        return self.status == "running"

    async def run(
        self,
        req: RunJobRequest,
        context: ProcessingContext,
    ):
        """
        Executes the entire workflow based on the provided request and context.

        Args:
            req (RunJobRequest): Contains the workflow graph and input parameters.
            context (ProcessingContext): Manages the execution state and inter-node communication.
        Raises:
            ValueError: If the graph is missing or if there's a mismatch between input parameters and graph input nodes.

        Post-conditions:
            - Updates workflow status to "completed", "cancelled", or "error".
            - Posts final JobUpdate message with results.

        Note:
            - Handles input validation, graph processing, and output collection.
            - Manages GPU resources if required by the workflow.
        """
        log.info(f"Starting workflow execution for job_id: {self.job_id}")

        Environment.load_settings()

        assert req.graph is not None, "Graph is required"

        graph = Graph(
            nodes=context.load_nodes(req.graph.nodes),
            edges=req.graph.edges,
        )

        self.context = context
        context.graph = graph
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
            except Exception as e:
                if TORCH_AVAILABLE and isinstance(e, torch.cuda.OutOfMemoryError):
                    error_message = f"VRAM OOM error: {str(e)}. No additional VRAM available after retries."
                    log.error(error_message)
                    context.post_message(
                        JobUpdate(
                            job_id=self.job_id, status="error", error=error_message
                        )
                    )
                    self.status = "error"
                else:
                    raise
            finally:
                log.info("Finalizing nodes")
                for node in graph.nodes:
                    await node.finalize(context)
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        log.info(f"Job {self.job_id} completed successfully")
        total_time = time.time() - start_time
        log.info(f"Finished job {self.job_id} - Total time: {total_time:.2f} seconds")
        context.post_message(
            JobUpdate(
                job_id=self.job_id,
                status="completed",
                result=self.outputs,
                message=f"Workflow {self.job_id} completed in {total_time:.2f} seconds",
            )
        )
        self.status = "completed"

    async def validate_graph(self, context: ProcessingContext, graph: Graph):
        """
        Validates all edges in the graph.
        Validates all nodes for missing input values.
        Every edge is validated for:
        - source node has the correct output type for the target node
        - target node has the correct input type for the source node

        Args:
            context (ProcessingContext): Manages the execution state and inter-node communication.
            graph (Graph): The directed acyclic graph of nodes to be processed.

        Raises:
            ValueError: If the graph has missing input values or contains circular dependencies.
        """
        is_valid = True

        for node in graph.nodes:
            input_edges = [edge for edge in graph.edges if edge.target == node.id]
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
            raise ValueError("Graph contains errors: " + "\n".join(errors))

    async def initialize_graph(self, context: ProcessingContext, graph: Graph):
        """
        Initializes all nodes in the graph.

        Args:
            context (ProcessingContext): Manages the execution state and inter-node communication.
            graph (Graph): The directed acyclic graph of nodes to be processed.

        Raises:
            Exception: Any exception raised during node initialization is caught and reported.
        """
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

    def send_messages(
        self, node: BaseNode, result: dict[str, Any], context: ProcessingContext
    ):
        for key, value in result.items():
            # find edge from node to key
            edges = context.graph.find_edges(node.id, key)
            for edge in edges:
                target_node = context.graph.find_node(edge.target)
                if target_node:
                    self.messages.append(
                        Message(target=target_node, slot=edge.targetHandle, value=value)
                    )
                else:
                    log.warning(f"Node {edge.target} not found")

    async def _process_initial_nodes(
        self,
        context: ProcessingContext,
        graph: Graph,
        processed_nodes: set[str],
    ):
        """
        Processes initial nodes in the graph (those with no incoming edges).

        These nodes are typically input nodes or nodes that don't depend on
        any other node's output within the current graph context. Their properties
        might have been set by request parameters or default values.

        Args:
            context (ProcessingContext): The execution context for the workflow.
            graph (Graph): The graph of nodes to be processed.
            processed_nodes (set[str]): A set of node IDs that have already been
                                       processed. This set will be updated by this method.
        """
        initial_processing_tasks = []
        initial_nodes_for_tasks = []

        for node in graph.nodes:
            # Check if the node has any incoming edges within the current graph
            if not any(edge.target == node.id for edge in graph.edges):
                if node._id not in processed_nodes:
                    log.debug(
                        f"Queueing initial node for processing: {node.get_title()} ({node._id})"
                    )
                    # These nodes have no defined inputs from other nodes via graph edges,
                    # so their 'inputs' dict for process_node is empty.
                    initial_processing_tasks.append(
                        self.process_node(context, node, {})
                    )
                    initial_nodes_for_tasks.append(node)

        if initial_processing_tasks:
            results = await asyncio.gather(
                *initial_processing_tasks, return_exceptions=True
            )
            for i, node in enumerate(initial_nodes_for_tasks):
                if isinstance(results[i], Exception):
                    log.error(
                        f"Error processing initial node {node.get_title()}: {results[i]}"
                    )
                    context.post_message(
                        NodeUpdate(
                            node_id=node.id,
                            node_name=node.get_title(),
                            status="error",
                            error=str(results[i])[:1000],
                        )
                    )
                    # Even if an initial node fails, we mark it as "processed"
                    # to prevent it from being reconsidered in the main loop,
                    # and to allow the job to terminate with an error status.
                    processed_nodes.add(node._id)
                    # Propagate the error to halt further processing if an initial node fails critically.
                    # Depending on desired behavior, this might be handled differently (e.g. allow partial completion).
                    # For now, let's assume an initial node error is critical for this graph.
                    # Consider if JobUpdate should be sent here or if `process_graph` caller handles it.
                    context.post_message(
                        JobUpdate(
                            job_id=self.job_id,
                            status="error",
                            error=f"Critical error processing initial node {node.get_title()}: {str(results[i])[:1000]}",
                        )
                    )
                    raise results[i]  # Re-raise to signal failure
                else:
                    processed_nodes.add(node._id)

    def _buffer_messages(
        self,
        node_inputs_buffer: dict[str, dict[str, Any]],
    ) -> bool:
        """
        Buffers messages from self.messages into node_inputs_buffer.

        For a given (target_node.id, slot_handle) key:
        1. If the corresponding slot in `node_inputs_buffer` already holds a value
           (from a previous call, implying the node hasn't consumed it yet),
           any messages for this key in the current `self.messages` batch are deferred
           (kept in `self.messages`) to avoid overwriting the existing buffered value.
        2. If the slot in `node_inputs_buffer` is empty, the *first* message
           encountered in the current `self.messages` batch for this key will have
           its value buffered. This message is then considered "consumed" for this call.
        3. Any subsequent messages for the same key *within the same `self.messages` batch*
           (i.e., after the first one was buffered, or if all were deferred due to condition 1)
           will also be deferred (kept in `self.messages`).

        Args:
            node_inputs_buffer (dict[str, dict[str, Any]]): A dictionary where keys
                are node IDs and values are dictionaries mapping input slot names
                to their received values. This buffer is updated by this method.

        Returns:
            bool: True if any messages were processed and buffered in this call, False otherwise.
        """
        if not self.messages:
            return False

        any_message_processed_this_call = False
        # Tracks (target_id, slot) for messages whose "first instance" is *considered* in this call.
        # This prevents multiple messages for the same key *within this single call* from being processed.
        keys_considered_in_this_call = set()

        messages_to_keep_for_next_time = deque()
        num_messages_at_start_of_call = len(self.messages)

        for _ in range(num_messages_at_start_of_call):
            msg = self.messages.popleft()
            key = (msg.target._id, msg.slot)
            target_id, slot_name = key

            if key not in keys_considered_in_this_call:
                # This is the first time we are *considering* this key in this specific call.
                keys_considered_in_this_call.add(key)

                # Check if the target slot in node_inputs_buffer is already populated from a *previous* cycle.
                if (
                    target_id in node_inputs_buffer
                    and slot_name in node_inputs_buffer[target_id]
                ):
                    # Slot is already full from a previous cycle. Defer this message.
                    log.debug(
                        f"Slot {slot_name} for node {msg.target.get_title()} ({target_id}) already has a value in buffer. "
                        f"Deferring message (key {key}) from current batch."
                    )
                    messages_to_keep_for_next_time.append(msg)
                else:
                    # Slot is free. This is the designated message for this key in this call. Buffer it.
                    log.debug(
                        f"Buffering message for {msg.target.get_title()} ({target_id}), slot {slot_name} "
                        f"(key {key} new this call, buffer slot free)."
                    )
                    node_inputs_buffer.setdefault(target_id, {})[slot_name] = msg.value
                    any_message_processed_this_call = True
                    # This message is "consumed" for this call.
            else:
                # This key was already "considered" (either buffered, or deferred because the slot was full)
                # earlier in *this same call*. This current message is therefore a subsequent one for a key
                # already handled in this batch. Defer it.
                log.debug(
                    f"Message for key {key} ({msg.target.get_title()}) is subsequent for a key already considered "
                    f"in this call. Deferring."
                )
                messages_to_keep_for_next_time.append(msg)

        self.messages.extend(messages_to_keep_for_next_time)

        return any_message_processed_this_call

    def _get_ready_nodes_and_prepare_tasks(
        self,
        context: ProcessingContext,
        graph: Graph,
        processed_nodes: set[str],
        node_inputs_buffer: dict[str, dict[str, Any]],
    ) -> tuple[list[tuple[BaseNode, dict[str, Any]]], list[asyncio.Task]]:
        """
        Identifies nodes that are ready to be processed and creates processing tasks for them.

        A node is considered ready if:
        1. It is not currently in self.active_processing_node_ids (i.e., not already running).
        2. All its required input slots (defined by incoming edges in the graph)
           have received values, which are stored in the `node_inputs_buffer`.

        Nodes that have already been processed (are in `processed_nodes`) can still be
        made ready again if new inputs arrive for them.

        Args:
            context (ProcessingContext): The execution context for the workflow.
            graph (Graph): The graph of nodes.
            processed_nodes (set[str]): A set of node IDs that have already been processed at least once.
            node_inputs_buffer (dict[str, dict[str, Any]]): Buffer containing
                received input values for nodes.

        Returns:
            tuple[list[tuple[BaseNode, dict[str, Any]]], list[asyncio.Task]]: A tuple containing:
                - A list of `(BaseNode, dict_of_inputs_used)` tuples for nodes ready to be processed.
                - A list of `asyncio.Task` objects for processing these nodes.
        """
        tasks_to_run_this_iteration = []
        # Stores tuples of (node_object, inputs_for_this_specific_run)
        ready_node_task_details_list: list[tuple[BaseNode, dict[str, Any]]] = []

        for node in graph.nodes:
            if node._id in self.active_processing_node_ids:
                log.debug(
                    f"Node {node.get_title()} ({node._id}) is already active. Skipping for this batch."
                )
                continue

            # Determine all input slots this node expects based on incoming edges
            required_input_slots = {
                edge.targetHandle for edge in graph.edges if edge.target == node._id
            }

            # For initial nodes (no graph inputs)
            if not required_input_slots:
                # Initial nodes are typically handled by _process_initial_nodes.
                # If an initial-like node (no configured inputs) is encountered here and not yet processed,
                # it might mean it's designed to be triggered by some other mechanism or a direct call
                # without graph-based inputs. For now, we assume initial nodes are run once.
                # If it's already in processed_nodes, it means its initial run is done.
                # If it's not in processed_nodes, _process_initial_nodes should have caught it.
                # This path is less likely for standard initial nodes after _process_initial_nodes has run.
                # However, if a node has no *defined* inputs via edges but expects to be called,
                # this logic might need refinement based on specific node types or flags.
                # For now, if it has no required_input_slots, it's ready with empty inputs,
                # but only if it hasn't been processed at all yet (as a true initial node).
                if (
                    node._id not in processed_nodes
                ):  # Only allow "initial-like" nodes to run once here if not caught by _process_initial_nodes
                    log.debug(
                        f"Node {node.get_title()} ({node._id}) has no graph inputs. Queueing with empty inputs."
                    )
                    inputs_for_node = {}
                    tasks_to_run_this_iteration.append(
                        self.process_node(context, node, inputs_for_node)
                    )
                    ready_node_task_details_list.append((node, inputs_for_node))
                    self.active_processing_node_ids.add(node._id)  # Mark active
                continue

            current_buffered_inputs = node_inputs_buffer.get(node._id, {})
            all_inputs_ready = all(
                slot in current_buffered_inputs for slot in required_input_slots
            )

            if all_inputs_ready:
                log.debug(f"All inputs ready for node {node.get_title()} ({node._id})")
                inputs_for_node = {
                    slot: current_buffered_inputs[slot] for slot in required_input_slots
                }
                tasks_to_run_this_iteration.append(
                    self.process_node(context, node, inputs_for_node)
                )
                ready_node_task_details_list.append((node, inputs_for_node))
                self.active_processing_node_ids.add(node._id)  # Mark active

        return ready_node_task_details_list, tasks_to_run_this_iteration

    async def _execute_node_batch(
        self,
        context: ProcessingContext,
        ready_node_task_details_list: list[tuple[BaseNode, dict[str, Any]]],
        tasks_to_run: list[asyncio.Task],
        processed_nodes: set[str],
        node_inputs_buffer: dict[str, dict[str, Any]],
    ) -> bool:
        """
        Executes a batch of node processing tasks concurrently and clears consumed inputs.

        Args:
            context (ProcessingContext): The execution context.
            ready_node_task_details_list: List of (BaseNode, dict_of_inputs_used) tuples.
            tasks_to_run (list[asyncio.Task]): The list of tasks to execute.
            processed_nodes (set[str]): Set of node IDs that have run at least once (updated here).
            node_inputs_buffer (dict[str, dict[str, Any]]): Buffer of inputs, to be modified by clearing consumed ones.

        Returns:
            bool: True if any tasks were dispatched for execution, False otherwise.
        """
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
                # Mark as processed (attempted) to avoid immediate retry loops with same bad state
                # if inputs are not cleared on error. For now, inputs are not cleared on error.
                processed_nodes.add(node_processed._id)
                # To halt the entire graph on any error from a batch:
                # context.post_message(JobUpdate(job_id=self.job_id, status="error", error=str(results[i])[:1000]))
                # raise results[i]
            else:
                # Node completed successfully
                processed_nodes.add(
                    node_processed._id
                )  # Mark as having run at least once

                # CRITICAL: Clear the inputs that were just consumed by this successful run
                if node_processed._id in node_inputs_buffer:
                    log.debug(
                        f"Node {node_processed.get_title()} completed. Clearing consumed inputs: {list(inputs_that_were_used.keys())}"
                    )
                    for slot_name in inputs_that_were_used.keys():
                        if slot_name in node_inputs_buffer[node_processed._id]:
                            del node_inputs_buffer[node_processed._id][slot_name]
                            log.debug(
                                f"Cleared consumed input slot '{slot_name}' for node {node_processed.get_title()} from buffer."
                            )

                    # If the node's entry in the buffer is now empty, remove the node's ID as a key
                    if not node_inputs_buffer[node_processed._id]:
                        del node_inputs_buffer[node_processed._id]
                        log.debug(
                            f"Removed empty input buffer entry for node {node_processed.get_title()}."
                        )
        return executed_something

    def _handle_potential_deadlock(
        self,
        context: ProcessingContext,
        graph: Graph,
        processed_nodes: set[str],
        node_inputs_buffer: dict[str, dict[str, Any]],
        max_iterations_limit: int,
    ):
        """
        Handles the situation where the maximum iteration limit is reached in the processing loop.

        This typically indicates a potential deadlock or a cycle in the graph logic
        (though structural cycles should be caught earlier). It logs details about
        nodes that are not yet processed and their missing inputs.

        Args:
            context (ProcessingContext): The execution context.
            graph (Graph): The graph being processed.
            processed_nodes (set[str]): Set of processed node IDs.
            node_inputs_buffer (dict[str, dict[str, Any]]): Buffer of received inputs.
            max_iterations_limit (int): The iteration limit that was reached.
        """
        log.error(
            f"Max iterations ({max_iterations_limit}) reached in processing loop. Potential deadlock."
        )
        for node_id, buffered_data in node_inputs_buffer.items():
            if (
                node_id not in processed_nodes and buffered_data
            ):  # Nodes that have received some inputs but not all
                node_obj = graph.find_node(node_id)
                if node_obj:
                    required = {
                        e.targetHandle for e in graph.edges if e.target == node_id
                    }
                    missing = required - set(buffered_data.keys())
                    if missing:
                        log.warning(
                            f"Node {node_obj.get_title()} ({node_id}) is waiting for missing inputs: {missing}"
                        )
                        context.post_message(
                            NodeUpdate(
                                node_id=node_id,
                                node_name=node_obj.get_title(),
                                status="error",  # Or a specific "stalled" status
                                error=f"Suspected deadlock, missing inputs: {missing}",
                            )
                        )
        # Also check nodes that haven't received *any* inputs yet but are not initial
        for node in graph.nodes:
            if node._id not in processed_nodes and node._id not in node_inputs_buffer:
                is_initial = not any(edge.target == node.id for edge in graph.edges)
                if not is_initial:
                    required_slots = {
                        edge.targetHandle
                        for edge in graph.edges
                        if edge.target == node._id
                    }
                    log.warning(
                        f"Node {node.get_title()} ({node._id}) has not received any inputs. Required: {required_slots}"
                    )
                    context.post_message(
                        NodeUpdate(
                            node_id=node.id,
                            node_name=node.get_title(),
                            status="error",
                            error="Suspected deadlock, node has not received any inputs.",
                        )
                    )
        context.post_message(
            JobUpdate(
                job_id=self.job_id,
                status="error",
                error="Max iterations reached, potential deadlock.",
            )
        )

    def _check_loop_termination_conditions(
        self,
        context: ProcessingContext,
        graph: Graph,
        processed_nodes: set[str],
        node_inputs_buffer: dict[str, dict[str, Any]],
        iterations_without_progress: int,
        max_iterations_limit: int,
        no_messages_this_iteration: bool,  # True if _buffer_messages consumed no messages
        no_nodes_processed_this_iteration: bool,  # True if _execute_node_batch processed no nodes
    ) -> bool:
        """
        Checks various conditions to determine if the main processing loop should terminate.

        Termination occurs if:
        1. All nodes in the graph have been processed.
        2. No messages are pending, and no progress (no new nodes processed) has been
           made for a few iterations (controlled by `iterations_without_progress`). This
           catches scenarios where processing has naturally completed.
        3. The maximum number of iterations is reached, indicating a potential deadlock
           or an issue with graph progression.
        4. A more sophisticated check (`all_potentially_done_check`) determines if remaining
           unprocessed nodes are unlikely to receive their required inputs because their
           source nodes are also stuck or unprocessed.

        Args:
            context (ProcessingContext): The execution context.
            graph (Graph): The graph being processed.
            processed_nodes (set[str]): Set of processed node IDs.
            node_inputs_buffer (dict[str, dict[str, Any]]): Buffer of received inputs.
            iterations_without_progress (int): Counter for iterations where no new
                                               nodes were processed and no messages consumed.
            max_iterations_limit (int): The hard limit for loop iterations.
            no_messages_this_iteration (bool): Whether messages were consumed in current iteration.
            no_nodes_processed_this_iteration (bool): Whether nodes were processed in current iteration.

        Returns:
            bool: True if the loop should terminate, False otherwise.
        """
        if len(processed_nodes) == len(graph.nodes):
            log.info("All nodes processed at least once.")
            # This condition is no longer a primary reason to exit the loop if nodes can run multiple times.
            # Loop termination is now primarily handled by iterations_without_progress.
            # However, knowing all nodes ran once can be useful info or for specific graph types.
            # For now, let's not return True here to allow for re-runs.
            # return True # Removed this line

        # Heuristic: If no messages for a few cycles and no actual node processing occurred.
        if iterations_without_progress > 2:  # Original condition was > 2
            log.info(
                f"No messages and no progress made for {iterations_without_progress} iterations. Concluding graph processing."
            )
            return True

        if iterations_without_progress > max_iterations_limit:
            self._handle_potential_deadlock(
                context,
                graph,
                processed_nodes,
                node_inputs_buffer,
                max_iterations_limit,
            )
            # Error message is posted by _handle_potential_deadlock
            return True  # Terminate due to max iterations

        # Check if all remaining nodes are waiting for inputs from sources that are themselves not processed.
        # This is a more complex check for graph completion / deadlock.
        # Original condition:
        # (not self.messages and all(nid in processed_nodes for nid, buffer in node_inputs_buffer.items() if buffer) and
        #  all(any(e.target == n.id for e in graph.edges) or n.id in processed_nodes for n in graph.nodes))
        # Simplified interpretation: If there are no more messages to process, and all nodes that
        # have received *some* inputs are processed, AND all other nodes are either processed or
        # have incoming edges (implying they are waiting), then we might be done or stuck.

        # if (
        #     no_messages_this_iteration and no_nodes_processed_this_iteration
        # ):  # Only check this if we are truly idle in this iter
        #     all_potentially_done_check = True
        #     for node_check in graph.nodes:
        #         if node_check._id not in processed_nodes:
        #             # Does this node expect inputs?
        #             required_slots = {
        #                 edge.targetHandle
        #                 for edge in graph.edges
        #                 if edge.target == node_check._id
        #             }
        #             if (
        #                 not required_slots
        #             ):  # Should have been an initial node and processed
        #                 # This indicates an issue if it's not processed.
        #                 # However, _process_initial_nodes should handle these.
        #                 continue

        #             buffered_for_node = node_inputs_buffer.get(node_check._id, {})
        #             # Is it missing any required inputs?
        #             if not required_slots.issubset(buffered_for_node.keys()):
        #                 # Yes, it's missing inputs. Are the sources of these missing inputs already processed?
        #                 # If a source is not processed, then this node_check might still receive input.
        #                 sources_for_missing_inputs_are_pending = False
        #                 for req_slot in required_slots:
        #                     if (
        #                         req_slot not in buffered_for_node
        #                     ):  # For each missing input
        #                         for edge in graph.edges:  # Find its source edge
        #                             if (
        #                                 edge.target == node_check._id
        #                                 and edge.targetHandle == req_slot
        #                             ):
        #                                 source_node_id = edge.source
        #                                 # If the source of this missing input is NOT yet processed,
        #                                 # then this node_check is legitimately waiting.
        #                                 if source_node_id not in processed_nodes:
        #                                     sources_for_missing_inputs_are_pending = (
        #                                         True
        #                                     )
        #                                     break
        #                         if sources_for_missing_inputs_are_pending:
        #                             break
        #                 if sources_for_missing_inputs_are_pending:
        #                     all_potentially_done_check = (
        #                         False  # This node is waiting for an unprocessed source
        #                     )
        #                     break  # No need to check other nodes, graph is not done/stuck yet
        #                 # else: this node is missing inputs, but all its sources ARE processed.
        #                 # This implies it will never get those inputs - a graph logic error.
        #                 # Or, if it's an iterator/group output, the internal logic dictates completion.
        #                 # This specific check is more about "can progress be made via existing message flow".

        #     if (
        #         all_potentially_done_check and not self.messages
        #     ):  # Double check self.messages as it might have been populated by a concurrent process
        #         log.info(
        #             "All remaining unprocessed nodes are either waiting for inputs from already processed sources (error) "
        #             "or their sources are also pending (legitimately waiting, but if no messages, implies potential stall). "
        #             "Given no new messages this cycle, and if this state persists, it's a stall. For now, relying on iterations_without_progress."
        #         )
        #         # This condition alone might be too aggressive if there are complex dependencies.
        #         # The original `iterations_without_progress > 2` is a more robust general completion check.
        #         # If after this check, `iterations_without_progress` increments and hits its threshold, then we exit.
        #         # Consider returning True here only if truly stuck (e.g. missing inputs from *processed* sources)

        return False  # Default: continue loop

    async def _main_processing_loop(
        self,
        context: ProcessingContext,
        graph: Graph,
        processed_nodes: set[str],
        node_inputs_buffer: dict[str, dict[str, Any]],
        parent_id: str | None,  # Used for logging context
    ):
        """
        The main event loop for processing nodes in the graph.

        This loop iteratively performs the following steps:
        1. Buffers any incoming messages from `self.messages` into `node_inputs_buffer`.
        2. Identifies all nodes whose input dependencies are now met based on `node_inputs_buffer`.
        3. Creates and executes a batch of processing tasks for these ready nodes.
        4. Updates the set of `processed_nodes`.
        5. Checks for termination conditions (e.g., all nodes processed, no progress, max iterations).

        The loop continues until all nodes are processed or a termination condition is met.

        Args:
            context (ProcessingContext): The execution context for the workflow.
            graph (Graph): The graph of nodes to be processed.
            processed_nodes (set[str]): A set of node IDs that have already been processed.
                                       This set is updated during the loop.
            node_inputs_buffer (dict[str, dict[str, Any]]): A buffer where incoming
                messages (node inputs) are stored. This is updated during the loop.
            parent_id (str | None): Optional ID of a parent group node, for logging purposes.
        """
        iterations_without_progress = 0
        # Heuristic limit: N*3 (3 passes per node for complex message patterns) + buffer
        max_iterations_limit = len(graph.nodes) * 3 + 10

        while True:
            # 1. Buffer current messages
            messages_were_consumed = self._buffer_messages(node_inputs_buffer)

            # 2. Check for ready nodes and prepare tasks
            ready_node_task_details_list, tasks_to_run = (
                self._get_ready_nodes_and_prepare_tasks(
                    context, graph, processed_nodes, node_inputs_buffer
                )
            )

            # 3. Execute the batch of ready nodes
            nodes_were_processed_this_iteration = await self._execute_node_batch(
                context,
                ready_node_task_details_list,
                tasks_to_run,
                processed_nodes,
                node_inputs_buffer,
            )

            # 4. Update progress counter
            if nodes_were_processed_this_iteration or messages_were_consumed:
                iterations_without_progress = 0
            else:
                iterations_without_progress += 1
                log.debug(
                    f"No progress in graph processing iteration {iterations_without_progress} for parent_id: {parent_id}"
                )

            # 5. Check termination conditions
            if self._check_loop_termination_conditions(
                context,
                graph,
                processed_nodes,
                node_inputs_buffer,
                iterations_without_progress,
                max_iterations_limit,
                not messages_were_consumed,  # no_messages_this_iteration
                not nodes_were_processed_this_iteration,  # no_nodes_processed_this_iteration
            ):
                break

    async def process_graph(
        self, context: ProcessingContext, graph: Graph, parent_id: str | None = None
    ):
        """
        Processes the graph by orchestrating node execution based on dependencies and message passing.

        The method first processes initial nodes (those with no incoming dependencies).
        Then, it enters a main loop that iteratively:
        1. Buffers incoming messages from completed nodes into an internal buffer.
        2. Identifies nodes that have all their required inputs available in the buffer.
        3. Executes these ready nodes in a batch.
        4. Updates the set of processed nodes.
        5. Checks for termination conditions (e.g., no more messages, no progress, or potential deadlock).

        This ensures that nodes are processed in the correct order, respecting their
        dependencies, and allows for parallel execution where possible.

        Args:
            context (ProcessingContext): The execution context for the workflow.
            graph (Graph): The graph of nodes to be processed.
            parent_id (str | None): Optional ID of a parent group node, if this graph is a subgraph.
                                    This is primarily used for logging and context.
        """
        log.info(f"Processing graph (parent_id: {parent_id})")

        processed_nodes: set[str] = set()
        # Buffer to store inputs for nodes as they arrive.
        # Key: node_id, Value: dict of {slot_name: value}
        node_inputs_buffer: dict[str, dict[str, Any]] = {}

        # Process initial nodes (those with no incoming edges or whose inputs are already satisfied externally)
        # These nodes can kickstart the graph processing.
        try:
            await self._process_initial_nodes(context, graph, processed_nodes)
        except Exception as e:
            # _process_initial_nodes logs and posts JobUpdate on error.
            # Re-raising here will ensure the run() method's finally block executes,
            # but the job status should already be 'error'.
            log.error(
                f"Halting graph processing due to error in initial nodes (parent_id: {parent_id})."
            )
            # No further processing if initial nodes fail critically.
            return

        # Start the main processing loop for the rest of the graph.
        # This loop handles message passing and sequential/parallel execution of nodes
        # based on their dependencies.
        await self._main_processing_loop(
            context, graph, processed_nodes, node_inputs_buffer, parent_id
        )

        # Final check: Log if not all nodes were processed, which might indicate an issue
        # if the loop terminated for reasons other than full completion.
        if len(processed_nodes) < len(graph.nodes):
            unprocessed_nodes = [
                node.get_title()
                for node in graph.nodes
                if node._id not in processed_nodes
            ]
            log.warning(
                f"Graph processing loop finished for parent_id: {parent_id}, but not all nodes were processed. Unprocessed: {unprocessed_nodes}"
            )
            # Note: Job status (error/completed) should have been set by _check_loop_termination_conditions
            # or if an unhandled exception occurred.

        log.info(f"Finished processing graph (parent_id: {parent_id})")

    async def process_node(
        self,
        context: ProcessingContext,
        node: BaseNode,
        inputs: dict[str, Any],
    ):
        """
        Processes a single node in the workflow graph.

        Args:
            context (ProcessingContext): Manages the execution state and inter-node communication.
            node (BaseNode): The node to be processed.
            inputs (dict[str, Any]): The inputs for the node.
        """
        log.info(f"Processing node: {node.get_title()} ({node._id})")

        self.current_node = node._id

        retries = 0

        while retries < MAX_RETRIES:
            try:
                if isinstance(node, OutputNode):
                    self.outputs[node.name] = inputs["value"]
                    break
                if isinstance(node, IteratorNode):
                    node.input_list = inputs["input_list"]
                    await self.process_iterator_node(context, node)
                else:
                    await self.process_node_with_inputs(context, node, inputs)
                break
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
                            for model in comfy.model_management.current_loaded_models:
                                model.model_unload()

                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        additional_vram = vram_before_cleanup - get_available_vram()
                        log.error(f"VRAM after cleanup: {get_available_vram()} GB")

                    if retries >= MAX_RETRIES:
                        raise

                    delay = min(
                        BASE_DELAY * (2 ** (retries - 1)) + random.uniform(0, 1),
                        MAX_DELAY,
                    )
                    log.warning(
                        f"VRAM OOM encountered for node {node._id}. Retrying in {delay:.2f} seconds. (Attempt {retries}/{MAX_RETRIES})"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

    async def process_node_with_inputs(
        self, context: ProcessingContext, node: BaseNode, inputs: dict[str, Any]
    ):
        """
        Process a regular node that is not a GroupNode.

        Args:
            context (ProcessingContext): The processing context.
            node (BaseNode): The node to process.
            inputs (dict[str, Any]): The inputs for the node.
        """
        from datetime import datetime

        started_at = datetime.now()
        try:
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
                log.info(
                    f"Using cached result for node: {node.get_title()} ({node._id})"
                )
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

                        result = await node.process_with_gpu(context)
                        result = await node.convert_output(context, result)
                    finally:
                        release_gpu_lock()
                        # Move to cpu after processing to avoid OOM
                        if Environment.is_production():
                            try:
                                await node.move_to_device("cpu")
                                self.log_vram_usage(
                                    f"Node {node.get_title()} ({node._id}) VRAM after move to cpu"
                                )
                            finally:
                                release_gpu_lock()
                else:
                    result = await node.process(context)
                    result = await node.convert_output(context, result)

                # Cache the result if the node is cacheable
                if node.is_cacheable():
                    log.debug(
                        f"Caching result for node: {node.get_title()} ({node._id})"
                    )
                    context.cache_result(node, result)

            # Send completion update
            node.send_update(context, "completed", result=result)

        except Exception as e:
            import traceback

            log.error(
                f"Error processing node {node.get_title()} ({node._id}): {str(e)}"
            )
            log.error(f"Exception stack trace:\n{traceback.format_exc()}")
            context.post_message(
                NodeUpdate(
                    node_id=node.id,
                    node_name=node.get_title(),
                    status="error",
                    error=str(e)[:1000],
                )
            )
            raise

        self.send_messages(node, result, context)
        log.info(
            f"{node.get_title()} ({node._id}) processing time: {datetime.now() - started_at}"
        )

    async def process_iterator_node(
        self, context: ProcessingContext, node: IteratorNode
    ):
        """
        Processes an IteratorNode, which emits a message for each item in the input_list.
        Assumes node.input_list (or equivalent property) has been set via assign_property
        from the inputs passed to process_node.
        """
        # Determine the list to iterate over, e.g. from node.properties['input_list_slot_name']
        # This relies on IteratorNode's internal logic to access its list property.
        items_to_iterate = node.input_list

        log.info(
            f"IteratorNode {node.get_title()} processing {len(items_to_iterate)} items."
        )

        for item in items_to_iterate:
            # Assuming "output" is the conventional name for the iterator's output handle
            edges = context.graph.find_edges(node.id, "output")
            for edge in edges:
                target = context.graph.find_node(edge.target)
                if target:
                    self.messages.append(
                        Message(target=target, slot=edge.targetHandle, value=item)
                    )
                else:
                    log.warning(f"Node {edge.target} not found for iterator message")

        # Iterator node completion
        summary_result = {"processed_item_count": len(items_to_iterate)}
        node.send_update(context, "completed", result=summary_result)

    def log_vram_usage(self, message=""):
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()
            vram = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024
            log.info(f"{message} VRAM: {vram:.2f} GB")

    @contextmanager
    def torch_context(self, context: ProcessingContext):
        """
        Context manager for handling GPU tasks.

        Args:
            context (ProcessingContext): Manages the execution state and inter-node communication.

        Note:
            - Sets up progress tracking hooks for ComfyUI operations.
            - Manages CUDA memory and PyTorch inference mode.
            - Cleans up models and CUDA cache after execution.
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
