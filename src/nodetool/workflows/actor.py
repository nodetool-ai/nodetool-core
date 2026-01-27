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
from nodetool.observability.tracing import trace_node
from nodetool.workflows.io import NodeInputs, NodeOutputs
from nodetool.workflows.suspendable_node import WorkflowSuspendedException
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

    def _get_list_handles(self) -> set[str]:
        """Return handles that require multi-edge list aggregation for this node."""
        return self.runner.multi_edge_list_inputs.get(self.node._id, set())

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

    async def _auto_save_assets(
        self,
        node: BaseNode,
        result: dict[str, Any],
        context: ProcessingContext,
    ) -> None:
        """Automatically save assets from node outputs when auto_save_asset is enabled.

        Scans the result dictionary for AssetRef instances and saves them to storage
        with proper tracking (node_id, job_id, workflow_id).

        Args:
            node: The node that produced the result
            result: The result dictionary containing node outputs
            context: The processing context with workflow/job information
        """
        from io import BytesIO

        from nodetool.metadata.types import AssetRef

        if not result:
            return

        self.logger.debug(
            "Auto-saving assets for node %s (%s)",
            node.get_title(),
            node._id,
        )

        # Recursively scan result for AssetRef instances
        def find_asset_refs(obj: Any, path: str = "") -> list[tuple[str, AssetRef]]:
            """Recursively find all AssetRef instances in the result."""
            refs: list[tuple[str, AssetRef]] = []

            if isinstance(obj, AssetRef):
                refs.append((path, obj))
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    refs.extend(find_asset_refs(value, new_path))
            elif isinstance(obj, (list, tuple)):
                for idx, value in enumerate(obj):
                    new_path = f"{path}[{idx}]"
                    refs.extend(find_asset_refs(value, new_path))

            return refs

        asset_refs = find_asset_refs(result)

        if not asset_refs:
            self.logger.debug(
                "No AssetRefs found in result for node %s (%s)",
                node.get_title(),
                node._id,
            )
            return

        self.logger.info(
            "Found %d asset(s) to auto-save for node %s (%s)",
            len(asset_refs),
            node.get_title(),
            node._id,
        )

        # Save each asset ref
        for path, asset_ref in asset_refs:
            try:
                # Skip if asset already has an asset_id (already saved)
                if asset_ref.asset_id:
                    self.logger.debug(
                        "Skipping asset at %s - already has asset_id: %s",
                        path,
                        asset_ref.asset_id,
                    )
                    continue

                # Skip if no data to save
                if not asset_ref.data and not asset_ref.uri:
                    self.logger.debug(
                        "Skipping asset at %s - no data or uri",
                        path,
                    )
                    continue

                # Get content type from asset ref type
                content_type = self._get_content_type_for_asset_ref(asset_ref)

                # Generate asset name
                asset_name = f"{node.get_title()}_{path}_{node._id[:8]}"

                # Get data as BytesIO
                if asset_ref.data:
                    # Handle DataframeRef specially - data is list of lists, not bytes
                    from nodetool.metadata.types import DataframeRef, JSONRef, SVGRef
                    if isinstance(asset_ref, DataframeRef):
                        import json
                        # Convert DataFrame data to JSON bytes
                        json_str = json.dumps(asset_ref.data)
                        content = BytesIO(json_str.encode("utf-8"))
                    elif isinstance(asset_ref, (JSONRef, SVGRef)):
                        # JSONRef and SVGRef have string data
                        if isinstance(asset_ref.data, str):
                            content = BytesIO(asset_ref.data.encode("utf-8"))
                        elif isinstance(asset_ref.data, bytes):
                            content = BytesIO(asset_ref.data)
                        else:
                            self.logger.warning(
                                "JSONRef/SVGRef data is not string or bytes at %s",
                                path,
                            )
                            continue
                    elif isinstance(asset_ref.data, bytes):
                        content = BytesIO(asset_ref.data)
                    else:
                        # Try to convert to bytes
                        try:
                            content = BytesIO(bytes(asset_ref.data))
                        except Exception:
                            self.logger.warning(
                                "Could not convert data to bytes for asset at %s",
                                path,
                            )
                            continue
                elif asset_ref.uri.startswith("memory://"):
                    # Resolve memory URI to get the data
                    from nodetool.runtime.resources import require_scope
                    scope = require_scope()
                    obj = scope.get_memory_uri_cache().get(asset_ref.uri)
                    if obj is not None:
                        # Convert object to bytes based on type
                        data_bytes = self._object_to_bytes(obj, asset_ref)
                        if data_bytes:
                            content = BytesIO(data_bytes)
                        else:
                            self.logger.warning(
                                "Could not convert memory object to bytes for asset at %s",
                                path,
                            )
                            continue
                    else:
                        self.logger.warning(
                            "Memory URI not found in cache for asset at %s",
                            path,
                        )
                        continue
                else:
                    # For other URIs, we can't auto-save
                    self.logger.debug(
                        "Skipping asset at %s - unsupported URI type: %s",
                        path,
                        asset_ref.uri,
                    )
                    continue

                # Create and save the asset
                asset = await context.create_asset(
                    name=asset_name,
                    content_type=content_type,
                    content=content,
                    node_id=node._id,
                )

                # Update the AssetRef with the new asset_id
                asset_ref.asset_id = asset.id

                self.logger.info(
                    "Auto-saved asset %s for node %s (%s) at %s",
                    asset.id,
                    node.get_title(),
                    node._id,
                    path,
                )

            except Exception as e:
                self.logger.error(
                    "Failed to auto-save asset at %s for node %s (%s): %s",
                    path,
                    node.get_title(),
                    node._id,
                    e,
                    exc_info=True,
                )

    def _get_content_type_for_asset_ref(self, asset_ref: Any) -> str:
        """Get the appropriate content type for an AssetRef based on its type."""
        from nodetool.metadata.types import (
            AudioRef,
            DataframeRef,
            DocumentRef,
            ExcelRef,
            FolderRef,
            ImageRef,
            JSONRef,
            Model3DRef,
            SVGRef,
            TextRef,
            VideoRef,
        )

        if isinstance(asset_ref, ImageRef):
            return "image/png"
        elif isinstance(asset_ref, AudioRef):
            return "audio/mp3"
        elif isinstance(asset_ref, VideoRef):
            return "video/mp4"
        elif isinstance(asset_ref, TextRef):
            return "text/plain"
        elif isinstance(asset_ref, DocumentRef):
            return "application/pdf"
        elif isinstance(asset_ref, DataframeRef):
            return "application/json"
        elif isinstance(asset_ref, ExcelRef):
            return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif isinstance(asset_ref, Model3DRef):
            return "model/gltf-binary"
        elif isinstance(asset_ref, FolderRef):
            return "folder"
        elif isinstance(asset_ref, JSONRef):
            return "application/json"
        elif isinstance(asset_ref, SVGRef):
            return "image/svg+xml"
        else:
            return "application/octet-stream"

    def _object_to_bytes(self, obj: Any, asset_ref: Any) -> bytes | None:
        """Convert a Python object to bytes based on the asset ref type."""
        from nodetool.metadata.types import AudioRef, DataframeRef, ImageRef, TextRef

        if isinstance(asset_ref, ImageRef):
            # Handle PIL Image
            try:
                from io import BytesIO

                from PIL import Image

                if isinstance(obj, Image.Image):
                    buf = BytesIO()
                    obj.save(buf, format="PNG")
                    return buf.getvalue()
            except Exception as e:
                self.logger.debug(f"Failed to convert image object: {e}")

        elif isinstance(asset_ref, AudioRef):
            # Handle AudioSegment
            try:
                from io import BytesIO

                from pydub import AudioSegment

                if isinstance(obj, AudioSegment):
                    buf = BytesIO()
                    obj.export(buf, format="mp3")
                    return buf.getvalue()
            except Exception as e:
                self.logger.debug(f"Failed to convert audio object: {e}")

        elif isinstance(asset_ref, TextRef):
            # Handle string
            if isinstance(obj, str):
                return obj.encode("utf-8")

        elif isinstance(asset_ref, DataframeRef):
            # Handle pandas DataFrame
            try:
                import pandas as pd

                if isinstance(obj, pd.DataFrame):
                    return obj.to_json(orient="records").encode("utf-8")
            except Exception as e:
                self.logger.debug(f"Failed to convert dataframe object: {e}")

        # For bytes, return as-is
        if isinstance(obj, bytes):
            return obj

        return None

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
        # Get tracer for this job if available
        job_id = self.runner.job_id if hasattr(self.runner, "job_id") else None

        async with trace_node(
            node_id=self.node._id,
            node_type=self.node.get_node_type(),
            job_id=job_id,
        ) as span:
            span.set_attribute("nodetool.node.title", self.node.get_title())
            span.set_attribute("nodetool.node.input_count", len(inputs))
            span.set_attribute("nodetool.node.requires_gpu", self.node.requires_gpu())

            await self._process_node_with_inputs_impl(inputs, span)

    async def _process_node_with_inputs_impl(
        self,
        inputs: dict[str, Any],
        span: Any,  # Span | NoOpSpan from observability.tracing
    ) -> None:
        """Internal implementation of process_node_with_inputs."""
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
        if node.is_cacheable() and not getattr(self.runner, "disable_caching", False):
            cached_result = context.get_cached_result(node)

        if cached_result is not None:
            self.logger.info(
                "Using cached result for node: %s (%s)",
                node.get_title(),
                node._id,
            )
            span.set_attribute("nodetool.node.cache_hit", True)
            result = cached_result
        else:
            span.set_attribute("nodetool.node.cache_hit", False)
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

                span.add_event("gpu_lock_waiting")
                await acquire_gpu_lock(node, context)
                span.add_event("gpu_lock_acquired")
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
                    span.add_event("gpu_lock_released")
            else:
                await node.preload_model(context)
                await node.run(context, node_inputs, outputs_collector)  # type: ignore[arg-type]

            result = outputs_collector.collected()

            if node.is_cacheable() and not getattr(self.runner, "disable_caching", False) and not driven_by_stream:
                self.logger.debug(
                    "Caching result for node: %s (%s)",
                    node.get_title(),
                    node._id,
                )
                await context.cache_result_async(node, result)

        span.set_attribute("nodetool.node.output_count", len(result) if result else 0)

        # Auto-save assets if the node has auto_save_asset enabled
        if node.__class__.auto_save_asset() and result:
            await self._auto_save_assets(node, result, context)

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
                        await self.runner.send_property_update(node, context, name)  # type: ignore[misc]
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

                    await self._execute_with_timeout(
                        context, node.run(context, node_inputs, outputs), node_inputs, outputs
                    )

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
                await self._execute_with_timeout(context, node.run(context, node_inputs, outputs), node_inputs, outputs)

            completed_successfully = True
        finally:
            if completed_successfully:
                await self._send_completed_update(context, node, outputs)
            await self._handle_post_execution(context, node)

    async def _execute_with_timeout(
        self,
        context: Any,
        coro: Awaitable[None],
        node_inputs: NodeInputs,
        outputs: NodeOutputs,
    ) -> None:
        """Execute a node coroutine with optional timeout.

        If the node defines a timeout via get_timeout_seconds(), wrap the execution
        in asyncio.wait_for(). On timeout, post an error message and re-raise.
        """
        timeout_seconds = self.node.get_timeout_seconds()
        if timeout_seconds is None or timeout_seconds <= 0:
            await coro
            return

        try:
            await asyncio.wait_for(coro, timeout=timeout_seconds)
        except TimeoutError:
            context.post_message(
                NodeUpdate(
                    node_id=self.node.id,
                    node_name=self.node.get_title(),
                    node_type=self.node.get_node_type(),
                    status="error",
                    error=f"Node timed out after {timeout_seconds}s",
                )
            )
            raise

    async def _send_completed_update(self, context: Any, node: Any, outputs: NodeOutputs) -> None:
        """Send the completed update with results."""
        result = self._filter_result(outputs.collected())

        # Auto-save assets if the node has auto_save_asset enabled
        if node.__class__.auto_save_asset() and result:
            await self._auto_save_assets(node, result, context)

        await node.send_update(context, "completed", result=result)

    async def _handle_post_execution(self, context: Any, node: Any) -> None:
        """Handle post-execution cleanup."""
        pass  # Placeholder for any post-execution logic

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
        """Run a non-streaming-input node with fan-out per arriving input message.

        Special handling for multi-edge list inputs:
        - Handles marked for list aggregation collect ALL values from ALL upstream
          sources until EOS, then pass the aggregated list to the processor.
        - Non-list handles use standard on_any/zip_all semantics.
        """
        node = self.node

        self.logger.debug(
            "Running batched node %s (%s) sync_mode=%s",
            node.get_title(),
            node._id,
            node.get_sync_mode(),
        )
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
            # Identify handles that need list aggregation (multi-edge to list[T])
            list_handles = self._get_list_handles() & handles

            # If there are list handles that require full aggregation, use list aggregation mode
            if list_handles:
                self.logger.debug(
                    "List aggregation enabled for node %s (%s): handles=%s",
                    node.get_title(),
                    node._id,
                    sorted(list_handles),
                )
                await self._run_with_list_aggregation(handles, list_handles, processor)
            else:
                # Standard behavior for non-list handles
                self.logger.debug(
                    "Standard batching for node %s (%s): handles=%s",
                    node.get_title(),
                    node._id,
                    sorted(handles),
                )
                await self._run_standard_batching(handles, processor)

        await node.handle_eos()
        await self._mark_downstream_eos()

    async def _run_with_list_aggregation(
        self,
        handles: set[str],
        list_handles: set[str],
        processor: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        """Handle multi-edge list input aggregation.

        For handles in `list_handles`, collect ALL values from ALL upstream sources
        until EOS, then aggregate into a list. For other handles, collect the first
        value (or use on_any semantics).

        The processor is called once with aggregated lists for list handles and
        single values for non-list handles.
        """
        node = self.node
        self.logger.debug(
            f"Running with list aggregation for node {node.get_title()} ({node._id}), "
            f"list_handles={list_handles}, all_handles={handles}"
        )

        # Buffers for list handles - collect all values
        list_buffers: dict[str, list[Any]] = {h: [] for h in list_handles}
        # Values for non-list handles - take first value
        non_list_values: dict[str, Any] = {}
        non_list_handles = handles - list_handles
        pending_non_list: set[str] = set(non_list_handles)

        # Drain all inputs until EOS on all handles
        async for handle, item in self.inbox.iter_any():
            if handle not in handles:
                continue

            if handle in list_handles:
                # Aggregate into list buffer - flatten if item is a list
                if isinstance(item, list):
                    list_buffers[handle].extend(item)
                    self.logger.debug(f"List aggregation: {handle} extended with {len(item)} items, buffer size={len(list_buffers[handle])}")
                else:
                    list_buffers[handle].append(item)
                    self.logger.debug(f"List aggregation: {handle} received item, buffer size={len(list_buffers[handle])}")
            else:
                # Non-list handle: take first value (like standard on_any)
                if handle not in non_list_values:
                    non_list_values[handle] = item
                    pending_non_list.discard(handle)
                else:
                    # Update with latest value (combineLatest semantics)
                    non_list_values[handle] = item

        # Build final inputs dict: lists for list handles, single values for others
        inputs: dict[str, Any] = {}

        # Add aggregated lists
        for handle in list_handles:
            inputs[handle] = list_buffers[handle]
            self.logger.debug(f"List aggregation complete for {handle}: {len(list_buffers[handle])} items")

        # Add non-list values
        for handle, value in non_list_values.items():
            inputs[handle] = value

        # Call processor once with all aggregated inputs
        self.logger.debug(f"Calling processor with aggregated inputs: {list(inputs.keys())}")
        await processor(inputs)

    async def _run_standard_batching(
        self,
        handles: set[str],
        processor: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        """Standard batching behavior without list aggregation."""
        node = self.node
        # Track inherent streaming nature of edges for stickiness determination
        inherent_streaming: dict[str, bool] = {}
        for handle in handles:
            edge = next(
                (e for e in self.context.graph.edges if e.target == node._id and e.targetHandle == handle),
                None,
            )
            # Determine if the edge is inherently streaming (from streaming source)
            inherent_streaming[handle] = self.runner.edge_streams(edge) if edge is not None else False

        # Always treat all handles as streaming for "always-on streaming" behavior.
        # This ensures every inbound input update triggers a node execution.
        handle_streaming: dict[str, bool] = dict.fromkeys(handles, True)

        sync_mode = node.get_sync_mode()
        self.logger.debug(
            "Batching mode for %s (%s): sync_mode=%s handle_streaming=%s inherent_streaming=%s",
            node.get_title(),
            node._id,
            sync_mode,
            handle_streaming,
            inherent_streaming,
        )

        if sync_mode == "zip_all" and any(handle_streaming.values()):
            from collections import deque

            buffers: dict[str, deque[Any]] = {h: deque() for h in handles}
            sticky_values: dict[str, Any] = {}
            # Use inherent_streaming for stickiness: non-inherently-streaming inputs are sticky
            is_sticky: dict[str, bool] = {h: not inherent_streaming.get(h, False) for h in handles}
            seen_counts: dict[str, int] = dict.fromkeys(handles, 0)

            def buffer_summary() -> dict[str, int]:
                return {h: len(buf) for h, buf in buffers.items() if buf}

            def ready_to_zip() -> bool:
                # Check if we have enough data to create a batch
                has_any_new_data = False
                for handle in handles:
                    if is_sticky.get(handle, False):
                        # For sticky handles: need either a new buffered value or existing sticky value
                        if buffers[handle]:
                            has_any_new_data = True  # New data available
                        elif handle not in sticky_values:
                            return False  # No value at all for this sticky handle
                    else:
                        # For non-sticky handles: always need buffered items
                        if not buffers[handle]:
                            return False
                        has_any_new_data = True  # Non-sticky data counts as new

                # If all handles are sticky and have sticky values, we need new data to batch.
                # This prevents infinite loops when processing only sticky values.
                all_sticky = all(is_sticky.get(h, False) for h in handles)
                return has_any_new_data or not all_sticky

            def any_closed_and_empty() -> bool:
                """Return True if any non-sticky handle is closed and has no buffered items.

                Sticky handles are allowed to be closed and empty if they have a sticky value,
                as they will reuse that value for future batches.

                Also checks the inbox buffer in addition to the local buffer to avoid
                premature termination when items are still pending in the inbox.
                """
                for h in handles:
                    # Check if handle is closed (no more items coming from upstream)
                    if not self.inbox.is_open(h):
                        # Check both local buffer AND inbox buffer
                        local_empty = not buffers[h]
                        inbox_has_items = self.inbox.has_buffered(h)
                        if local_empty and not inbox_has_items:
                            # For sticky handles, check if we have a sticky value
                            if is_sticky.get(h, False) and h in sticky_values:
                                continue  # Sticky handle with value is OK
                            return True
                return False

            async for handle, item in self.inbox.iter_any():
                if handle not in buffers:
                    continue

                buffers[handle].append(item)
                seen_counts[handle] = seen_counts.get(handle, 0) + 1

                if is_sticky.get(handle, False):
                    sticky_values[handle] = item

                self.logger.debug(
                    "zip_all received: node=%s (%s) handle=%s seen=%s buffers=%s open=%s",
                    node.get_title(),
                    node._id,
                    handle,
                    seen_counts.get(handle, 0),
                    buffer_summary(),
                    {h: self.inbox.is_open(h) for h in handles},
                )

                while ready_to_zip():
                    batch: dict[str, Any] = {}
                    for h in handles:
                        if is_sticky.get(h, False):
                            # For sticky handles: use new value if available, else reuse sticky value
                            if buffers[h]:
                                sticky_values[h] = buffers[h].popleft()
                            batch[h] = sticky_values[h]
                            # NOTE: We do NOT re-add the sticky value to the buffer.
                            # The sticky value is only reused when no new value is available.
                        else:
                            batch[h] = buffers[h].popleft()

                    self.logger.debug(
                        "zip_all batch ready: node=%s (%s) batch_handles=%s buffers=%s",
                        node.get_title(),
                        node._id,
                        list(batch.keys()),
                        buffer_summary(),
                    )
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
                    self.logger.debug(
                        "on_any initial batch ready: node=%s (%s) handles=%s",
                        node.get_title(),
                        node._id,
                        list(current.keys()),
                    )
                    await processor(dict(current))
                    initial_fired = True
                else:
                    self.logger.debug(
                        "on_any batch update: node=%s (%s) updated_handle=%s handles=%s",
                        node.get_title(),
                        node._id,
                        handle,
                        list(current.keys()),
                    )
                    await processor(dict(current))

                # NOTE: With always-on streaming, we do NOT mark non-streaming handles
                # as done. This keeps the node alive for re-invocations with sticky inputs.

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
        if getattr(self.runner, "state_manager", None):
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
        event_logger = getattr(self.runner, "event_logger", None)
        if event_logger:
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
            state_manager = getattr(self.runner, "state_manager", None)
            if state_manager:
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
            state_manager = getattr(self.runner, "state_manager", None)
            if state_manager:
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
            event_logger = getattr(self.runner, "event_logger", None)
            if event_logger:
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
        except WorkflowSuspendedException as e:
            self.logger.info(
                "Node execution suspended: %s (%s) [%s] - %s",
                node.get_title(),
                node._id,
                node.get_node_type(),
                e.reason,
            )

            # Queue state update: suspended (non-blocking via StateManager)
            state_manager = getattr(self.runner, "state_manager", None)
            if state_manager:
                try:
                    await self.runner.state_manager.update_node_state(
                        node_id=node_id,
                        status="suspended",
                        suspended_at=datetime.now(),
                        suspension_reason=e.reason,
                        resume_state_json=e.state,
                    )
                    self.logger.debug(f"Queued state update (suspended) for node {node_id}")
                except Exception as e2:
                    self.logger.error(f"Failed to queue state update: {e2}")

            # Post suspended update (not error)
            try:
                ctx.post_message(
                    NodeUpdate(
                        node_id=node.id,
                        node_name=node.get_title(),
                        node_type=node.get_node_type(),
                        status="suspended",
                        properties={},  # No extra properties for now
                    )
                )
            finally:
                # Suspension stops execution, so we don't mark downstream as drained yet
                # (unless we want downstream to stop too? Usually suspension means pause)
                # For now, let's allow rerun.
                pass
            self.logger.info("Re-raising WorkflowSuspendedException via 'raise e'")
            raise e
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
            state_manager = getattr(self.runner, "state_manager", None)
            if state_manager:
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
            event_logger = getattr(self.runner, "event_logger", None)
            if event_logger:
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
