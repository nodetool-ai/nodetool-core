# Tech Design: Shared Actor Pool for Sub-Graph Tools

## Overview

This document describes the design for enabling `GraphTool` to maintain persistent actors across multiple invocations, allowing stateful nodes (like code executors) to remain running while the agent makes repeated tool calls to the same sub-graph.

## Problem Statement

Currently, `GraphTool.process()` creates a new `WorkflowRunner`, `ProcessingContext`, and full set of `NodeActor` instances for every tool invocation. This causes:

1. **No state persistence**: Nodes like code executors lose their runtime state between calls
2. **Initialization overhead**: Each invocation pays the full cost of graph setup
3. **Resource waste**: GPU models are loaded/unloaded repeatedly
4. **Blocking execution**: Tool must wait for complete workflow termination before returning

## Goals

1. Allow sub-graph actors to persist across multiple `GraphTool` invocations
2. Enable the parent context/runner to share actors with sub-graphs
3. Return results immediately when `ToolResultNode` receives input (don't wait for workflow completion)
4. Maintain isolation between different `GraphTool` instances
5. Support proper cleanup when the parent workflow terminates

## Non-Goals

- Sharing actors between different agent sessions
- Distributed actor execution
- Actor state persistence across process restarts

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Parent Workflow                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    WorkflowRunner (Parent)                   │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │    │
│  │  │ NodeActor A │  │ NodeActor B │  │ AgentNode (Actor C) │  │    │
│  │  └─────────────┘  └─────────────┘  └──────────┬──────────┘  │    │
│  │                                                │             │    │
│  │  node_inboxes: {A: inbox_a, B: inbox_b, C: inbox_c}         │    │
│  │  actor_registry: {A: actor_a, B: actor_b, C: actor_c}       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                   │                  │
│                                    ┌──────────────▼──────────────┐  │
│                                    │         GraphTool           │  │
│                                    │  ┌──────────────────────┐   │  │
│                                    │  │ SubGraphController   │   │  │
│                                    │  │  - result_queue      │   │  │
│                                    │  │  - input_injector    │   │  │
│                                    │  └──────────┬───────────┘   │  │
│                                    └─────────────┼───────────────┘  │
│                                                  │                  │
│  ┌───────────────────────────────────────────────▼──────────────┐  │
│  │                  WorkflowRunner (Sub-Graph)                   │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │  │
│  │  │ InputProxy  │  │ CodeExec    │  │   ToolResultNode    │   │  │
│  │  │   (Actor)   │  │  (Actor)    │  │      (Actor)        │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘   │  │
│  │                                                               │  │
│  │  shared_actor_pool: reference to parent's actor_registry     │  │
│  │  result_queue: pipes ToolResultUpdate to GraphTool           │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow for Repeated Tool Calls

```
Call 1:                          Call 2:                          Call 3:
─────────                        ─────────                        ─────────
GraphTool.process(params1)       GraphTool.process(params2)       GraphTool.process(params3)
         │                                │                                │
         ▼                                ▼                                ▼
┌─────────────────┐              ┌─────────────────┐              ┌─────────────────┐
│ _ensure_started │              │ (already running)│             │ (already running)│
│ - create runner │              │                 │              │                 │
│ - start actors  │              │                 │              │                 │
│ - start bg task │              │                 │              │                 │
└────────┬────────┘              └────────┬────────┘              └────────┬────────┘
         │                                │                                │
         ▼                                ▼                                ▼
┌─────────────────┐              ┌─────────────────┐              ┌─────────────────┐
│ _inject_inputs  │              │ _inject_inputs  │              │ _inject_inputs  │
│ inbox.put(p1)   │              │ inbox.put(p2)   │              │ inbox.put(p3)   │
└────────┬────────┘              └────────┬────────┘              └────────┬────────┘
         │                                │                                │
         ▼                                ▼                                ▼
┌─────────────────┐              ┌─────────────────┐              ┌─────────────────┐
│ CodeExec node   │              │ CodeExec node   │              │ CodeExec node   │
│ (SAME ACTOR)    │──────────────│ (SAME ACTOR)    │──────────────│ (SAME ACTOR)    │
│ state preserved │              │ state preserved │              │ state preserved │
└────────┬────────┘              └────────┬────────┘              └────────┬────────┘
         │                                │                                │
         ▼                                ▼                                ▼
┌─────────────────┐              ┌─────────────────┐              ┌─────────────────┐
│ ToolResultNode  │              │ ToolResultNode  │              │ ToolResultNode  │
│ posts result    │              │ posts result    │              │ posts result    │
└────────┬────────┘              └────────┬────────���              └────────┬────────┘
         │                                │                                │
         ▼                                ▼                                ▼
   result_queue.put              result_queue.put               result_queue.put
         │                                │                                │
         ▼                                ▼                                ▼
   return result1                 return result2                  return result3
```

## Detailed Design

### 1. New Class: `SubGraphController`

A controller that manages a persistent sub-graph execution with input injection and result extraction.

```python name=src/nodetool/agents/tools/sub_graph_controller.py
"""
Controller for persistent sub-graph execution.

Manages the lifecycle of a sub-graph that can receive multiple input injections
and return results without full workflow restart.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from nodetool.config.logging_config import get_logger
from nodetool.workflows.graph import Graph
from nodetool.workflows.inbox import NodeInbox
from nodetool.workflows.processing_context import AssetOutputMode, ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.types import JobUpdate, ToolResultUpdate
from nodetool.workflows.workflow_runner import WorkflowRunner

if TYPE_CHECKING:
    from nodetool.types.api_graph import Edge, Graph as ApiGraph

log = get_logger(__name__)


class SubGraphController:
    """
    Controls a persistent sub-graph execution that stays alive across multiple invocations.
    
    Lifecycle:
    1. start() - Initialize the runner and start background execution
    2. inject_and_wait() - Inject inputs and wait for next result (called per tool invocation)
    3. stop() - Gracefully shut down the sub-graph
    
    Thread Safety:
        - inject_and_wait() is NOT thread-safe; calls must be serialized
        - stop() can be called from any context
    
    Attributes:
        graph: The API graph definition for the sub-graph
        input_node_ids: Mapping of input handle -> node ID for input injection
        runner: The persistent WorkflowRunner instance
        context: The persistent ProcessingContext
        result_queue: Queue for receiving ToolResultUpdate messages
    """
    
    def __init__(
        self,
        api_graph: ApiGraph,
        input_edges: list[Edge],
        parent_context: ProcessingContext,
    ):
        """
        Initialize the controller.
        
        Args:
            api_graph: The API graph to execute
            input_edges: Edges from external inputs to graph nodes (defines injection points)
            parent_context: Parent context for inheriting user_id, auth_token, device, etc.
        """
        self.api_graph = api_graph
        self.input_edges = input_edges
        self.parent_context = parent_context
        
        # Persistent state (initialized in start())
        self._runner: WorkflowRunner | None = None
        self._context: ProcessingContext | None = None
        self._result_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._background_task: asyncio.Task | None = None
        self._started = False
        self._stopped = False
        
        # Build input injection map: handle_name -> target_node_id
        self._input_map: dict[str, str] = {
            edge.targetHandle: edge.target for edge in input_edges
        }
    
    @property
    def is_running(self) -> bool:
        """Check if the sub-graph is currently running."""
        return self._started and not self._stopped and self._background_task is not None
    
    async def start(self) -> None:
        """
        Start the sub-graph execution in the background.
        
        This initializes the runner, creates actors for all nodes, and starts
        the background task that collects results. The sub-graph will continue
        running until stop() is called.
        
        Raises:
            RuntimeError: If already started or stopped
        """
        if self._started:
            raise RuntimeError("SubGraphController already started")
        if self._stopped:
            raise RuntimeError("SubGraphController already stopped; create a new instance")
        
        self._started = True
        
        # Create persistent context with isolated message queue
        import queue as _queue
        isolated_queue: _queue.Queue = _queue.Queue()
        
        self._context = ProcessingContext(
            user_id=self.parent_context.user_id,
            auth_token=self.parent_context.auth_token,
            graph=Graph.from_dict(self.api_graph.model_dump()),
            message_queue=isolated_queue,
            device=self.parent_context.device,
            workspace_dir=self.parent_context.workspace_dir,
            asset_output_mode=getattr(
                self.parent_context, "asset_output_mode", AssetOutputMode.TEMP_URL
            ),
        )
        
        # Create persistent runner
        self._runner = WorkflowRunner(
            job_id=uuid4().hex,
            disable_caching=True,
        )
        
        # Start background collection task
        self._background_task = asyncio.create_task(
            self._run_and_collect(),
            name=f"subgraph-{self._runner.job_id}",
        )
        
        log.info(
            "SubGraphController started: job_id=%s, input_handles=%s",
            self._runner.job_id,
            list(self._input_map.keys()),
        )
    
    async def _run_and_collect(self) -> None:
        """
        Background task that runs the workflow and collects ToolResultUpdate messages.
        
        This task runs indefinitely (or until cancelled/error) and forwards
        ToolResultUpdate messages to the result queue.
        """
        assert self._context is not None
        assert self._runner is not None
        
        req = RunJobRequest(
            user_id=self._context.user_id,
            auth_token=self._context.auth_token,
            graph=self.api_graph,
        )
        
        try:
            async for msg in run_workflow(
                request=req,
                runner=self._runner,
                context=self._context,
                use_thread=False,  # Run in same thread to share event loop
                send_job_updates=False,
                initialize_graph=True,
                validate_graph=False,
            ):
                # Forward non-job messages to parent context
                if not isinstance(msg, JobUpdate):
                    self.parent_context.post_message(msg)
                
                # Capture tool results
                if isinstance(msg, ToolResultUpdate):
                    if msg.result is not None:
                        await self._result_queue.put(msg.result)
                        log.debug(
                            "SubGraphController captured result: %s",
                            list(msg.result.keys()) if isinstance(msg.result, dict) else type(msg.result),
                        )
        except asyncio.CancelledError:
            log.info("SubGraphController background task cancelled")
            raise
        except Exception as e:
            log.error("SubGraphController background task error: %s", e)
            # Put error in queue so inject_and_wait can handle it
            await self._result_queue.put({"__error__": str(e)})
            raise
    
    async def inject_and_wait(
        self,
        params: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """
        Inject input parameters and wait for the next result.
        
        This method:
        1. Injects the provided params into the appropriate node inboxes
        2. Waits for a ToolResultUpdate to arrive in the result queue
        3. Returns the result
        
        Args:
            params: Input parameters matching the graph's input schema
            timeout: Optional timeout in seconds; None means wait indefinitely
        
        Returns:
            The result dictionary from ToolResultNode
        
        Raises:
            RuntimeError: If not started or already stopped
            asyncio.TimeoutError: If timeout is reached before result
            Exception: If the sub-graph encountered an error
        """
        if not self._started:
            raise RuntimeError("SubGraphController not started; call start() first")
        if self._stopped:
            raise RuntimeError("SubGraphController already stopped")
        if self._runner is None:
            raise RuntimeError("Runner not initialized")
        
        # Inject inputs into the appropriate node inboxes
        for handle, value in params.items():
            target_node_id = self._input_map.get(handle)
            if target_node_id is None:
                log.warning("No input mapping for handle: %s", handle)
                continue
            
            inbox = self._runner.node_inboxes.get(target_node_id)
            if inbox is None:
                log.warning("No inbox found for node: %s", target_node_id)
                continue
            
            await inbox.put(handle, value)
            log.debug(
                "Injected input: handle=%s, node=%s, value_type=%s",
                handle,
                target_node_id,
                type(value).__name__,
            )
        
        # Wait for result
        try:
            if timeout is not None:
                result = await asyncio.wait_for(
                    self._result_queue.get(),
                    timeout=timeout,
                )
            else:
                result = await self._result_queue.get()
        except asyncio.TimeoutError:
            log.warning("inject_and_wait timed out after %s seconds", timeout)
            raise
        
        # Check for error
        if isinstance(result, dict) and "__error__" in result:
            raise Exception(result["__error__"])
        
        return result
    
    async def stop(self) -> None:
        """
        Stop the sub-graph execution gracefully.
        
        This cancels the background task and cleans up resources.
        Safe to call multiple times.
        """
        if self._stopped:
            return
        
        self._stopped = True
        
        # Cancel background task
        if self._background_task is not None:
            self._background_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._background_task
            self._background_task = None
        
        # Clean up runner
        if self._runner is not None:
            # Drain any pending messages
            if self._context is not None and self._context.graph is not None:
                self._runner.drain_active_edges(self._context, self._context.graph)
            self._runner = None
        
        self._context = None
        
        log.info("SubGraphController stopped")
    
    async def __aenter__(self) -> SubGraphController:
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
```

### 2. Modified `GraphTool` Class

Update `GraphTool` to use `SubGraphController` for persistent execution:

```python name=src/nodetool/agents/tools/workflow_tool.py
"""
Tool for executing specific workflows as agent tools.

This module provides GraphTool and WorkflowTool, which allow workflows to be 
used as tools by agents. GraphTool supports persistent execution for stateful
nodes like code executors.
"""

from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from nodetool.agents.tools.base import Tool
from nodetool.agents.tools.sub_graph_controller import SubGraphController
from nodetool.config.logging_config import get_logger
from nodetool.types.api_graph import Edge, Node, Graph as ApiGraph
from nodetool.workflows.base_node import BaseNode, ToolResultNode
from nodetool.workflows.graph import Graph
from nodetool.workflows.processing_context import ProcessingContext

log = get_logger(__name__)


class GraphTool(Tool):
    """
    Tool that executes a specific graph using its input schema.
    
    Supports two execution modes:
    
    1. **Transient mode** (default): Creates a new runner for each invocation.
       Use when the graph has no stateful nodes.
    
    2. **Persistent mode**: Maintains a running sub-graph across invocations.
       Use when nodes need to maintain state (e.g., code executor with REPL).
       Enable by setting `persistent=True` in the constructor.
    
    Example:
        ```python
        # Transient mode (default)
        tool = GraphTool(graph, "my_tool", "description", edges, nodes)
        
        # Persistent mode for stateful nodes
        tool = GraphTool(graph, "code_runner", "Execute code", edges, nodes, persistent=True)
        ```
    
    Attributes:
        graph: The Graph instance to execute
        name: Tool name for agent registration
        description: Tool description for LLM context
        initial_edges: Edges from external inputs to graph nodes
        initial_nodes: Nodes receiving external inputs
        persistent: Whether to maintain actors across invocations
    """
    
    def __init__(
        self,
        graph: Graph,
        name: str,
        description: str,
        initial_edges: list[Edge],
        initial_nodes: list[BaseNode],
        persistent: bool = False,
        result_timeout: float | None = 300.0,  # 5 minute default timeout
    ):
        """
        Initialize the GraphTool.
        
        Args:
            graph: The graph to execute
            name: Tool name
            description: Tool description
            initial_edges: Edges defining input injection points
            initial_nodes: Nodes receiving inputs
            persistent: If True, maintain running actors across invocations
            result_timeout: Timeout in seconds for waiting for results (persistent mode only)
        """
        self.graph = graph
        self.name = name
        self.description = description
        self.initial_edges = initial_edges
        self.initial_nodes = initial_nodes
        self.persistent = persistent
        self.result_timeout = result_timeout
        
        # Build input schema from initial edges/nodes
        def get_property_schema(node: BaseNode, handle: str) -> dict[str, Any]:
            for prop in node.properties():
                if prop.name == handle:
                    schema = prop.type.get_json_schema()
                    schema["description"] = prop.description or ""
                    return schema
            raise ValueError(
                f"Property {handle} not found on node {node.get_node_type()} {node.id}"
            )
        
        self.input_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                edge.targetHandle: get_property_schema(node, edge.targetHandle)
                for edge, node in zip(self.initial_edges, self.initial_nodes, strict=False)
            },
        }
        
        # Persistent execution state (lazy initialized)
        self._controller: SubGraphController | None = None
        self._api_graph: ApiGraph | None = None
    
    def _build_api_graph(self, params: dict[str, Any]) -> tuple[list[Node], list[Edge]]:
        """
        Build the API graph representation for execution.
        
        Args:
            params: Input parameters (used for property injection in transient mode)
        
        Returns:
            Tuple of (nodes, edges) for the API graph
        """
        initial_edges_by_target = {edge.target: edge for edge in self.initial_edges}
        excluded_source_ids = {edge.source for edge in self.initial_edges}
        
        def properties_for_node(node: BaseNode) -> dict[str, Any]:
            props = node.node_properties()
            if node.id in initial_edges_by_target:
                edge = initial_edges_by_target[node.id]
                # In persistent mode, don't inject params here - they come via inbox
                if not self.persistent and edge.targetHandle in params:
                    props[edge.targetHandle] = params[edge.targetHandle]
            return props
        
        # Build node list
        nodes = [
            Node(
                id=node.id,
                type=node.get_node_type(),
                data=properties_for_node(node),
                parent_id=node.parent_id,
                ui_properties=node.ui_properties,
                dynamic_properties=node.dynamic_properties,
                dynamic_outputs=node.dynamic_outputs,
            )
            for node in self.graph.nodes
            if node.id not in excluded_source_ids
        ]
        
        # Build edge list
        edges = [
            Edge(
                id=edge.id,
                source=edge.source,
                target=edge.target,
                sourceHandle=edge.sourceHandle,
                targetHandle=edge.targetHandle,
                ui_properties=edge.ui_properties,
            )
            for edge in self.graph.edges
            if edge.source not in excluded_source_ids and edge.target not in excluded_source_ids
        ]
        
        # Ensure ToolResultNode exists
        has_tool_result = any(
            isinstance(node, ToolResultNode)
            for node in self.graph.nodes
            if node.id not in excluded_source_ids
        )
        
        if not has_tool_result:
            # Auto-add ToolResultNode for single-node graphs
            remaining_nodes = [n for n in self.graph.nodes if n.id not in excluded_source_ids]
            if len(remaining_nodes) == 1:
                single_node = remaining_nodes[0]
                result_node_id = uuid4().hex
                
                nodes.append(
                    Node(
                        id=result_node_id,
                        type=ToolResultNode.get_node_type(),
                        data={},
                        parent_id=None,
                        ui_properties={},
                        dynamic_properties={},
                        dynamic_outputs={},
                    )
                )
                
                for output_slot in single_node.outputs_for_instance():
                    edges.append(
                        Edge(
                            id=uuid4().hex,
                            source=single_node.id,
                            target=result_node_id,
                            sourceHandle=output_slot.name,
                            targetHandle=output_slot.name,
                            ui_properties={},
                        )
                    )
        
        return nodes, edges
    
    async def _ensure_controller_started(self, context: ProcessingContext) -> None:
        """
        Ensure the SubGraphController is started (persistent mode only).
        
        This lazily initializes the controller on first invocation.
        """
        if self._controller is not None and self._controller.is_running:
            return
        
        # Build API graph
        nodes, edges = self._build_api_graph({})
        self._api_graph = ApiGraph(nodes=nodes, edges=edges)
        
        # Create and start controller
        self._controller = SubGraphController(
            api_graph=self._api_graph,
            input_edges=self.initial_edges,
            parent_context=context,
        )
        
        await self._controller.start()
        log.info("GraphTool '%s' started persistent controller", self.name)
    
    async def process(self, context: ProcessingContext, params: dict[str, Any]) -> Any:
        """
        Execute the graph with the provided parameters.
        
        Args:
            context: The processing context
            params: Input parameters matching the input schema
        
        Returns:
            The result from ToolResultNode
        """
        if self.persistent:
            return await self._process_persistent(context, params)
        else:
            return await self._process_transient(context, params)
    
    async def _process_persistent(
        self,
        context: ProcessingContext,
        params: dict[str, Any],
    ) -> Any:
        """
        Execute in persistent mode using SubGraphController.
        
        The sub-graph stays running across invocations.
        """
        await self._ensure_controller_started(context)
        
        assert self._controller is not None
        
        try:
            result = await self._controller.inject_and_wait(
                params=params,
                timeout=self.result_timeout,
            )
            
            # Normalize result values
            normalized = {}
            for key, value in result.items():
                if hasattr(value, "model_dump"):
                    value = value.model_dump()
                normalized[key] = value
            
            log.debug("GraphTool '%s' persistent result: %s", self.name, list(normalized.keys()))
            return normalized
            
        except Exception as e:
            log.error("GraphTool '%s' persistent execution error: %s", self.name, e)
            return {"error": str(e)}
    
    async def _process_transient(
        self,
        context: ProcessingContext,
        params: dict[str, Any],
    ) -> Any:
        """
        Execute in transient mode (original behavior).
        
        Creates a new runner for each invocation.
        """
        from nodetool.workflows.run_workflow import run_workflow
        from nodetool.workflows.run_job_request import RunJobRequest
        from nodetool.workflows.workflow_runner import WorkflowRunner
        from nodetool.workflows.types import JobUpdate, ToolResultUpdate, OutputUpdate
        from nodetool.workflows.processing_context import AssetOutputMode
        import queue as _queue
        
        nodes, edges = self._build_api_graph(params)
        
        # Check for ToolResultNode
        has_tool_result = any(
            n.type == ToolResultNode.get_node_type() for n in nodes
        )
        
        try:
            req = RunJobRequest(
                user_id=context.user_id,
                auth_token=context.auth_token,
                graph=ApiGraph(nodes=nodes, edges=edges),
            )
            
            isolated_queue: _queue.Queue = _queue.Queue()
            sub_context = ProcessingContext(
                user_id=context.user_id,
                auth_token=context.auth_token,
                graph=Graph.from_dict(req.graph.model_dump()),
                message_queue=isolated_queue,
                device=context.device,
                workspace_dir=context.workspace_dir,
                asset_output_mode=getattr(context, "asset_output_mode", AssetOutputMode.TEMP_URL),
            )
            
            result = {}
            runner = WorkflowRunner(job_id=uuid4().hex, disable_caching=True)
            
            # Find leaf node for fallback result capture
            leaf_node_id: str | None = None
            leaf_output_slot: str | None = None
            if not has_tool_result:
                nodes_with_outgoing = {edge.source for edge in edges}
                leaf_nodes = [
                    node for node in self.graph.nodes
                    if node.id not in {e.source for e in self.initial_edges}
                    and node.id not in nodes_with_outgoing
                ]
                if len(leaf_nodes) == 1:
                    outputs = leaf_nodes[0].outputs_for_instance()
                    for o in outputs:
                        if o.name == "output":
                            leaf_node_id = leaf_nodes[0].id
                            leaf_output_slot = "output"
                            break
                    if leaf_output_slot is None and len(outputs) == 1:
                        leaf_node_id = leaf_nodes[0].id
                        leaf_output_slot = outputs[0].name
            
            async for msg in run_workflow(
                request=req,
                runner=runner,
                context=sub_context,
                use_thread=True,
                send_job_updates=False,
                initialize_graph=False,
                validate_graph=False,
            ):
                if not isinstance(msg, JobUpdate):
                    context.post_message(msg)
                
                if isinstance(msg, ToolResultUpdate):
                    if msg.result is not None:
                        for key, value in msg.result.items():
                            if hasattr(value, "model_dump"):
                                value = value.model_dump()
                            if result.get(key) is None:
                                result[key] = value
                            elif isinstance(result[key], list):
                                result[key].append(value)
                            elif isinstance(result[key], str):
                                result[key] += value
                            else:
                                result[key] = value
                
                elif not has_tool_result and isinstance(msg, OutputUpdate):
                    if msg.node_id == leaf_node_id and msg.output_name == leaf_output_slot:
                        value = msg.value
                        if hasattr(value, "model_dump"):
                            value = value.model_dump()
                        key = leaf_output_slot if leaf_output_slot != "output" else "output"
                        result[key] = value
            
            return result
            
        except Exception as e:
            log.error("GraphTool '%s' transient execution error: %s", self.name, e)
            return {"error": str(e)}
    
    async def cleanup(self) -> None:
        """
        Clean up persistent resources.
        
        Should be called when the tool is no longer needed.
        """
        if self._controller is not None:
            await self._controller.stop()
            self._controller = None
        self._api_graph = None
    
    def __del__(self):
        """Destructor - attempt to clean up if not done explicitly."""
        if self._controller is not None:
            log.warning(
                "GraphTool '%s' was not cleaned up explicitly; "
                "call cleanup() to properly release resources",
                self.name,
            )
```

### 3. Modified `WorkflowRunner` for Actor Sharing

Add support for shared actors in `WorkflowRunner`:

```python name=src/nodetool/workflows/workflow_runner.py (additions)
class WorkflowRunner:
    """
    Actor-based DAG execution engine for computational nodes.
    
    [... existing docstring ...]
    
    New in v2: Actor Sharing
        The runner can optionally share its actor pool with sub-runners via
        the `shared_actor_registry` parameter. This enables persistent actors
        for sub-graph tool execution.
    """
    
    def __init__(
        self,
        job_id: str,
        device: str | None = None,
        disable_caching: bool = False,
        buffer_limit: int | None = 3,
        # NEW: Actor sharing support
        shared_actor_registry: dict[str, NodeActor] | None = None,
        shared_inboxes: dict[str, NodeInbox] | None = None,
    ):
        """
        Initializes a new WorkflowRunner instance.
        
        Args:
            job_id: Unique identifier for this workflow execution.
            device: The device to run on ("cpu", "cuda", "mps"). Auto-detects if None.
            disable_caching: Whether to disable result caching for cacheable nodes.
            buffer_limit: Maximum inbox buffer size for backpressure. None = unlimited.
            shared_actor_registry: Optional external actor registry for actor sharing.
                                   If provided, actors in this registry will be reused.
            shared_inboxes: Optional external inbox registry for inbox sharing.
                           Must be provided if shared_actor_registry is provided.
        """
        # ... existing init code ...
        
        # Actor sharing
        self._shared_actor_registry = shared_actor_registry
        self._shared_inboxes = shared_inboxes
        self._owns_actors: set[str] = set()  # Track actors we created (vs shared)
    
    def get_or_create_actor(
        self,
        node: BaseNode,
        context: ProcessingContext,
        inbox: NodeInbox,
    ) -> tuple[NodeActor, bool]:
        """
        Get an existing shared actor or create a new one.
        
        Args:
            node: The node to get/create an actor for
            context: The processing context
            inbox: The node's inbox
        
        Returns:
            Tuple of (actor, is_new) where is_new indicates if we created it
        """
        from nodetool.workflows.actor import NodeActor
        
        node_id = node._id
        
        # Check shared registry first
        if self._shared_actor_registry is not None and node_id in self._shared_actor_registry:
            return self._shared_actor_registry[node_id], False
        
        # Create new actor
        actor = NodeActor(self, node, context, inbox)
        self._owns_actors.add(node_id)
        
        # Register in shared registry if available
        if self._shared_actor_registry is not None:
            self._shared_actor_registry[node_id] = actor
        
        return actor, True
    
    async def process_graph(
        self,
        context: ProcessingContext,
        graph: Graph,
        parent_id: str | None = None,
    ) -> None:
        """
        Actor-based processing with support for shared actors.
        
        [... existing docstring ...]
        """
        from nodetool.workflows.actor import NodeActor
        
        log.info(
            "Processing graph (%d nodes, %d edges)",
            len(graph.nodes),
            len(graph.edges),
        )
        
        # Initialize inboxes (may use shared inboxes)
        if self._shared_inboxes is not None:
            # Merge shared inboxes with local ones
            for node in graph.nodes:
                if node._id in self._shared_inboxes:
                    self.node_inboxes[node._id] = self._shared_inboxes[node._id]
        
        self._initialize_inboxes(context, graph)
        
        # ... existing node state loading for resumption ...
        
        tasks: list[asyncio.Task] = []
        
        for node in graph.nodes:
            # Skip if actor already exists and is running (shared actor)
            if self._shared_actor_registry is not None and node._id in self._shared_actor_registry:
                existing_actor = self._shared_actor_registry[node._id]
                if existing_actor._task is not None and not existing_actor._task.done():
                    log.debug(
                        "Reusing existing actor for node: %s (%s)",
                        node.get_title(),
                        node._id,
                    )
                    continue
            
            inbox = self.node_inboxes.get(node._id)
            if inbox is None:
                log.warning("No inbox for node %s, skipping", node._id)
                continue
            
            actor, is_new = self.get_or_create_actor(node, context, inbox)
            
            if is_new:
                task = asyncio.create_task(
                    actor.run(),
                    name=f"actor-{node._id}",
                )
                actor._task = task
                tasks.append(task)
                self.active_processing_node_ids.add(node._id)
        
        # ... rest of existing process_graph implementation ...
```

### 4. Modifications to `NodeActor` for Continuous Input

Update `NodeActor` to support continuous input injection:

```python name=src/nodetool/workflows/actor.py (additions)
class NodeActor:
    """
    Drives a single node to completion.
    
    [... existing docstring ...]
    
    Continuous Mode:
        For persistent sub-graphs, actors can be configured to run in "continuous mode"
        where they don't terminate after processing but wait for more inputs.
        This is controlled by the `continuous` flag.
    """
    
    def __init__(
        self,
        runner: WorkflowRunner,
        node: BaseNode,
        context: ProcessingContext,
        inbox: NodeInbox,
        continuous: bool = False,
    ) -> None:
        """
        Initialize the actor.
        
        Args:
            runner: The active WorkflowRunner instance
            node: The node instance to execute
            context: The processing context
            inbox: The per-node inbox providing input values
            continuous: If True, don't terminate after processing; wait for more inputs
        """
        self.runner = runner
        self.node = node
        self.context = context
        self.inbox = inbox
        self.continuous = continuous
        self._task: asyncio.Task | None = None
        self._stop_requested = False
        self.logger = get_logger(__name__)
    
    def request_stop(self) -> None:
        """Request the actor to stop after current processing completes."""
        self._stop_requested = True
    
    async def _run_continuous_node(self) -> None:
        """
        Run node in continuous mode, processing inputs until stop is requested.
        
        In continuous mode, the actor:
        1. Waits for inputs to arrive
        2. Processes them through the node
        3. Sends outputs downstream
        4. Returns to step 1 (instead of terminating)
        """
        node = self.node
        handles = self._effective_inbound_handles()
        
        self.logger.info(
            "Starting continuous actor for node: %s (%s)",
            node.get_title(),
            node._id,
        )
        
        while not self._stop_requested:
            try:
                # Wait for next input batch
                inputs = await self._gather_next_input_batch(handles)
                
                if not inputs and self._stop_requested:
                    break
                
                if inputs:
                    # Process the batch
                    await self.process_node_with_inputs(inputs)
                
            except asyncio.CancelledError:
                self.logger.info(
                    "Continuous actor cancelled: %s (%s)",
                    node.get_title(),
                    node._id,
                )
                break
        
        await self._mark_downstream_eos()
    
    async def _gather_next_input_batch(
        self,
        handles: set[str],
        timeout: float = 1.0,
    ) -> dict[str, Any]:
        """
        Wait for the next batch of inputs (with timeout for stop checking).
        
        Args:
            handles: The input handles to gather from
            timeout: How long to wait before checking stop flag
        
        Returns:
            Input dictionary, or empty dict if nothing arrived
        """
        if not handles:
            # No inputs - wait a bit then check stop flag
            await asyncio.sleep(timeout)
            return {}
        
        # Wait for any input with timeout
        try:
            handle, value = await asyncio.wait_for(
                self.inbox.get_any(),
                timeout=timeout,
            )
            return {handle: value}
        except asyncio.TimeoutError:
            return {}
    
    async def run(self) -> None:
        """Entry point: choose execution mode and run."""
        if self.continuous:
            await self._run_continuous_node()
        else:
            # Existing behavior
            await self._run_standard()
    
    async def _run_standard(self) -> None:
        """Original run implementation (renamed from run)."""
        # ... existing run() implementation moved here ...
```

### 5. Updates to `NodeInbox` for Continuous Input

Add method for getting any available input:

```python name=src/nodetool/workflows/inbox.py (additions)
class NodeInbox:
    """
    Per-node inbox for receiving inputs from upstream nodes.
    
    [... existing docstring ...]
    """
    
    async def get_any(self) -> tuple[str, Any]:
        """
        Wait for and return the next available input from any handle.
        
        Unlike iter_any(), this returns a single item and can be called
        repeatedly for continuous input processing.
        
        Returns:
            Tuple of (handle_name, value)
        
        Raises:
            StopAsyncIteration: If all handles are closed with no pending items
        """
        # Implementation similar to iter_any but returns single item
        while True:
            # Check all handles for pending items
            for handle in list(self._buffers.keys()):
                if self._buffers[handle]:
                    value = self._buffers[handle].popleft()
                    self._buffer_not_full[handle].set()
                    return handle, value
            
            # Check if all handles are closed
            if not any(self._open_sources.get(h, 0) > 0 for h in self._buffers):
                raise StopAsyncIteration
            
            # Wait for new item
            await self._item_available.wait()
            self._item_available.clear()
```

## Cleanup and Lifecycle Management

### Agent-Level Cleanup

The `Agent` class should track and clean up `GraphTool` instances:

```python name=src/nodetool/agents/agent.py (additions)
class Agent:
    """
    [... existing docstring ...]
    """
    
    def __init__(self, ...):
        # ... existing init ...
        self._persistent_tools: list[GraphTool] = []
    
    def register_persistent_tool(self, tool: GraphTool) -> None:
        """Register a persistent GraphTool for lifecycle management."""
        self._persistent_tools.append(tool)
    
    async def cleanup(self) -> None:
        """Clean up all persistent resources."""
        for tool in self._persistent_tools:
            await tool.cleanup()
        self._persistent_tools.clear()
    
    async def __aenter__(self) -> Agent:
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.cleanup()
```

## Testing Strategy

### Unit Tests

1. **SubGraphController tests**:
   - Start/stop lifecycle
   - Multiple inject_and_wait calls
   - Timeout handling
   - Error propagation

2. **GraphTool persistent mode tests**:
   - State preservation across calls
   - Cleanup behavior
   - Fallback to transient mode

3. **Actor sharing tests**:
   - Shared actor reuse
   - Inbox sharing
   - Proper cleanup

### Integration Tests

```python name=tests/agents/tools/test_persistent_graph_tool.py
import pytest
from nodetool.agents.tools.workflow_tool import GraphTool
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.graph import Graph


class StatefulCounter(BaseNode):
    """A node that maintains state across invocations."""
    _count: int = 0
    
    async def process(self, context) -> dict[str, int]:
        self._count += 1
        return {"count": self._count}


@pytest.mark.asyncio
async def test_persistent_graph_tool_preserves_state():
    """Verify that persistent mode preserves node state across calls."""
    counter = StatefulCounter(id="counter")
    graph = Graph(nodes=[counter], edges=[])
    
    tool = GraphTool(
        graph=graph,
        name="counter_tool",
        description="Counts invocations",
        initial_edges=[],
        initial_nodes=[],
        persistent=True,
    )
    
    context = ProcessingContext(user_id="test", auth_token="test")
    
    try:
        result1 = await tool.process(context, {})
        result2 = await tool.process(context, {})
        result3 = await tool.process(context, {})
        
        assert result1["count"] == 1
        assert result2["count"] == 2
        assert result3["count"] == 3
    finally:
        await tool.cleanup()


@pytest.mark.asyncio
async def test_transient_graph_tool_resets_state():
    """Verify that transient mode creates fresh state each call."""
    counter = StatefulCounter(id="counter")
    graph = Graph(nodes=[counter], edges=[])
    
    tool = GraphTool(
        graph=graph,
        name="counter_tool",
        description="Counts invocations",
        initial_edges=[],
        initial_nodes=[],
        persistent=False,  # Transient mode
    )
    
    context = ProcessingContext(user_id="test", auth_token="test")
    
    result1 = await tool.process(context, {})
    result2 = await tool.process(context, {})
    
    # Each call should start fresh
    assert result1["count"] == 1
    assert result2["count"] == 1
```

## Migration Guide

### Enabling Persistent Mode for Existing Tools

```python
# Before (transient):
tool = GraphTool(graph, "code_exec", "Execute code", edges, nodes)

# After (persistent):
tool = GraphTool(graph, "code_exec", "Execute code", edges, nodes, persistent=True)
```

### Required Cleanup

```python
# Always clean up persistent tools
async with agent:  # Uses __aenter__/__aexit__
    await agent.run(task)
# Cleanup happens automatically

# Or manually:
try:
    await agent.run(task)
finally:
    await agent.cleanup()
```

## Performance Considerations

| Aspect | Transient Mode | Persistent Mode |
|--------|---------------|-----------------|
| Startup latency | Full graph init per call | One-time init |
| Memory usage | Released after each call | Held until cleanup |
| GPU model loading | Per call | Once |
| State persistence | None | Full |
| Suitable for | Stateless tools | Stateful tools (REPL, etc.) |

## Security Considerations

1. **Resource limits**: Persistent actors consume memory; implement max lifetime/idle timeout
2. **Isolation**: Each agent should have its own actor pool (no cross-agent sharing)
3. **Cleanup on error**: Ensure cleanup happens even on agent errors

## Future Enhancements

1. **Actor checkpointing**: Save/restore actor state for process restart resilience
2. **Distributed actors**: Support actors running on different machines
3. **Actor pooling**: Reuse actors across similar graphs (not just same graph)
4. **Metrics**: Track actor lifetime, invocation counts, resource usage

## File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/nodetool/agents/tools/sub_graph_controller.py` | New | SubGraphController class |
| `src/nodetool/agents/tools/workflow_tool.py` | Modified | Add persistent mode to GraphTool |
| `src/nodetool/workflows/workflow_runner.py` | Modified | Add actor sharing support |
| `src/nodetool/workflows/actor.py` | Modified | Add continuous mode |
| `src/nodetool/workflows/inbox.py` | Modified | Add get_any() method |
| `src/nodetool/agents/agent.py` | Modified | Add cleanup lifecycle |
| `tests/agents/tools/test_persistent_graph_tool.py` | New | Integration tests |
| `tests/agents/tools/test_sub_graph_controller.py` | New | Unit tests |
