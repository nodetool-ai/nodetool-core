# Execution Plan: Agent Control Connections (Core Repository Only)

## Overview

This plan adds "control connections" to the nodetool-core repository that allow Agent nodes to dynamically set parameters of other nodes. The agent sees the controlled node's current parameters (including upstream data), can retry with trial history, and works with both streaming and non-streaming nodes.

**Key Design Decisions:**
- Control messages use reserved handle name `__control__`
- Multiple controllers allowed per node (last control message wins)
- Control params take precedence over data edge inputs

---

## Phase 1: Edge Model Extension

**File:** `src/nodetool/types/api_graph.py`

**Tasks:**
1. Add `edge_type` field to Edge model:
   ```python
   from typing import Literal
   
   class Edge(BaseModel):
       id: str | None = None
       source: str
       sourceHandle: str
       target: str
       targetHandle: str
       ui_properties: dict[str, str] | None = None
       edge_type: Literal["data", "control"] = "data"
   ```

2. Add helper method to check if edge is control type:
   ```python
   def is_control(self) -> bool:
       """Return True if this is a control edge."""
       return self.edge_type == "control"
   ```

---

## Phase 2: Graph Extensions

**File:** `src/nodetool/workflows/graph.py`

**Tasks:**

1. Add method to get all control edges targeting a node:
   ```python
   def get_control_edges(self, target_id: str) -> list[Edge]:
       """Return all control edges targeting the given node."""
       return [
           edge for edge in self.edges 
           if edge.target == target_id and edge.edge_type == "control"
       ]
   ```

2. Add method to get controller nodes:
   ```python
   def get_controller_nodes(self, target_id: str) -> list[BaseNode]:
       """Return all nodes that control the given target node."""
       control_edges = self.get_control_edges(target_id)
       controllers = []
       for edge in control_edges:
           node = self.find_node(edge.source)
           if node:
               controllers.append(node)
       return controllers
   ```

3. Add method to get controlled nodes:
   ```python
   def get_controlled_nodes(self, source_id: str) -> list[str]:
       """Return IDs of all nodes controlled by the given source."""
       return [
           edge.target for edge in self.edges 
           if edge.source == source_id and edge.edge_type == "control"
       ]
   ```

4. Create new validation method `validate_control_edges()`:
   ```python
   def validate_control_edges(self) -> list[str]:
       """
       Validate control edges in the graph.
       
       Rules:
       - Control edges must originate from Agent-type nodes
       - Control edges must target valid nodes
       - Control edges must use '__control__' as targetHandle
       - Circular control chains are forbidden
       
       Returns:
           List of validation error messages (empty if valid)
       """
       errors = []
       
       # Check each control edge
       for edge in self.edges:
           if edge.edge_type != "control":
               continue
           
           # Rule 1: Source must be an Agent node
           source_node = self.find_node(edge.source)
           if not source_node:
               errors.append(f"Control edge {edge.id} has invalid source {edge.source}")
               continue
           
           # Check if source is Agent-type (check node type string)
           if "agent" not in source_node.get_node_type().lower():
               errors.append(
                   f"Control edge {edge.id} source {edge.source} must be an Agent node, "
                   f"got {source_node.get_node_type()}"
               )
           
           # Rule 2: Target must exist
           target_node = self.find_node(edge.target)
           if not target_node:
               errors.append(f"Control edge {edge.id} has invalid target {edge.target}")
               continue
           
           # Rule 3: Must use __control__ as targetHandle
           if edge.targetHandle != "__control__":
               errors.append(
                   f"Control edge {edge.id} must use '__control__' as targetHandle, "
                   f"got '{edge.targetHandle}'"
               )
       
       # Rule 4: Check for circular control dependencies
       circular_errors = self._check_circular_control(self.edges)
       errors.extend(circular_errors)
       
       return errors
   ```

5. Add circular dependency detection:
   ```python
   def _check_circular_control(self, edges: Sequence[Edge]) -> list[str]:
       """
       Check for circular dependencies in control edges.
       
       Returns:
           List of error messages for circular dependencies
       """
       errors = []
       
       # Build control adjacency list
       control_graph: dict[str, list[str]] = defaultdict(list)
       for edge in edges:
           if edge.edge_type == "control":
               control_graph[edge.source].append(edge.target)
       
       # DFS to detect cycles
       def has_cycle(node: str, visited: set[str], rec_stack: set[str]) -> tuple[bool, list[str]]:
           visited.add(node)
           rec_stack.add(node)
           
           for neighbor in control_graph.get(node, []):
               if neighbor not in visited:
                   found, path = has_cycle(neighbor, visited, rec_stack)
                   if found:
                       return True, [node] + path
               elif neighbor in rec_stack:
                   return True, [node, neighbor]
           
           rec_stack.remove(node)
           return False, []
       
       visited: set[str] = set()
       for node_id in control_graph.keys():
           if node_id not in visited:
               found, path = has_cycle(node_id, visited, set())
               if found:
                   errors.append(
                       f"Circular control dependency detected: {' -> '.join(path)}"
                   )
       
       return errors
   ```

6. Update existing `validate_edge_types()` or create if doesn't exist:
   ```python
   def validate_edge_types(self) -> list[str]:
       """
       Validate edge types and connections.
       
       Returns:
           List of validation error messages (empty if valid)
       """
       errors = []
       
       # Existing validation logic for data edges...
       # (keep existing code if method exists)
       
       # Add control edge validation
       control_errors = self.validate_control_edges()
       errors.extend(control_errors)
       
       return errors
   ```

---

## Phase 3: WorkflowRunner Control Processing

**File:** `src/nodetool/workflows/workflow_runner.py`

**Tasks:**

1. Add instance variable to track control edges (add to `__init__`):
   ```python
   # In WorkflowRunner.__init__, after existing instance variables:
   
   # Control edges: {target_id: [control_edges]}
   self._control_edges: dict[str, list[Edge]] = defaultdict(list)
   ```

2. Add method to classify control edges:
   ```python
   def _classify_control_edges(self, graph: Graph) -> None:
       """
       Build lookup for control edges by target node.
       
       Populates self._control_edges with {target_id: [Edge, Edge, ...]}.
       Multiple controllers are allowed; they are processed in topological order.
       """
       self._control_edges.clear()
       
       for edge in graph.edges:
           if edge.edge_type == "control":
               self._control_edges[edge.target].append(edge)
       
       log.debug(
           f"Classified control edges: {len(self._control_edges)} nodes have controllers"
       )
       for target_id, edges in self._control_edges.items():
           log.debug(
               f"  Node {target_id} controlled by: {[e.source for e in edges]}"
           )
   ```

3. Add method to build control context for a node:
   ```python
   def _build_control_context(self, node: BaseNode, graph: Graph) -> dict[str, Any]:
       """
       Build context for control edge execution.
       
       Returns:
           Dictionary with:
           - node_id: str
           - node_type: str
           - properties: dict[str, dict] (name -> {value, type, description})
           - upstream_data: dict[str, Any] (handle -> value, for data edges with values)
       """
       context = {
           "node_id": node._id,
           "node_type": node.get_node_type(),
           "properties": {},
           "upstream_data": {}
       }
       
       # Gather property metadata from node class
       for prop in node.properties():
           context["properties"][prop.name] = {
               "value": getattr(node, prop.name, None),
               "type": str(prop.type),
               "description": prop.description or "",
               "default": prop.default
           }
       
       # Note: upstream_data is populated by NodeActor when data arrives
       # This method just sets up the structure
       
       return context
   ```

4. Modify `process_graph()` to classify control edges:
   ```python
   # In process_graph(), after graph validation and before creating actors:
   
   async def process_graph(...):
       # ... existing code ...
       
       # Classify control edges
       self._classify_control_edges(graph)
       
       # ... continue with existing code (topological sort, create actors, etc.) ...
   ```

---

## Phase 4: NodeActor Control Input Application

**File:** `src/nodetool/workflows/actor.py`

**Tasks:**

1. Add method to check if node has control edges:
   ```python
   def _has_control_edges(self) -> bool:
       """Return True if this node is controlled by control edges."""
       return self.node._id in self.runner._control_edges
   
   def _get_control_edges(self) -> list[Edge]:
       """Return control edges targeting this node."""
       return self.runner._control_edges.get(self.node._id, [])
   ```

2. Add method to wait for and collect control parameters:
   ```python
   async def _wait_for_control_params(self) -> dict[str, Any]:
       """
       Wait for control parameters from all controllers.
       
       Multiple controllers are allowed. Control params are merged in order,
       with later controllers overriding earlier ones.
       
       Returns:
           Merged dictionary of control parameters
       """
       control_edges = self._get_control_edges()
       if not control_edges:
           return {}
       
       merged_params: dict[str, Any] = {}
       
       for edge in control_edges:
           # Wait for control message on __control__ handle
           control_data = None
           async for item in self.inbox.iter_input("__control__"):
               # Check if this control message is from the expected controller
               # In practice, inbox should filter by source, but we validate here
               control_data = item
               break  # Take first message on __control__ handle
           
           if control_data and isinstance(control_data, dict):
               # Merge control params (later controllers override)
               merged_params.update(control_data)
               log.debug(
                   f"Node {self.node._id} received control params from {edge.source}: "
                   f"{list(control_data.keys())}"
               )
           else:
               log.warning(
                   f"Node {self.node._id} expected control params from {edge.source} "
                   f"but got invalid data: {type(control_data)}"
               )
       
       return merged_params
   ```

3. Add method to validate control parameters:
   ```python
   def _validate_control_params(self, params: dict[str, Any]) -> list[str]:
       """
       Validate control parameters against node properties.
       
       Args:
           params: Control parameters to validate
       
       Returns:
           List of validation errors (empty if valid)
       """
       errors = []
       
       # Get node property definitions
       property_map = {prop.name: prop for prop in self.node.properties()}
       
       for param_name, param_value in params.items():
           if param_name not in property_map:
               errors.append(
                   f"Control param '{param_name}' does not exist on node "
                   f"{self.node.get_node_type()}"
               )
               continue
           
           # Type checking would go here
           # For now, we rely on node.assign_property() to handle type conversion
       
       return errors
   ```

4. Modify `process_node_with_inputs()` to apply control params:
   ```python
   async def process_node_with_inputs(
       self,
       inputs: dict[str, Any],
   ) -> None:
       """
       Process a non-streaming node instance with resolved inputs.
       
       Modified to support control edges:
       1. Check for control edges
       2. Wait for and validate control parameters
       3. Merge control params with data inputs (control takes precedence)
       4. Execute node with merged inputs
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
           
           # NEW: Handle control edges
           if self._has_control_edges():
               span.set_attribute("nodetool.node.has_control_edges", True)
               
               # Wait for control parameters from all controllers
               control_params = await self._wait_for_control_params()
               
               if control_params:
                   # Validate control params
                   validation_errors = self._validate_control_params(control_params)
                   if validation_errors:
                       error_msg = "; ".join(validation_errors)
                       self.logger.error(
                           f"Control param validation failed for {self.node._id}: {error_msg}"
                       )
                       raise ValueError(f"Control parameter validation failed: {error_msg}")
                   
                   # Merge control params with data inputs
                   # Control params take precedence
                   inputs = {**inputs, **control_params}
                   span.set_attribute(
                       "nodetool.node.control_params_applied", 
                       list(control_params.keys())
                   )
                   self.logger.info(
                       f"Applied control params to {self.node._id}: {list(control_params.keys())}"
                   )
           
           # Continue with existing implementation
           await self._process_node_with_inputs_impl(inputs, span)
   ```

5. Modify `process_streaming_node_with_inputs()` similarly:
   ```python
   async def process_streaming_node_with_inputs(
       self,
       inputs: dict[str, Any],
   ) -> None:
       """
       Process a streaming-output node for one aligned batch of inputs.
       
       Modified to support control edges (same as non-streaming case).
       """
       context = self.context
       node = self.node

       self.logger.debug(
           "process_streaming_node_with_inputs for %s (%s) with inputs: %s",
           node.get_title(),
           node._id,
           list(inputs.keys()),
       )
       
       # NEW: Handle control edges (before assigning any inputs)
       if self._has_control_edges():
           control_params = await self._wait_for_control_params()
           
           if control_params:
               validation_errors = self._validate_control_params(control_params)
               if validation_errors:
                   error_msg = "; ".join(validation_errors)
                   self.logger.error(
                       f"Control param validation failed for {self.node._id}: {error_msg}"
                   )
                   raise ValueError(f"Control parameter validation failed: {error_msg}")
               
               # Merge control params with data inputs (control wins)
               inputs = {**inputs, **control_params}
               self.logger.info(
                   f"Applied control params to streaming node {self.node._id}: "
                   f"{list(control_params.keys())}"
               )
       
       # Continue with existing implementation
       for name, value in inputs.items():
           # ... existing code ...
   ```

6. Modify `_run_streaming_input_node()` for streaming input nodes:
   ```python
   async def _run_streaming_input_node(self) -> None:
       """
       Modified to apply control params before starting streaming.
       """
       node = self.node
       ctx = self.context

       # NEW: Apply control params before pre_process
       if self._has_control_edges():
           control_params = await self._wait_for_control_params()
           
           if control_params:
               validation_errors = self._validate_control_params(control_params)
               if validation_errors:
                   error_msg = "; ".join(validation_errors)
                   self.logger.error(
                       f"Control param validation failed for {self.node._id}: {error_msg}"
                   )
                   raise ValueError(f"Control parameter validation failed: {error_msg}")
               
               # Apply control params directly to node properties
               for param_name, param_value in control_params.items():
                   error = node.assign_property(param_name, param_value)
                   if error:
                       self.logger.error(
                           f"Error assigning control param {param_name}: {error}"
                       )
                   else:
                       self.logger.info(
                           f"Applied control param to streaming input node {self.node._id}: "
                           f"{param_name}={param_value}"
                       )
       
       # Continue with existing implementation
       await node.pre_process(ctx)
       # ... rest of existing code ...
   ```

---

## Phase 5: Inbox Handling for Control Messages

**File:** `src/nodetool/workflows/inbox.py` (if modifications needed)

**Note:** The existing `NodeInbox` should already support the `__control__` handle as a regular handle. No changes should be needed, but verify that:

1. The inbox can receive messages on `__control__` handle
2. Messages are properly queued per-handle
3. `iter_input("__control__")` works as expected

If modifications are needed:

**Tasks:**
1. Ensure `__control__` is treated as a valid handle
2. Add logging for control message delivery (optional, for debugging)

---

## Phase 6: WorkflowRunner Message Routing

**File:** `src/nodetool/workflows/workflow_runner.py`

**Tasks:**

1. Modify `send_messages()` to handle control edge routing:
   ```python
   async def send_messages(
       self,
       node: BaseNode,
       result: dict[str, Any],
       context: ProcessingContext,
   ) -> None:
       """
       Modified to route control messages to __control__ handle.
       
       When a node outputs control parameters (result contains special key),
       route them to controlled nodes via __control__ handle.
       """
       # Check if this node is a controller (has outgoing control edges)
       controlled_node_ids = [
           edge.target for edge in context.graph.edges
           if edge.source == node._id and edge.edge_type == "control"
       ]
       
       if controlled_node_ids and "__control_output__" in result:
           # This node is outputting control parameters
           control_params = result["__control_output__"]
           
           if not isinstance(control_params, dict):
               log.error(
                   f"Node {node._id} output invalid control params: "
                   f"expected dict, got {type(control_params)}"
               )
           else:
               # Route control params to each controlled node
               for target_id in controlled_node_ids:
                   inbox = self.node_inboxes.get(target_id)
                   if inbox:
                       await inbox.put("__control__", control_params)
                       log.debug(
                           f"Routed control params from {node._id} to {target_id}: "
                           f"{list(control_params.keys())}"
                       )
       
       # Continue with existing data edge routing
       for key, value in result.items():
           if key == "__control_output__":
               continue  # Already handled above
           
           # ... existing routing logic for data edges ...
   ```

---

## Phase 7: API/Serialization Updates

**File:** `src/nodetool/types/api_graph.py`

**Tasks:**
1. Ensure Pydantic serialization includes `edge_type` (already done in Phase 1)

**File:** `src/nodetool/api/workflow.py` (or wherever workflow APIs are defined)

**Tasks:**
1. Update API endpoints to accept `edge_type` in edge creation/update:
   - `POST /api/workflows/{id}/edges`
   - `PUT /api/workflows/{id}/edges/{edge_id}`

2. Add validation in API layer to call `graph.validate_control_edges()`

3. Update any JSON schema generation for frontend

**File:** Database migrations (if edges are persisted)

**Tasks:**
1. Add `edge_type` column to edges table (if applicable)
2. Set default value to `"data"` for existing edges
3. Add index on `edge_type` if needed for performance

---

## Phase 8: Testing

**File:** `tests/workflows/test_control_edges.py` (new file)

**Test Cases:**

### Basic Control Edge Tests
```python
def test_control_edge_model():
    """Test Edge model with control type."""
    edge = Edge(
        id="e1",
        source="agent1",
        sourceHandle="output",
        target="node2",
        targetHandle="__control__",
        edge_type="control"
    )
    assert edge.is_control()
    assert edge.edge_type == "control"

def test_data_edge_model():
    """Test Edge model with data type (default)."""
    edge = Edge(
        id="e1",
        source="node1",
        sourceHandle="output",
        target="node2",
        targetHandle="input"
    )
    assert not edge.is_control()
    assert edge.edge_type == "data"
```

### Graph Validation Tests
```python
def test_validate_control_edge_target_handle():
    """Control edge must use __control__ as targetHandle."""
    agent = AgentNode(id="agent1")
    node = ProcessingNode(id="node2")
    edge = Edge(
        id="e1",
        source="agent1",
        sourceHandle="output",
        target="node2",
        targetHandle="wrong_handle",  # Invalid
        edge_type="control"
    )
    graph = Graph(nodes=[agent, node], edges=[edge])
    errors = graph.validate_control_edges()
    assert len(errors) > 0
    assert "__control__" in errors[0]

def test_validate_control_edge_from_non_agent():
    """Control edge must originate from Agent node."""
    node1 = ProcessingNode(id="node1")
    node2 = ProcessingNode(id="node2")
    edge = Edge(
        id="e1",
        source="node1",
        sourceHandle="output",
        target="node2",
        targetHandle="__control__",
        edge_type="control"
    )
    graph = Graph(nodes=[node1, node2], edges=[edge])
    errors = graph.validate_control_edges()
    assert len(errors) > 0
    assert "Agent" in errors[0]

def test_validate_circular_control_dependency():
    """Circular control dependencies should be detected."""
    agent1 = AgentNode(id="agent1")
    agent2 = AgentNode(id="agent2")
    edges = [
        Edge(
            id="e1",
            source="agent1",
            sourceHandle="output",
            target="agent2",
            targetHandle="__control__",
            edge_type="control"
        ),
        Edge(
            id="e2",
            source="agent2",
            sourceHandle="output",
            target="agent1",
            targetHandle="__control__",
            edge_type="control"
        )
    ]
    graph = Graph(nodes=[agent1, agent2], edges=edges)
    errors = graph.validate_control_edges()
    assert len(errors) > 0
    assert "circular" in errors[0].lower()

def test_multiple_controllers_allowed():
    """Multiple controllers for same node should be allowed."""
    agent1 = AgentNode(id="agent1")
    agent2 = AgentNode(id="agent2")
    target = ProcessingNode(id="target")
    edges = [
        Edge(
            id="e1",
            source="agent1",
            sourceHandle="output",
            target="target",
            targetHandle="__control__",
            edge_type="control"
        ),
        Edge(
            id="e2",
            source="agent2",
            sourceHandle="output",
            target="target",
            targetHandle="__control__",
            edge_type="control"
        )
    ]
    graph = Graph(nodes=[agent1, agent2, target], edges=edges)
    errors = graph.validate_control_edges()
    assert len(errors) == 0  # Should be valid
```

### Control Parameter Application Tests
```python
@pytest.mark.asyncio
async def test_control_params_override_data_inputs():
    """Control params should take precedence over data inputs."""
    # Setup: Agent outputs control params, target receives both data and control
    # Expected: Control params override data inputs
    pass

@pytest.mark.asyncio
async def test_multiple_controllers_merge_params():
    """Multiple controllers should merge params, last wins."""
    # Setup: Two agents control same node, both output different params
    # Expected: Later controller's params override earlier ones
    pass

@pytest.mark.asyncio
async def test_control_params_validation():
    """Invalid control params should raise validation error."""
    # Setup: Agent outputs params with wrong types
    # Expected: Validation error before node execution
    pass
```

### Streaming Node Tests
```python
@pytest.mark.asyncio
async def test_control_streaming_output_node():
    """Control params applied once before streaming starts."""
    pass

@pytest.mark.asyncio
async def test_control_streaming_input_node():
    """Control params applied before streaming input begins."""
    pass
```

### Integration Tests
```python
@pytest.mark.asyncio
async def test_chained_control():
    """Agent A controls Agent B, Agent B controls Node C."""
    pass

@pytest.mark.asyncio
async def test_control_with_upstream_data():
    """Control agent sees upstream data in context."""
    pass
```

---

## Implementation Notes

### Control Parameter Output Format

Agent nodes should output control parameters in their result dict with special key:

```python
result = {
    "__control_output__": {
        "param1": value1,
        "param2": value2,
        # ... control parameters to apply to controlled node
    },
    # ... other regular outputs ...
}
```

### Control Context Structure

When building control context (for agent to see):

```python
{
    "node_id": "node_123",
    "node_type": "nodetool.processing.Transform",
    "properties": {
        "threshold": {
            "value": 0.5,
            "type": "float",
            "description": "Threshold for processing",
            "default": 0.0
        },
        "mode": {
            "value": "fast",
            "type": "str",
            "description": "Processing mode",
            "default": "normal"
        }
    },
    "upstream_data": {
        "input_handle_1": <value>,
        "input_handle_2": <value>
    }
}
```

### Execution Order with Multiple Controllers

When multiple controllers target the same node:
1. Topological sort ensures all controllers execute before target
2. Control params are merged in order (implementation-defined order, typically edge creation order)
3. Later controllers override earlier ones (dict merge semantics)

### Control Edge vs. Data Edge Priority

When a property receives input from both data edge and control edge:
```python
# Initial value from data edge
inputs = {"param1": "from_data_edge"}

# Control params override
control_params = {"param1": "from_control_edge"}
inputs = {**inputs, **control_params}

# Final value: "from_control_edge"
```

---

## Execution Order Summary

```
Phase 1: Edge Model Extension
  └─ src/nodetool/types/api_graph.py
  └─ Add edge_type field with Literal["data", "control"]
  └─ Add is_control() method

Phase 2: Graph Extensions
  └─ src/nodetool/workflows/graph.py
  └─ Add get_control_edges(), get_controller_nodes(), get_controlled_nodes()
  └─ Add validate_control_edges() with circular dependency detection
  └─ Update validate_edge_types() to include control validation

Phase 3: WorkflowRunner Control Processing
  └─ src/nodetool/workflows/workflow_runner.py
  └─ Add _control_edges instance variable
  └─ Add _classify_control_edges()
  └─ Add _build_control_context()
  └─ Call _classify_control_edges() in process_graph()
  └─ Modify send_messages() to route control params

Phase 4: NodeActor Control Input Application
  └─ src/nodetool/workflows/actor.py
  └─ Add _has_control_edges(), _get_control_edges()
  └─ Add _wait_for_control_params()
  └─ Add _validate_control_params()
  └─ Modify process_node_with_inputs() to apply control params
  └─ Modify process_streaming_node_with_inputs() to apply control params
  └─ Modify _run_streaming_input_node() to apply control params

Phase 5: Inbox Handling
  └─ src/nodetool/workflows/inbox.py
  └─ Verify __control__ handle support (likely no changes needed)

Phase 6: WorkflowRunner Message Routing
  └─ src/nodetool/workflows/workflow_runner.py
  └─ Modify send_messages() to handle __control_output__

Phase 7: API/Serialization
  └─ src/nodetool/types/api_graph.py (already done in Phase 1)
  └─ src/nodetool/api/workflow.py (update endpoints)
  └─ Database migrations (if applicable)

Phase 8: Testing
  └─ tests/workflows/test_control_edges.py
  └─ Basic model tests
  └─ Graph validation tests
  └─ Control parameter application tests
  └─ Streaming node tests
  └─ Integration tests
```

---

## Key Differences from Original Plan

1. **Multiple Controllers Allowed**: Changed from "forbid multiple controllers" to "allow multiple, merge params with later override"

2. **Control Handle Name**: Using `__control__` as reserved handle name instead of special inbox mechanism

3. **No Backward Signaling**: Removed retry coordination from Phase 4 (agents handle retries internally in nodetool-base)

4. **Simplified Validation**: No check for "multiple controllers" since it's now allowed

5. **Core Repository Only**: Removed all agent node modifications (Phase 5 from original plan) - these belong in nodetool-base repository
