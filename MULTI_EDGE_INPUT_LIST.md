I# Technical Design:  Multiple Edges Feeding into Single List Input Edge

## Executive Summary

This document proposes a design for simplifying edge handling in nodetool-core by enabling **multiple edges to feed into a single list input property** without requiring intermediate collect nodes.  Currently, users must explicitly insert a `Collect` node when multiple outputs need to merge into a list-type input.  The proposed solution will automatically aggregate values from multiple incoming edges into a list when the target property is of type `list[T]`.

## 1. Current State Analysis

### 1.1 Edge and Connection Model

**Key Files:**
- `src/nodetool/types/graph. py`: Defines `Edge` and `Node` data structures
- `src/nodetool/workflows/graph. py`: Graph validation and traversal
- `src/nodetool/workflows/base_node.py`: Node property definitions and validation
- `src/nodetool/workflows/workflow_runner.py`: Execution engine with actor model

**Current Edge Structure:**
```python
class Edge(BaseModel):
    id: str | None = None
    source: str              # Source node ID
    sourceHandle: str        # Output slot name
    target: str              # Target node ID
    targetHandle: str        # Input slot/property name
    ui_properties: dict[str, str] | None = None
```

**Current Constraints:**
- **One-to-one mapping**: Each edge connects one source handle to one target handle
- **Property overwriting**: When multiple edges target the same property, the last value wins (via `assign_property`)
- **Manual aggregation required**: Users must insert `Collect` nodes to merge multiple streams

### 1.2 Input Handling Architecture

The system uses a **per-node inbox** model with two synchronization modes: 

**NodeInbox** (`src/nodetool/workflows/inbox.py`):
- Per-handle FIFO buffers
- Tracks upstream source counts for EOS (End-of-Stream)
- Supports backpressure via buffer limits
- Methods: `put(handle, item)`, `mark_source_done(handle)`, `iter_input(handle)`

**Synchronization Modes** (non-streaming nodes):
1. **`on_any`** (default): Fires on every arrival, using latest values from other handles (like RxJS `combineLatest`)
2. **`zip_all`**: Waits for one value from each handle, then consumes aligned tuples

**Key Observation**: Multiple edges can already target the same `targetHandle` at the inbox level.  The runner's `send_messages` delivers each value to `inbox.put(targetHandle, value)`, creating a stream of values per handle.  The challenge is **how to present this stream to the property** when the property is a list type.

### 1.3 Streaming vs Non-Streaming

**Streaming Propagation** (`_analyze_streaming` in `workflow_runner.py`):
- Streaming outputs are detected via `is_streaming_output()`
- BFS marks all downstream edges as streaming
- Non-streaming nodes consume via actor batching
- Streaming-input nodes (`is_streaming_input() = True`) manually drain inbox via `iter_input()`/`iter_any()`

**Current State**: 
- Non-streaming nodes with multiple incoming edges on the same handle see only the **last** value via `assign_property`
- Streaming-input nodes can manually consume all values but must implement custom aggregation logic

### 1.4 Property Validation

**Property Definition** (`src/nodetool/workflows/property.py`):
```python
class Property(BaseModel):
    name: str
    type: TypeMetadata        # Includes list[T] detection
    default: Optional[Any]
    required: bool
```

**Type Checking** (`src/nodetool/metadata/typecheck.py`):
- `typecheck(type1, type2)`: Validates type compatibility
- List types:  `type == "list"` with `type_args` for element type
- Handles `list[any]`, `list[int]`, etc. 

**Current Validation** (`validate_inputs` in `base_node.py`):
- Checks for **missing required inputs** (at least one edge exists)
- Does **not** enforce cardinality (one vs many edges per property)
- Edge type compatibility checked at graph level

## 2. Problem Statement and Use Cases

### 2.1 User Pain Points

**Scenario 1: Simple Fan-In (Non-Streaming → Non-Streaming)**
```
NodeA (output:  int) ─┐
NodeB (output: int) ─┼─→ ProcessList (items: list[int])
NodeC (output: int) ─┘
```
**Current**:  Requires intermediate `Collect` node
**Desired**: Direct connections, automatic list formation

**Scenario 2: Streaming Fan-In (Streaming → Non-Streaming)**
```
StreamingAgent (chunk: str) ─┐
                              ├─→ Aggregate (texts: list[str])
Logger (log: str) ────────────┘
```
**Current**: Unclear behavior; likely drops values
**Desired**: Collect all streamed values from both sources

**Scenario 3: Mixed Streaming (Streaming → Streaming)**
```
StreamingProducerA ─┐
StreamingProducerB ─┼─→ StreamingConsumer (is_streaming_input=True)
StreamingProducerC ─┘
```
**Current**: Node must manually `iter_any()` and aggregate
**Desired**: Optionally receive pre-aggregated list

### 2.2 Edge Cases

1. **Streaming + Non-Streaming Mix**:  What if one upstream is streaming, another is not?
2. **Timing**:  When do we "close" the list? Wait for all upstreams to finish (EOS)?
3. **Ordering**:  Arrival order?  Source order? Deterministic?
4. **Type Mismatch**: Multiple edges with incompatible element types feeding `list[T]`
5. **Backward Compatibility**:  Existing graphs with single edges to list properties must not break

## 3. Proposed Solution

### 3.1 Design Principles

1. **Implicit Aggregation**: Multiple edges to a `list[T]` property auto-collect without explicit collect nodes
2. **Type Safety**: Edge validation ensures all source types are compatible with `T`
3. **Streaming-Aware**: Behavior varies based on upstream streaming characteristics
4. **Backward Compatible**: Single-edge-to-list remains unchanged
5. **Clear EOS Semantics**: Aggregation completes when all upstreams signal EOS

### 3.2 Core Mechanism

#### 3.2.1 Detection Logic

**In `workflow_runner.py` initialization**:
```python
def _classify_list_inputs(self, graph: Graph) -> dict[str, set[str]]:
    """
    Identify properties that: 
    1. Have type list[T]
    2. Have multiple incoming edges on the same targetHandle
    
    Returns: {node_id: {handle_names_requiring_aggregation}}
    """
    multi_edge_list_inputs:  dict[str, set[str]] = defaultdict(set)
    
    # Group edges by target + targetHandle
    edges_by_target_handle:  dict[tuple[str, str], list[Edge]] = defaultdict(list)
    for edge in graph.edges:
        edges_by_target_handle[(edge.target, edge.targetHandle)].append(edge)
    
    # Check each target handle with multiple edges
    for (node_id, handle), edges in edges_by_target_handle.items():
        if len(edges) <= 1:
            continue
        
        node = graph.find_node(node_id)
        if not node:
            continue
        
        prop = node.find_property(handle)
        if not prop or not prop.type.is_list_type():
            # Multiple edges to non-list property:  validation error (existing behavior)
            continue
        
        multi_edge_list_inputs[node_id].add(handle)
    
    return multi_edge_list_inputs
```

#### 3.2.2 Actor-Level Aggregation

**Modified `NodeActor._run_non_streaming_internal` in `actor.py`**:

Current behavior uses `_gather_initial_inputs` to wait for one value per handle, then calls `process_node_with_inputs`.

**Proposed Behavior**:
```python
async def _gather_list_inputs(
    self,
    handles: set[str],
    list_handles: set[str]  # NEW: handles requiring aggregation
) -> dict[str, Any]:
    """
    Gather inputs, aggregating multi-source list handles. 
    
    For handles in `list_handles`:
    - Wait for EOS on ALL upstream sources
    - Collect ALL values into a list
    
    For other handles:
    - Existing behavior (first value or zip_all/on_any semantics)
    """
    result:  dict[str, Any] = {}
    
    for handle in handles:
        if handle in list_handles:
            # Aggregation mode: collect all values until EOS
            collected = []
            async for item in self.inbox.iter_input(handle):
                collected.append(item)
            result[handle] = collected
        else:
            # Existing behavior
            async for item in self. inbox.iter_input(handle):
                result[handle] = item
                break  # Take first value
    
    return result
```

**Edge Detection**:  Pass `list_handles = runner.multi_edge_list_inputs. get(node._id, set())` to actor initialization.

#### 3.2.3 Upstream Counting

The `NodeInbox. add_upstream(handle, count)` already tracks how many sources feed a handle. When initializing inboxes: 

```python
# In WorkflowRunner._initialize_inboxes
for edge in graph.edges:
    inbox = self.node_inboxes[edge.target]
    inbox.add_upstream(edge.targetHandle, count=1)
```

This ensures `iter_input(handle)` terminates only when **all** sources have called `mark_source_done(handle)`.

### 3.3 Behavior Matrix

| Upstream Streaming | Downstream Streaming | List Input Mode | Behavior |
|-------------------|---------------------|----------------|----------|
| **None** | **No** | **Multi-edge list** | Actor waits for EOS on all upstreams, collects all values into list, calls `process()` once |
| **All** | **No** | **Multi-edge list** | Same as above; streaming just means values arrive incrementally |
| **Mixed** | **No** | **Multi-edge list** | Same as above; some upstreams finish quickly, others stream |
| **None** | **Yes (gen_process)** | **Multi-edge list** | Same batching behavior; list passed to `gen_process` |
| **All** | **Yes (is_streaming_input)** | **N/A** | Node opts into manual inbox draining; auto-aggregation disabled (explicit opt-out) |

**Key Insight**: For non-streaming-input nodes, the actor **already waits** for all inputs before calling `process()`. We extend this to collect *all* values (not just first/latest) for list-typed handles.

### 3.4 Type Validation Enhancement

**In `Graph.validate_edge_types()` (`graph.py`)**:

Current validation checks single-edge type compatibility.  Extend to validate multi-edge scenarios:

```python
def validate_edge_types(self) -> list[str]:
    errors = []
    
    # Group edges by target handle
    edges_by_target_handle = defaultdict(list)
    for edge in self.edges:
        edges_by_target_handle[(edge.target, edge.targetHandle)].append(edge)
    
    for (target_id, handle), edges in edges_by_target_handle.items():
        target_node = self.find_node(target_id)
        if not target_node: 
            continue
        
        prop = target_node.find_property(handle)
        if not prop: 
            continue
        
        if len(edges) > 1:
            # Multiple edges to same property
            if not prop.type.is_list_type():
                errors.append(
                    f"{target_id}:{handle}:  Multiple edges target non-list property "
                    f"(type: {prop.type.type}). Either change property type to list "
                    f"or use a Collect node."
                )
                continue
            
            # Validate all source types are compatible with list element type
            element_type = prop.type.type_args[0] if prop.type.type_args else TypeMetadata(type="any")
            for edge in edges:
                source_node = self.find_node(edge. source)
                if not source_node:
                    continue
                source_type = self._get_output_type(source_node, edge.sourceHandle)
                if not typecheck(element_type, source_type):
                    errors.append(
                        f"{target_id}:{handle}: Edge from {edge.source}:{edge.sourceHandle} "
                        f"has incompatible type {source_type.type} for list element type {element_type.type}"
                    )
        else:
            # Single edge:  existing validation
            # ... existing code...
    
    return errors
```

### 3.5 UI/UX Implications

**Visual Feedback** (frontend changes, out of scope for this doc):
- Multiple edges converging on a list input should show a "merge" indicator
- Property tooltip:  "This list will collect values from N sources"

**Migration Path**:
- Existing graphs with `Collect` nodes remain valid
- Users can gradually refactor to direct multi-edge connections
- Detection of "redundant Collect nodes" could be added as a linter warning

## 4. Implementation Plan

### Phase 1: Foundation (1-2 weeks)

1. **Add `multi_edge_list_inputs` tracking** to `WorkflowRunner.__init__`
   - File: `src/nodetool/workflows/workflow_runner.py`
   - Method: `_classify_list_inputs(graph)`
   - Store in `self.multi_edge_list_inputs:  dict[str, set[str]]`

2. **Extend `NodeInbox` metadata** (optional, for debugging)
   - Add `_aggregation_mode:  dict[str, bool]` to track which handles auto-aggregate
   - Expose via `is_aggregating(handle:  str) -> bool`

3. **Update `NodeActor` initialization**
   - Pass `list_handles` from `runner.multi_edge_list_inputs` to actor
   - File: `src/nodetool/workflows/actor.py`

### Phase 2: Aggregation Logic (1-2 weeks)

4. **Implement `_gather_list_inputs` in `NodeActor`**
   - Replace `_gather_initial_inputs` with enhanced version
   - File: `src/nodetool/workflows/actor.py`
   - Method: `_gather_list_inputs(handles, list_handles)`

5. **Integrate into `_run_non_streaming_internal`**
   - Replace call to `_gather_initial_inputs`
   - Pass `list_handles` to new method

6. **Handle streaming-output nodes**
   - Extend `process_streaming_node_with_inputs` similarly
   - Ensure `gen_process` receives aggregated lists

### Phase 3: Validation (1 week)

7. **Enhance `Graph.validate_edge_types`**
   - File: `src/nodetool/workflows/graph.py`
   - Add multi-edge cardinality checks
   - Validate type compatibility for all source edges

8. **Add validation tests**
   - File: `tests/workflows/test_graph_validation.py`
   - Test cases: 
     - Multiple edges to list property (valid)
     - Multiple edges to non-list property (error)
     - Type mismatch in multi-edge list (error)
     - Single edge to list (unchanged)

### Phase 4: Integration Testing (1-2 weeks)

9. **Non-streaming multi-edge tests**
   - File: `tests/workflows/test_multi_edge_list_input.py`
   - Scenarios:
     - 3 non-streaming nodes → list input
     - Verify all values collected
     - Verify EOS semantics

10. **Streaming multi-edge tests**
    - Streaming producer → list input
    - Mixed streaming/non-streaming → list input
    - Verify order preservation (or document as undefined)

11. **Edge case tests**
    - Empty list (no values from any source)
    - One source errors, others succeed
    - Different upstream completion times

12. **Backward compatibility tests**
    - Existing single-edge-to-list graphs
    - Existing Collect node patterns (should still work)

### Phase 5: Documentation and Rollout (1 week)

13. **Update workflow README**
    - File: `src/nodetool/workflows/README.md`
    - Document multi-edge list behavior
    - Add diagrams for streaming/non-streaming cases

14. **Add migration guide**
    - How to refactor Collect nodes
    - When to still use Collect (e.g., custom aggregation logic)

15. **Update graph planner prompts** (if applicable)
    - File: `src/nodetool/agents/graph_planner.py`
    - Teach AI to use multi-edge connections for lists

## 5. Alternative Approaches Considered

### 5.1 Property-Level Annotation

**Idea**: Add `aggregation:  Literal["first", "last", "collect"] = "first"` to `Property` definition.

**Pros**: 
- Explicit control per property
- Could support different aggregation strategies (first, last, collect, custom)

**Cons**:
- Requires schema changes to all node definitions
- More complex mental model
- Doesn't solve type ambiguity (is `list[int]` expecting one list or many ints?)

**Decision**: Rejected. Implicit behavior based on type + cardinality is simpler. 

### 5.2 Special "Merge" Edge Type

**Idea**:  Introduce `Edge.merge_mode:  Literal["replace", "append"]` to explicitly mark aggregating edges.

**Pros**: 
- Clear visual distinction in UI
- Backward compatible (default = "replace")

**Cons**:
- Redundant with type information
- Adds complexity to edge model
- Harder to infer correct mode automatically

**Decision**: Rejected. Type system already encodes the intent.

### 5.3 Explicit MultiEdgeInput Node Type

**Idea**: Create a `MultiEdgeInput[T]` property type distinct from `list[T]`.

**Pros**:
- Unambiguous: `list[T]` always expects a single list value
- No behavior change for existing nodes

**Cons**:
- Breaks semantic clarity (users think in terms of "list of things")
- Requires updating many node definitions
- Doesn't align with Python's type system

**Decision**: Rejected.  Users expect `list[T]` to accept multiple items.

## 6. Open Questions and Risks

### 6.1 Ordering Guarantees

**Question**: When multiple edges feed a list, what order are elements in? 

**Options**:
1. **Arrival order**: As values hit the inbox (non-deterministic with parallelism)
2. **Source order**: Based on edge IDs or node IDs (deterministic but arbitrary)
3. **Undefined**:  Document as unordered; users must sort explicitly if needed

**Recommendation**: **Arrival order** (existing inbox behavior). Document that order is not guaranteed across sources.  If order matters, use a single source or explicit ordering nodes.

### 6.2 Performance Impact

**Concern**:  Collecting all values into memory before processing could cause OOM for large streams.

**Mitigation**:
- For streaming-input nodes (`is_streaming_input=True`), aggregation is opt-in via manual `iter_input` usage
- Non-streaming nodes already batch inputs; this just extends batch size
- Add optional `max_buffer_size` limit to `NodeInbox` (existing mechanism)

**Monitoring**: Add metrics for inbox buffer depths to detect pathological cases.

### 6.3 Partial Failure Handling

**Question**: If one upstream fails, do we: 
1. Wait for other upstreams to complete, deliver partial list?
2. Cancel the target node immediately?
3. Deliver empty list?

**Current Behavior**: Errors propagate immediately; node doesn't execute.

**Recommendation**: **Maintain current behavior**. If any upstream errors, the node doesn't run. This aligns with existing error handling and avoids silent data loss.

### 6.4 Dynamic Properties and List Inputs

**Concern**: Dynamic nodes (`is_dynamic=True`) allow arbitrary properties at runtime.  How to handle multi-edge list inputs?

**Solution**: Dynamic properties are stored in `_dynamic_properties` dict. Apply same aggregation logic if property name matches a handle with multiple edges.  Type validation relies on edge source types only (no declared property type).

### 6.5 Backward Compatibility with Collect Nodes

**Risk**: Users may have graphs where `Collect` node is essential (e.g., using `Collect.separator` or custom logic).

**Mitigation**:
- `Collect` nodes remain fully functional
- No automatic removal or deprecation
- Document as "optional optimization" rather than replacement

## 7. Success Metrics

- **Code Coverage**: >90% for new aggregation logic
- **Performance**: No >10% regression in workflow execution time (existing graphs)
- **User Adoption**: Track usage of multi-edge-to-list patterns in new graphs (telemetry)
- **Error Reduction**: Decrease in "missing input" errors for list properties

## 8. Future Enhancements

### 8.1 Custom Aggregation Functions

Allow nodes to specify `list_aggregation:  Literal["concat", "zip", "custom"]`:
- `concat`: Default behavior (flat list)
- `zip`: Creates list of tuples (one per source)
- `custom`: Calls `node.aggregate_list_input(handle, values)`

### 8.2 Streaming List Aggregation

For streaming-input nodes, provide `iter_aggregated_lists()` that yields lists as upstreams complete, rather than waiting for all EOS.

### 8.3 UI Enhancements

- Visual "merge point" indicator on list input handles
- Drag-and-drop multiple edges at once
- Auto-suggest removing redundant Collect nodes

## 9. Conclusion

This design enables intuitive multi-edge-to-list connections while preserving backward compatibility and respecting the existing actor-based execution model. The implementation leverages existing inbox EOS tracking and type validation infrastructure, minimizing risk.  The phased rollout allows for iterative testing and refinement.

**Recommendation**: Proceed with Phase 1-2 implementation, validate with early users, then complete Phases 3-5 based on feedback.

---

**Document Version**: 1.0  
**Author**: Technical Design Team  
**Date**: 2026-01-07  
**Status**:  Proposed
