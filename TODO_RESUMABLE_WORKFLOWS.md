# Resumable Workflows - TODO and Known Issues

## Critical Issues (Must Fix Before Production)

### 1. Attempt Tracking
**Location**: `src/nodetool/workflows/actor.py` lines 668, 695, 731
**Issue**: NodeActor hardcodes `attempt=1` for all events
**Impact**: Retries won't be tracked correctly, projection will be wrong
**Fix Required**:
```python
# Before scheduling, read from projection:
projection = await self.runner.event_logger.get_projection()
node_state = projection.get_node_state(self.node._id)
current_attempt = node_state.get("attempt", 0) + 1 if node_state else 1

# Use current_attempt in events:
await self.runner.event_logger.log_node_scheduled(
    node_id=node._id,
    node_type=node.get_node_type(),
    attempt=current_attempt,
)
```

### 2. Retryability Detection
**Location**: `src/nodetool/workflows/actor.py` line 733
**Issue**: NodeFailed event hardcodes `retryable=False`
**Impact**: No failed nodes will be retried during recovery
**Fix Required**:
```python
# Determine if error is retryable
retryable = self._is_retryable_error(e)

await self.runner.event_logger.log_node_failed(
    node_id=node._id,
    attempt=current_attempt,
    error=str(e)[:1000],
    retryable=retryable,
)

def _is_retryable_error(self, error: Exception) -> bool:
    """Determine if an error is retryable."""
    # GPU OOM errors are retryable
    if self.runner._torch_support.is_cuda_oom_exception(error):
        return True
    # Timeout errors are retryable
    if isinstance(error, asyncio.TimeoutError):
        return True
    # Network errors are retryable
    if isinstance(error, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    # Default: not retryable
    return False
```

### 3. Output Tracking with Large Object Handling
**Location**: `src/nodetool/workflows/actor.py` line 696
**Issue**: NodeCompleted event has empty `outputs={}` 
**Impact**: Recovery can't verify node outputs, may cause duplicate work
**Strategy**: Store large objects separately to avoid bloating event log
- AssetRef types (ImageRef, VideoRef, AudioRef, etc.) → use temp storage for in-flight outputs
- Large objects (>1MB) → store in temp storage, reference by ID
- Small objects (<1MB) → serialize directly in event payload
- **Streaming nodes** → compress all output chunks into single log entry at completion

**Important for Streaming**:
- Streaming nodes emit thousands of small chunks (e.g., video frames, audio samples)
- Logging each chunk creates database write contention
- Solution: Log only at node completion, compress all chunks into one entry
- Store compressed chunks in temp storage if needed
- Example: 1000 image chunks → stored as single compressed array reference

**Fix Required**:
```python
# Capture outputs from send_messages
result = {}
is_streaming = node.__class__.is_streaming_output()

if is_streaming:
    # For streaming nodes, compress all chunks into one entry
    result = self._compress_streaming_outputs(outputs_collector.collected())
else:
    # For regular nodes, serialize each output
    for key, value in outputs_collector.collected().items():
        result[key] = self._serialize_output_for_event_log(value)

await self.runner.event_logger.log_node_completed(
    node_id=node._id,
    attempt=current_attempt,
    outputs=result,
    duration_ms=duration_ms,
)

def _serialize_output_for_event_log(self, value: Any, max_size_bytes: int = 1_000_000) -> dict:
    """Serialize output for event log, storing large objects separately.
    
    Returns dict with either:
    - {'type': 'asset_ref', 'uri': 'temp://...', 'asset_id': '...'} for AssetRef types
    - {'type': 'inline', 'value': {...}} for small objects
    - {'type': 'external_ref', 'storage_id': '...'} for large objects stored separately
    """
    # Handle AssetRef types - use temp storage for in-flight outputs
    if isinstance(value, AssetRef):
        # If using memory URI, migrate to temp storage for durability
        if value.uri.startswith('memory://'):
            temp_uri, temp_id = await self._migrate_to_temp_storage(value)
            return {
                'type': 'asset_ref',
                'asset_type': value.__class__.__name__,
                'uri': temp_uri,
                'asset_id': temp_id,
            }
        return {
            'type': 'asset_ref',
            'asset_type': value.__class__.__name__,
            'uri': value.uri,
            'asset_id': value.asset_id,
        }
    
    # Try to serialize inline
    try:
        serialized = json.dumps(value)
        if len(serialized) <= max_size_bytes:
            return {'type': 'inline', 'value': value}
    except (TypeError, ValueError):
        pass
    
    # Too large or not JSON-serializable - store in temp storage
    storage_id = await self._store_large_output_to_temp(value)
    return {'type': 'external_ref', 'storage_id': storage_id}

def _compress_streaming_outputs(self, outputs: dict) -> dict:
    """Compress streaming outputs into single log entry.
    
    For nodes that emit many chunks (e.g., 1000 video frames), this compresses
    all chunks into a single entry to avoid write contention on the database.
    
    Returns:
    - {'type': 'streaming_compressed', 'chunk_count': N, 'storage_id': '...'}
    """
    chunk_count = sum(len(v) if isinstance(v, list) else 1 for v in outputs.values())
    
    # Store compressed chunks in temp storage
    storage_id = await self._store_compressed_chunks_to_temp(outputs)
    
    return {
        'type': 'streaming_compressed',
        'chunk_count': chunk_count,
        'storage_id': storage_id,
        'size_bytes': len(json.dumps(outputs).encode('utf-8')),
    }
```

## High Priority Issues

### 4. Resume API Endpoint
**Status**: Not implemented
**Required**: Yes
**Description**: Add API endpoint to manually resume workflows
**Location**: Create `src/nodetool/api/workflow_resume.py`
**Implementation**:
```python
@router.post("/workflows/{run_id}/resume")
async def resume_workflow(
    run_id: str,
    user_id: str = Depends(get_current_user),
):
    recovery = WorkflowRecoveryService()
    success, message = await recovery.resume_workflow(
        run_id=run_id,
        graph=...,  # Load from database
        context=...,  # Create new context
    )
    return {"success": success, "message": message}
```

### 5. Message Deduplication
**Status**: Events defined but not used
**Required**: Yes
**Description**: Track OutboxEnqueued/Sent to prevent duplicate processing
**Location**: `src/nodetool/workflows/workflow_runner.py` in `send_messages()`
**Implementation**:
```python
async def send_messages(self, node, result, context):
    for key, value in result.items():
        for edge in context.graph.find_edges(node.id, key):
            message_id = f"{node.id}-{edge.id}-{hash(value)}"
            
            # Log outbox enqueue
            if self.event_logger:
                await self.event_logger.log_outbox_enqueued(
                    node_id=node.id,
                    edge_id=edge.id,
                    message_id=message_id,
                    data={"value": value},
                )
            
            # Deliver message
            inbox = self.node_inboxes.get(edge.target)
            if inbox is not None:
                await inbox.put(edge.targetHandle, value)
                
                # Log outbox sent
                if self.event_logger:
                    await self.event_logger.log_outbox_sent(
                        node_id=node.id,
                        edge_id=edge.id,
                        message_id=message_id,
                    )
```

## Medium Priority Issues

### 6. Trigger Cursor Persistence
**Status**: Events defined but nodes don't implement
**Required**: For trigger nodes only
**Description**: Save and restore trigger consumption cursors
**Location**: Trigger node implementations
**Implementation**: Add `resume_from_cursor()` method to trigger nodes

### 7. Checkpoint Support
**Status**: Event defined but not used
**Required**: For long-running nodes
**Description**: Allow nodes to checkpoint progress
**Location**: Long-running node implementations
**Implementation**: Add `checkpoint()` method that nodes can call

### 8. Multi-Worker Testing
**Status**: Not tested
**Required**: For production deployment
**Description**: Test concurrent recovery attempts, lease expiry, heartbeat
**Location**: Create `tests/workflows/test_distributed_recovery.py`

## Low Priority Issues

### 9. Event Log Compaction
**Status**: Not implemented
**Required**: For long-term operation
**Description**: Archive old events, keep only recent ones
**Implementation**: Periodic background job to snapshot and archive

### 10. Batch Event Appending
**Status**: Not implemented
**Required**: Performance optimization
**Description**: Batch multiple events in single transaction
**Implementation**: Buffer events in memory, flush periodically

### 11. Distributed Tracing
**Status**: Not implemented
**Required**: Observability enhancement
**Description**: Add trace IDs to events, integrate with OpenTelemetry
**Implementation**: Add trace context to event payloads

## Code Quality Issues

### 12. Error Handling in Event Logging
**Location**: Throughout `workflow_runner.py` and `actor.py`
**Issue**: Exceptions in event logging are caught and logged as warnings
**Impact**: Events may be silently lost
**Fix**: Consider making event logging failures fatal, or add retry logic

### 13. Projection Cache Invalidation
**Location**: `event_logger.py`
**Issue**: In-memory projection cache is never invalidated
**Impact**: Stale state if projection is updated externally
**Fix**: Add cache invalidation logic or TTL

### 14. Missing Type Hints
**Location**: Some methods in recovery service
**Issue**: Not all parameters have type hints
**Fix**: Add complete type annotations

## Documentation Improvements

### 15. API Documentation
**Status**: Missing
**Required**: Yes
**Description**: Document public APIs for event log queries
**Location**: Add to existing API docs

### 16. Recovery Playbook
**Status**: Missing
**Required**: For operations team
**Description**: Step-by-step guide for manual recovery
**Location**: Create `docs/recovery_playbook.md`

### 17. Performance Tuning Guide
**Status**: Missing
**Required**: For optimization
**Description**: Guidelines for tuning event log performance
**Location**: Add to design doc or create separate guide

## Testing Gaps

### 18. End-to-End Crash Tests
**Status**: Unit tests only
**Required**: Yes
**Description**: Simulate actual crashes and verify recovery
**Location**: Create `tests/workflows/test_crash_recovery.py`

### 19. Load Testing
**Status**: Not done
**Required**: For production
**Description**: Test event log performance under load
**Implementation**: Generate high-volume workflows, measure overhead

### 20. Chaos Engineering
**Status**: Not done
**Required**: For reliability
**Description**: Random failure injection to test recovery
**Implementation**: Use chaos testing framework

## Priority Matrix

**P0 (Block Production)**:
1. Attempt Tracking (#1)
2. Retryability Detection (#2)
3. Output Tracking (#3)

**P1 (Required for Launch)**:
4. Resume API Endpoint (#4)
5. Message Deduplication (#5)
18. End-to-End Crash Tests (#18)

**P2 (Post-Launch)**:
6. Trigger Cursor Persistence (#6)
7. Checkpoint Support (#7)
8. Multi-Worker Testing (#8)
12. Error Handling (#12)

**P3 (Future Enhancements)**:
9. Event Log Compaction (#9)
10. Batch Event Appending (#10)
11. Distributed Tracing (#11)
13-20. Other improvements

## Estimated Effort

**P0 Issues**: 1-2 days
**P1 Issues**: 2-3 days
**P2 Issues**: 3-5 days
**P3 Issues**: 5+ days

**Total to Production**: ~5-8 days of development work
