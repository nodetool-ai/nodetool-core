"""
Tests for state table operations (Phase 7 - Testing).

Tests the new mutable state tables that serve as source of truth:
- RunState
- RunNodeState
- RunInboxMessage
- TriggerInput
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from nodetool.models.run_state import RunState
from nodetool.models.run_node_state import RunNodeState
from nodetool.models.run_inbox_message import RunInboxMessage
from nodetool.models.trigger_input import TriggerInput


class TestRunState:
    """Test RunState model operations."""
    
    @pytest.mark.asyncio
    async def test_create_run(self):
        """Test creating a new run state."""
        run_id = f"test-run-{datetime.now().timestamp()}"
        run_state = await RunState.create_run(
            run_id=run_id,
            graph_json={"nodes": []},
            params_json={"param1": "value1"}
        )
        
        assert run_state.run_id == run_id
        assert run_state.status == "running"
        assert run_state.version == 1
        
    @pytest.mark.asyncio
    async def test_mark_completed(self):
        """Test marking a run as completed."""
        run_id = f"test-run-{datetime.now().timestamp()}"
        run_state = await RunState.create_run(run_id=run_id)
        
        await run_state.mark_completed(outputs_json={"result": 42})
        
        assert run_state.status == "completed"
        assert run_state.outputs_json == {"result": 42}
        assert run_state.completed_at is not None
        
    @pytest.mark.asyncio
    async def test_mark_failed(self):
        """Test marking a run as failed."""
        run_id = f"test-run-{datetime.now().timestamp()}"
        run_state = await RunState.create_run(run_id=run_id)
        
        await run_state.mark_failed(
            error="Something went wrong",
            node_id="node-1"
        )
        
        assert run_state.status == "failed"
        assert run_state.failed_node_id == "node-1"
        assert "Something went wrong" in run_state.error
        
    @pytest.mark.asyncio
    async def test_mark_suspended(self):
        """Test marking a run as suspended."""
        run_id = f"test-run-{datetime.now().timestamp()}"
        run_state = await RunState.create_run(run_id=run_id)
        
        await run_state.mark_suspended(
            node_id="approval-node",
            reason="Waiting for approval",
            state_json={"request_id": "req-123"},
            metadata_json={"approver": "admin"}
        )
        
        assert run_state.status == "suspended"
        assert run_state.suspended_node_id == "approval-node"
        assert run_state.suspension_reason == "Waiting for approval"
        assert run_state.suspension_state_json == {"request_id": "req-123"}
        
    @pytest.mark.asyncio
    async def test_optimistic_locking(self):
        """Test optimistic locking via version field."""
        run_id = f"test-run-{datetime.now().timestamp()}"
        run_state = await RunState.create_run(run_id=run_id)
        
        original_version = run_state.version
        await run_state.mark_completed()
        
        assert run_state.version > original_version


class TestRunNodeState:
    """Test RunNodeState model operations."""
    
    @pytest.mark.asyncio
    async def test_create_node_state(self):
        """Test creating a new node state."""
        run_id = f"test-run-{datetime.now().timestamp()}"
        node_id = "node-1"
        
        node_state = await RunNodeState.get_or_create(
            run_id=run_id,
            node_id=node_id
        )
        
        assert node_state.run_id == run_id
        assert node_state.node_id == node_id
        assert node_state.status == "idle"
        assert node_state.attempt == 0
        
    @pytest.mark.asyncio
    async def test_mark_scheduled(self):
        """Test marking a node as scheduled."""
        run_id = f"test-run-{datetime.now().timestamp()}"
        node_state = await RunNodeState.get_or_create(run_id, "node-1")
        
        await node_state.mark_scheduled(attempt=1)
        
        assert node_state.status == "scheduled"
        assert node_state.attempt == 1
        assert node_state.scheduled_at is not None
        
    @pytest.mark.asyncio
    async def test_mark_running(self):
        """Test marking a node as running."""
        run_id = f"test-run-{datetime.now().timestamp()}"
        node_state = await RunNodeState.get_or_create(run_id, "node-1")
        await node_state.mark_scheduled(attempt=1)
        
        await node_state.mark_running()
        
        assert node_state.status == "running"
        assert node_state.started_at is not None
        
    @pytest.mark.asyncio
    async def test_mark_completed(self):
        """Test marking a node as completed."""
        run_id = f"test-run-{datetime.now().timestamp()}"
        node_state = await RunNodeState.get_or_create(run_id, "node-1")
        await node_state.mark_scheduled(attempt=1)
        await node_state.mark_running()
        
        await node_state.mark_completed(
            outputs_json={"result": 42},
            duration_ms=100
        )
        
        assert node_state.status == "completed"
        assert node_state.outputs_json == {"result": 42}
        assert node_state.completed_at is not None
        
    @pytest.mark.asyncio
    async def test_mark_failed(self):
        """Test marking a node as failed."""
        run_id = f"test-run-{datetime.now().timestamp()}"
        node_state = await RunNodeState.get_or_create(run_id, "node-1")
        await node_state.mark_scheduled(attempt=1)
        await node_state.mark_running()
        
        await node_state.mark_failed(
            error="Node execution failed",
            retryable=True
        )
        
        assert node_state.status == "failed"
        assert node_state.last_error == "Node execution failed"
        assert node_state.retryable == True
        
    @pytest.mark.asyncio
    async def test_get_incomplete_nodes(self):
        """Test finding incomplete nodes."""
        run_id = f"test-run-{datetime.now().timestamp()}"
        
        # Create some nodes in different states
        node1 = await RunNodeState.get_or_create(run_id, "node-1")
        await node1.mark_scheduled(attempt=1)
        
        node2 = await RunNodeState.get_or_create(run_id, "node-2")
        await node2.mark_scheduled(attempt=1)
        await node2.mark_running()
        
        node3 = await RunNodeState.get_or_create(run_id, "node-3")
        await node3.mark_scheduled(attempt=1)
        await node3.mark_running()
        await node3.mark_completed()
        
        incomplete = await RunNodeState.get_incomplete_nodes(run_id)
        
        assert len(incomplete) == 2
        assert any(n.node_id == "node-1" for n in incomplete)
        assert any(n.node_id == "node-2" for n in incomplete)


class TestRunInboxMessage:
    """Test RunInboxMessage model operations."""
    
    @pytest.mark.asyncio
    async def test_append_message(self):
        """Test appending a message to inbox."""
        run_id = f"test-run-{datetime.now().timestamp()}"
        
        message = await RunInboxMessage.append_message(
            run_id=run_id,
            node_id="node-1",
            handle="input",
            message_id="msg-123",
            payload_json={"data": 42}
        )
        
        assert message.run_id == run_id
        assert message.node_id == "node-1"
        assert message.handle == "input"
        assert message.message_id == "msg-123"
        assert message.status == "pending"
        assert message.msg_seq > 0
        
    @pytest.mark.asyncio
    async def test_idempotent_append(self):
        """Test that duplicate message_id is ignored."""
        run_id = f"test-run-{datetime.now().timestamp()}"
        message_id = f"msg-{datetime.now().timestamp()}"
        
        # First append
        msg1 = await RunInboxMessage.append_message(
            run_id=run_id,
            node_id="node-1",
            handle="input",
            message_id=message_id,
            payload_json={"data": 1}
        )
        
        # Second append with same message_id (should be ignored)
        msg2 = await RunInboxMessage.append_message(
            run_id=run_id,
            node_id="node-1",
            handle="input",
            message_id=message_id,
            payload_json={"data": 2}
        )
        
        # Should return the same message
        assert msg1.id == msg2.id
        assert msg1.payload_json == {"data": 1}  # Original payload
        
    @pytest.mark.asyncio
    async def test_get_pending_messages(self):
        """Test getting pending messages."""
        run_id = f"test-run-{datetime.now().timestamp()}"
        
        # Append multiple messages
        await RunInboxMessage.append_message(run_id, "node-1", "input", "msg-1", {"n": 1})
        await RunInboxMessage.append_message(run_id, "node-1", "input", "msg-2", {"n": 2})
        await RunInboxMessage.append_message(run_id, "node-1", "input", "msg-3", {"n": 3})
        
        pending = await RunInboxMessage.get_pending_messages(run_id, "node-1", "input")
        
        assert len(pending) >= 3
        # Check ordering by seq
        seqs = [m.msg_seq for m in pending]
        assert seqs == sorted(seqs)
        
    @pytest.mark.asyncio
    async def test_mark_consumed(self):
        """Test marking a message as consumed."""
        run_id = f"test-run-{datetime.now().timestamp()}"
        
        message = await RunInboxMessage.append_message(
            run_id, "node-1", "input", "msg-123", {"data": 42}
        )
        
        await message.mark_consumed()
        
        assert message.status == "consumed"
        
        # Should not appear in pending
        pending = await RunInboxMessage.get_pending_messages(run_id, "node-1", "input")
        assert not any(m.message_id == "msg-123" for m in pending)


class TestTriggerInput:
    """Test TriggerInput model operations."""
    
    @pytest.mark.asyncio
    async def test_add_trigger_input(self):
        """Test adding a trigger input."""
        run_id = f"test-run-{datetime.now().timestamp()}"
        
        trigger = await TriggerInput.add_trigger_input(
            run_id=run_id,
            node_id="trigger-1",
            input_id="webhook-123",
            payload_json={"event": "user_action"}
        )
        
        assert trigger.run_id == run_id
        assert trigger.node_id == "trigger-1"
        assert trigger.input_id == "webhook-123"
        assert trigger.processed == False
        
    @pytest.mark.asyncio
    async def test_idempotent_trigger_input(self):
        """Test that duplicate input_id is ignored."""
        run_id = f"test-run-{datetime.now().timestamp()}"
        input_id = f"webhook-{datetime.now().timestamp()}"
        
        # First add
        t1 = await TriggerInput.add_trigger_input(
            run_id, "trigger-1", input_id, {"n": 1}
        )
        
        # Second add with same input_id (should be ignored)
        t2 = await TriggerInput.add_trigger_input(
            run_id, "trigger-1", input_id, {"n": 2}
        )
        
        # Should return the same trigger
        assert t1.id == t2.id
        assert t1.payload_json == {"n": 1}  # Original payload
        
    @pytest.mark.asyncio
    async def test_mark_processed(self):
        """Test marking a trigger input as processed."""
        run_id = f"test-run-{datetime.now().timestamp()}"
        
        trigger = await TriggerInput.add_trigger_input(
            run_id, "trigger-1", "webhook-123", {"event": "test"}
        )
        
        await trigger.mark_processed()
        
        assert trigger.processed == True
        
    @pytest.mark.asyncio
    async def test_get_pending_inputs(self):
        """Test getting pending trigger inputs."""
        run_id = f"test-run-{datetime.now().timestamp()}"
        
        # Add multiple trigger inputs
        await TriggerInput.add_trigger_input(run_id, "trigger-1", "input-1", {"n": 1})
        await TriggerInput.add_trigger_input(run_id, "trigger-1", "input-2", {"n": 2})
        await TriggerInput.add_trigger_input(run_id, "trigger-1", "input-3", {"n": 3})
        
        pending = await TriggerInput.get_pending_inputs(run_id, "trigger-1")
        
        assert len(pending) >= 3
        assert all(not t.processed for t in pending)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
