"""
Comprehensive tests for stop command functionality in ChatWebSocketRunner.

This test suite covers all backend stop command scenarios as documented in
STOP_COMMAND_SCENARIOS.md, ensuring proper stop event handling, completion
signals, and state management across all chat processing modes.
"""

import pytest
import asyncio
import json
import msgpack
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncGenerator, Any, Dict, List

from nodetool.common.chat_websocket_runner import ChatWebSocketRunner
from nodetool.metadata.types import Message, MessageTextContent, ToolCall, Provider
from nodetool.workflows.types import Chunk, ToolCallUpdate, TaskUpdate, PlanningUpdate, OutputUpdate
from nodetool.agents.tools.base import Tool
from nodetool.workflows.processing_context import ProcessingContext


class MockWebSocket:
    """Mock WebSocket for testing"""
    def __init__(self):
        self.accepted = False
        self.closed = False
        self.sent_messages = []
        self.received_messages = []
        
    async def accept(self):
        self.accepted = True
        
    async def close(self, code=None, reason=None):
        self.closed = True
        self.close_code = code
        self.close_reason = reason
        
    async def send_bytes(self, data):
        self.sent_messages.append({"type": "bytes", "data": data})
        
    async def send_text(self, text):
        self.sent_messages.append({"type": "text", "data": text})
        
    async def receive(self):
        if self.received_messages:
            return self.received_messages.pop(0)
        # Simulate disconnect
        return {"type": "websocket.disconnect"}
        
    def queue_message(self, message):
        """Queue a message to be received"""
        self.received_messages.append(message)


class MockTool(Tool):
    """Mock tool for testing"""
    def __init__(self, name="mock_tool", delay=0.1):
        self.name = name
        self.delay = delay
        self.call_count = 0
        
    async def process(self, context, args):
        self.call_count += 1
        await asyncio.sleep(self.delay)
        return {"result": f"mock_result_{self.call_count}", "args": args}
        
    def user_message(self, args):
        return f"Running {self.name} with {args}"


class MockProvider:
    """Mock chat provider for testing"""
    def __init__(self, chunks=None, tools=None):
        self.chunks = chunks or [Chunk(content="Mock response", done=True)]
        self.tool_calls = tools or []
        self.call_count = 0
        
    async def generate_messages(self, messages, model, tools=None):
        self.call_count += 1
        for chunk in self.chunks:
            yield chunk
        for tool_call in self.tool_calls:
            yield tool_call


@pytest.fixture
def mock_websocket():
    return MockWebSocket()


@pytest.fixture
def chat_runner():
    runner = ChatWebSocketRunner(auth_token="test_token")
    runner.user_id = "test_user"
    runner.all_tools = [MockTool("test_tool")]
    return runner


@pytest.fixture
def test_message():
    return Message(
        role="user",
        content="Test message",
        provider=Provider.OpenAI,
        model="gpt-3.5-turbo"
    )


class TestStopCommandInfrastructure:
    """Test the basic stop command infrastructure"""
    
    def test_stop_event_initialization(self, chat_runner):
        """Test that stop event is properly initialized"""
        assert chat_runner.stop_event is not None
        assert not chat_runner.stop_event.is_set()
        assert not chat_runner.is_generating
        
    def test_generation_state_management(self, chat_runner):
        """Test generation state flag management"""
        assert not chat_runner.is_generating
        
        chat_runner.is_generating = True
        assert chat_runner.is_generating
        
        chat_runner.is_generating = False
        assert not chat_runner.is_generating
        
    async def test_disconnect_resets_state(self, chat_runner, mock_websocket):
        """Test that disconnect properly resets generation state"""
        chat_runner.websocket = mock_websocket
        chat_runner.is_generating = True
        chat_runner.stop_event.set()
        
        await chat_runner.disconnect()
        
        assert not chat_runner.is_generating
        assert mock_websocket.closed


class TestStopCommandHandling:
    """Test stop command message handling"""
    
    async def test_stop_command_when_generating(self, chat_runner, mock_websocket):
        """Test stop command received during generation"""
        chat_runner.websocket = mock_websocket
        chat_runner.is_generating = True
        
        # Mock the send_message method to capture output
        sent_messages = []
        async def mock_send_message(msg):
            sent_messages.append(msg)
        chat_runner.send_message = mock_send_message
        
        # Simulate stop command
        stop_message = {"type": "stop"}
        mock_websocket.queue_message({"text": json.dumps(stop_message)})
        
        # Process one iteration of the run loop
        with patch.object(chat_runner, 'connect'):
            try:
                await chat_runner.run(mock_websocket)
            except:
                pass  # Expected when receive() returns disconnect
        
        # Verify stop was handled
        assert any(msg.get("type") == "generation_stopped" for msg in sent_messages)
        
    async def test_stop_command_when_not_generating(self, chat_runner, mock_websocket):
        """Test stop command received when not generating"""
        chat_runner.websocket = mock_websocket
        chat_runner.is_generating = False
        
        # Mock the send_message method to capture output
        sent_messages = []
        async def mock_send_message(msg):
            sent_messages.append(msg)
        chat_runner.send_message = mock_send_message
        
        # Simulate stop command
        stop_message = {"type": "stop"}
        mock_websocket.queue_message({"text": json.dumps(stop_message)})
        
        # Process one iteration of the run loop
        with patch.object(chat_runner, 'connect'):
            try:
                await chat_runner.run(mock_websocket)
            except:
                pass  # Expected when receive() returns disconnect
        
        # Verify error was sent
        assert any(msg.get("type") == "error" for msg in sent_messages)


class TestRegularChatMessages:
    """Test stop command handling in regular chat messages"""
    
    async def test_regular_chat_completion_signal(self, chat_runner, test_message):
        """Test that regular chat sends completion signal"""
        chat_runner.chat_history = [test_message]
        
        # Mock provider
        mock_provider = MockProvider([
            Chunk(content="Hello", done=False),
            Chunk(content=" world", done=False),
        ])
        
        sent_messages = []
        async def mock_send_message(msg):
            sent_messages.append(msg)
        chat_runner.send_message = mock_send_message
        
        with patch('nodetool.common.chat_websocket_runner.get_provider', return_value=mock_provider):
            result = await chat_runner.process_messages()
            
        # Verify completion signal was sent
        completion_signals = [msg for msg in sent_messages if msg.get("type") == "chunk" and msg.get("done") is True]
        assert len(completion_signals) >= 1
        assert result.role == "assistant"
        
    async def test_regular_chat_stop_during_generation(self, chat_runner, test_message):
        """Test stopping regular chat during generation"""
        chat_runner.chat_history = [test_message]
        
        # Mock provider with delayed chunks
        async def mock_generate_messages(messages, model, tools=None):
            yield Chunk(content="Starting", done=False)
            await asyncio.sleep(0.1)
            # Check if stop was requested
            if chat_runner.stop_event.is_set():
                return
            yield Chunk(content=" continuing", done=False)
            
        mock_provider = MagicMock()
        mock_provider.generate_messages = mock_generate_messages
        
        sent_messages = []
        async def mock_send_message(msg):
            sent_messages.append(msg)
        chat_runner.send_message = mock_send_message
        
        # Start generation and stop it
        async def stop_after_delay():
            await asyncio.sleep(0.05)
            chat_runner.stop_event.set()
            
        with patch('nodetool.common.chat_websocket_runner.get_provider', return_value=mock_provider):
            await asyncio.gather(
                chat_runner.process_messages(),
                stop_after_delay()
            )
            
        # Verify stop was handled
        chunk_messages = [msg for msg in sent_messages if msg.get("type") == "chunk"]
        assert len(chunk_messages) >= 1
        # Should have completion signal
        assert any(msg.get("done") is True for msg in chunk_messages)
        
    async def test_regular_chat_with_tool_calls(self, chat_runner, test_message):
        """Test regular chat with tool calls can be stopped"""
        test_message.tools = ["test_tool"]
        chat_runner.chat_history = [test_message]
        
        # Mock provider that returns tool call
        tool_call = ToolCall(
            id="test_call",
            name="test_tool",
            args={"param": "value"}
        )
        mock_provider = MockProvider(chunks=[], tools=[tool_call])
        
        sent_messages = []
        async def mock_send_message(msg):
            sent_messages.append(msg)
        chat_runner.send_message = mock_send_message
        
        with patch('nodetool.common.chat_websocket_runner.get_provider', return_value=mock_provider):
            result = await chat_runner.process_messages()
            
        # Verify tool was called and completion signal sent
        tool_messages = [msg for msg in sent_messages if msg.get("type") == "tool_call_update"]
        assert len(tool_messages) >= 1
        completion_signals = [msg for msg in sent_messages if msg.get("type") == "chunk" and msg.get("done") is True]
        assert len(completion_signals) >= 1


class TestAgentMode:
    """Test stop command handling in agent mode"""
    
    async def test_agent_mode_completion_signal(self, chat_runner, test_message):
        """Test that agent mode sends completion signal"""
        test_message.agent_mode = True
        chat_runner.chat_history = [test_message]
        
        sent_messages = []
        async def mock_send_message(msg):
            sent_messages.append(msg)
        chat_runner.send_message = mock_send_message
        
        # Mock agent execution
        async def mock_agent_execute(context):
            yield Chunk(content="Agent response", done=False)
            
        with patch('nodetool.common.chat_websocket_runner.get_provider'), \
             patch('nodetool.common.chat_websocket_runner.Agent') as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.execute = mock_agent_execute
            mock_agent_class.return_value = mock_agent
            
            result = await chat_runner.process_agent_messages()
            
        # Verify completion signal was sent
        completion_signals = [msg for msg in sent_messages if msg.get("type") == "chunk" and msg.get("done") is True]
        assert len(completion_signals) >= 1
        assert result.role == "assistant"
        
    async def test_agent_mode_stop_during_execution(self, chat_runner, test_message):
        """Test stopping agent mode during execution"""
        test_message.agent_mode = True
        chat_runner.chat_history = [test_message]
        
        sent_messages = []
        async def mock_send_message(msg):
            sent_messages.append(msg)
        chat_runner.send_message = mock_send_message
        
        # Mock agent execution with stop check
        async def mock_agent_execute(context):
            yield Chunk(content="Starting", done=False)
            await asyncio.sleep(0.1)
            if chat_runner.stop_event.is_set():
                return
            yield Chunk(content=" continuing", done=False)
            
        # Start execution and stop it
        async def stop_after_delay():
            await asyncio.sleep(0.05)
            chat_runner.stop_event.set()
            
        with patch('nodetool.common.chat_websocket_runner.get_provider'), \
             patch('nodetool.common.chat_websocket_runner.Agent') as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.execute = mock_agent_execute
            mock_agent_class.return_value = mock_agent
            
            await asyncio.gather(
                chat_runner.process_agent_messages(),
                stop_after_delay()
            )
            
        # Verify stop was handled
        chunk_messages = [msg for msg in sent_messages if msg.get("type") == "chunk"]
        assert len(chunk_messages) >= 1
        # Should have completion signal
        assert any(msg.get("done") is True for msg in chunk_messages)


class TestHelpMode:
    """Test stop command handling in help mode"""
    
    async def test_help_mode_completion_signal(self, chat_runner, test_message):
        """Test that help mode sends completion signal"""
        test_message.help_mode = True
        chat_runner.chat_history = [test_message]
        
        sent_messages = []
        async def mock_send_message(msg):
            sent_messages.append(msg)
        chat_runner.send_message = mock_send_message
        
        # Mock provider for help mode
        mock_provider = MockProvider([
            Chunk(content="Help response", done=False),
        ])
        
        with patch('nodetool.common.chat_websocket_runner.get_provider', return_value=mock_provider):
            result = await chat_runner._process_help_messages(test_message)
            
        # Verify completion signal was sent
        completion_signals = [msg for msg in sent_messages if msg.get("type") == "chunk" and msg.get("done") is True]
        assert len(completion_signals) >= 1
        assert result.role == "assistant"
        
    async def test_help_mode_stop_during_generation(self, chat_runner, test_message):
        """Test stopping help mode during generation"""
        test_message.help_mode = True
        chat_runner.chat_history = [test_message]
        
        sent_messages = []
        async def mock_send_message(msg):
            sent_messages.append(msg)
        chat_runner.send_message = mock_send_message
        
        # Mock provider with delayed chunks
        async def mock_generate_messages(messages, model, tools=None):
            yield Chunk(content="Starting help", done=False)
            await asyncio.sleep(0.1)
            if chat_runner.stop_event.is_set():
                return
            yield Chunk(content=" continuing help", done=False)
            
        mock_provider = MagicMock()
        mock_provider.generate_messages = mock_generate_messages
        
        # Start generation and stop it
        async def stop_after_delay():
            await asyncio.sleep(0.05)
            chat_runner.stop_event.set()
            
        with patch('nodetool.common.chat_websocket_runner.get_provider', return_value=mock_provider):
            await asyncio.gather(
                chat_runner._process_help_messages(test_message),
                stop_after_delay()
            )
            
        # Verify stop was handled
        chunk_messages = [msg for msg in sent_messages if msg.get("type") == "chunk"]
        assert len(chunk_messages) >= 1
        # Should have completion signal
        assert any(msg.get("done") is True for msg in chunk_messages)


class TestWorkflowExecution:
    """Test stop command handling in workflow execution"""
    
    async def test_workflow_execution_completion_signal(self, chat_runner, test_message):
        """Test that workflow execution sends completion signal"""
        test_message.workflow_id = "test_workflow"
        chat_runner.chat_history = [test_message]
        
        sent_messages = []
        async def mock_send_message(msg):
            sent_messages.append(msg)
        chat_runner.send_message = mock_send_message
        
        # Mock workflow execution
        async def mock_run_workflow(request, runner, context):
            yield OutputUpdate(
                node_id="test_node_id",
                node_name="test_node", 
                output_name="test_output",
                value="test_result",
                output_type="text"
            )
            
        with patch('nodetool.common.chat_websocket_runner.run_workflow', mock_run_workflow), \
             patch('nodetool.common.chat_websocket_runner.WorkflowRunner'):
            result = await chat_runner.process_messages_for_workflow()
            
        # Verify completion signal was sent
        completion_signals = [msg for msg in sent_messages if msg.get("type") == "chunk" and msg.get("done") is True]
        assert len(completion_signals) >= 1
        assert result.role == "assistant"
        
    async def test_workflow_execution_stop_during_run(self, chat_runner, test_message):
        """Test stopping workflow execution during run"""
        test_message.workflow_id = "test_workflow"
        chat_runner.chat_history = [test_message]
        
        sent_messages = []
        async def mock_send_message(msg):
            sent_messages.append(msg)
        chat_runner.send_message = mock_send_message
        
        # Mock workflow execution with stop check
        async def mock_run_workflow(request, runner, context):
            yield OutputUpdate(
                node_id="start_node_id",
                node_name="start", 
                output_name="start_output",
                value="starting",
                output_type="text"
            )
            await asyncio.sleep(0.1)
            if chat_runner.stop_event.is_set():
                return
            yield OutputUpdate(
                node_id="end_node_id",
                node_name="end", 
                output_name="end_output",
                value="ending",
                output_type="text"
            )
            
        # Start execution and stop it
        async def stop_after_delay():
            await asyncio.sleep(0.05)
            chat_runner.stop_event.set()
            
        with patch('nodetool.common.chat_websocket_runner.run_workflow', mock_run_workflow), \
             patch('nodetool.common.chat_websocket_runner.WorkflowRunner'):
            await asyncio.gather(
                chat_runner.process_messages_for_workflow(),
                stop_after_delay()
            )
            
        # Verify stop was handled
        job_updates = [msg for msg in sent_messages if msg.get("type") == "job_update"]
        assert any(msg.get("status") == "cancelled" for msg in job_updates)


class TestWorkflowCreation:
    """Test stop command handling in workflow creation"""
    
    async def test_workflow_creation_stop_during_planning(self, chat_runner):
        """Test stopping workflow creation during planning"""
        sent_messages = []
        async def mock_send_message(msg):
            sent_messages.append(msg)
        chat_runner.send_message = mock_send_message
        
        # Mock planner with stop check
        async def mock_create_graph(context):
            yield PlanningUpdate(phase="Planning", status="Starting", content="Planning workflow")
            await asyncio.sleep(0.1)
            if chat_runner.stop_event.is_set():
                return
            yield PlanningUpdate(phase="Planning", status="Complete", content="Workflow planned")
            
        mock_planner = MagicMock()
        mock_planner.create_graph = mock_create_graph
        mock_planner.graph = None
        
        # Start creation and stop it
        async def stop_after_delay():
            await asyncio.sleep(0.05)
            chat_runner.stop_event.set()
            
        with patch('nodetool.common.chat_websocket_runner.GraphPlanner', return_value=mock_planner):
            await asyncio.gather(
                chat_runner._trigger_workflow_creation("test objective"),
                stop_after_delay()
            )
            
        # Verify stop was handled
        chunk_messages = [msg for msg in sent_messages if msg.get("type") == "chunk"]
        assert any(msg.get("done") is True for msg in chunk_messages)


class TestWorkflowEditing:
    """Test stop command handling in workflow editing"""
    
    async def test_workflow_editing_stop_during_planning(self, chat_runner):
        """Test stopping workflow editing during planning"""
        sent_messages = []
        async def mock_send_message(msg):
            sent_messages.append(msg)
        chat_runner.send_message = mock_send_message
        
        # Mock planner with stop check
        async def mock_create_graph(context):
            yield PlanningUpdate(phase="Editing", status="Starting", content="Editing workflow")
            await asyncio.sleep(0.1)
            if chat_runner.stop_event.is_set():
                return
            yield PlanningUpdate(phase="Editing", status="Complete", content="Workflow edited")
            
        mock_planner = MagicMock()
        mock_planner.create_graph = mock_create_graph
        mock_planner.graph = None
        
        # Start editing and stop it
        async def stop_after_delay():
            await asyncio.sleep(0.05)
            chat_runner.stop_event.set()
            
        with patch('nodetool.common.chat_websocket_runner.GraphPlanner', return_value=mock_planner):
            await asyncio.gather(
                chat_runner._trigger_workflow_editing("test objective", "test_workflow"),
                stop_after_delay()
            )
            
        # Verify stop was handled
        chunk_messages = [msg for msg in sent_messages if msg.get("type") == "chunk"]
        assert any(msg.get("done") is True for msg in chunk_messages)


class TestToolExecution:
    """Test stop command handling in tool execution"""
    
    async def test_tool_execution_stop_check(self, chat_runner):
        """Test that tool execution checks stop event"""
        context = ProcessingContext(user_id="test_user")
        tool_call = ToolCall(id="test", name="test_tool", args={"param": "value"})
        tools = [MockTool("test_tool")]
        
        sent_messages = []
        async def mock_send_message(msg):
            sent_messages.append(msg)
        chat_runner.send_message = mock_send_message
        
        # Set stop event before tool execution
        chat_runner.stop_event.set()
        
        result = await chat_runner.run_tool(context, tool_call, tools)
        
        # Verify tool was stopped
        assert result.result.get("stopped") is True
        assert "stopped by user" in result.result.get("error", "")
        
    async def test_tool_execution_normal_flow(self, chat_runner):
        """Test normal tool execution flow"""
        context = ProcessingContext(user_id="test_user")
        tool_call = ToolCall(id="test", name="test_tool", args={"param": "value"})
        tools = [MockTool("test_tool")]
        
        sent_messages = []
        async def mock_send_message(msg):
            sent_messages.append(msg)
        chat_runner.send_message = mock_send_message
        
        result = await chat_runner.run_tool(context, tool_call, tools)
        
        # Verify tool executed normally
        assert result.result.get("result") == "mock_result_1"
        assert result.result.get("args") == {"param": "value"}
        
        # Verify tool call update was sent
        tool_updates = [msg for msg in sent_messages if msg.get("type") == "tool_call_update"]
        assert len(tool_updates) >= 1


class TestErrorCases:
    """Test stop command handling in error cases"""
    
    async def test_connection_error_completion_signal(self, chat_runner, test_message):
        """Test that connection errors send completion signal"""
        chat_runner.chat_history = [test_message]
        
        sent_messages = []
        async def mock_send_message(msg):
            sent_messages.append(msg)
        chat_runner.send_message = mock_send_message
        
        # Mock provider that raises connection error
        async def mock_generate_messages(messages, model, tools=None):
            from httpx import ConnectError
            raise ConnectError("Connection failed")
            
        mock_provider = MagicMock()
        mock_provider.generate_messages = mock_generate_messages
        
        with patch('nodetool.common.chat_websocket_runner.get_provider', return_value=mock_provider):
            result = await chat_runner.process_messages()
            
        # Verify error and completion signal were sent
        error_messages = [msg for msg in sent_messages if msg.get("type") == "error"]
        assert len(error_messages) >= 1
        completion_signals = [msg for msg in sent_messages if msg.get("type") == "chunk" and msg.get("done") is True]
        assert len(completion_signals) >= 1
        assert "connection error" in result.content.lower()
        
    async def test_general_error_completion_signal(self, chat_runner, test_message):
        """Test that general errors send completion signal"""
        chat_runner.chat_history = [test_message]
        
        sent_messages = []
        async def mock_send_message(msg):
            sent_messages.append(msg)
        chat_runner.send_message = mock_send_message
        
        # Mock provider that raises general error
        async def mock_generate_messages(messages, model, tools=None):
            raise ValueError("General error")
            
        mock_provider = MagicMock()
        mock_provider.generate_messages = mock_generate_messages
        
        with patch('nodetool.common.chat_websocket_runner.get_provider', return_value=mock_provider):
            # This should be handled by the outer try-catch in run()
            with pytest.raises(ValueError):
                await chat_runner.process_messages()
        
    async def test_help_mode_connection_error(self, chat_runner, test_message):
        """Test help mode connection error handling"""
        test_message.help_mode = True
        chat_runner.chat_history = [test_message]
        
        sent_messages = []
        async def mock_send_message(msg):
            sent_messages.append(msg)
        chat_runner.send_message = mock_send_message
        
        # Mock provider that raises connection error
        async def mock_generate_messages(messages, model, tools=None):
            from httpx import ConnectError
            raise ConnectError("Connection failed")
            
        mock_provider = MagicMock()
        mock_provider.generate_messages = mock_generate_messages
        
        with patch('nodetool.common.chat_websocket_runner.get_provider', return_value=mock_provider):
            result = await chat_runner._process_help_messages(test_message)
            
        # Verify error and completion signal were sent
        error_messages = [msg for msg in sent_messages if msg.get("type") == "error"]
        assert len(error_messages) >= 1
        completion_signals = [msg for msg in sent_messages if msg.get("type") == "chunk" and msg.get("done") is True]
        assert len(completion_signals) >= 1
        assert "connection error" in result.content.lower()
        
    async def test_agent_mode_error_handling(self, chat_runner, test_message):
        """Test agent mode error handling"""
        test_message.agent_mode = True
        chat_runner.chat_history = [test_message]
        
        sent_messages = []
        async def mock_send_message(msg):
            sent_messages.append(msg)
        chat_runner.send_message = mock_send_message
        
        # Mock agent that raises error
        async def mock_agent_execute(context):
            raise ValueError("Agent error")
            
        with patch('nodetool.common.chat_websocket_runner.get_provider'), \
             patch('nodetool.common.chat_websocket_runner.Agent') as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.execute = mock_agent_execute
            mock_agent_class.return_value = mock_agent
            
            result = await chat_runner.process_agent_messages()
            
        # Verify error and completion signal were sent
        error_messages = [msg for msg in sent_messages if msg.get("type") == "error"]
        assert len(error_messages) >= 1
        completion_signals = [msg for msg in sent_messages if msg.get("type") == "chunk" and msg.get("done") is True]
        assert len(completion_signals) >= 1
        assert "agent execution error" in result.content.lower()


class TestMessageFormats:
    """Test message format handling"""
    
    async def test_binary_message_format(self, chat_runner, mock_websocket):
        """Test binary (MessagePack) message format"""
        chat_runner.websocket = mock_websocket
        
        test_data = {"type": "test", "content": "binary test"}
        await chat_runner.send_message(test_data)
        
        assert len(mock_websocket.sent_messages) == 1
        sent = mock_websocket.sent_messages[0]
        assert sent["type"] == "bytes"
        
        # Verify message can be unpacked
        unpacked = msgpack.unpackb(sent["data"])
        assert unpacked == test_data
        
    async def test_text_message_format(self, chat_runner, mock_websocket):
        """Test text (JSON) message format"""
        chat_runner.websocket = mock_websocket
        chat_runner.mode = chat_runner.WebSocketMode.TEXT
        
        test_data = {"type": "test", "content": "text test"}
        await chat_runner.send_message(test_data)
        
        assert len(mock_websocket.sent_messages) == 1
        sent = mock_websocket.sent_messages[0]
        assert sent["type"] == "text"
        
        # Verify message can be parsed
        parsed = json.loads(sent["data"])
        assert parsed == test_data


class TestStateManagement:
    """Test state management across different scenarios"""
    
    async def test_state_reset_after_processing(self, chat_runner, test_message):
        """Test that generation state is reset after processing"""
        chat_runner.chat_history = [test_message]
        
        # Mock provider
        mock_provider = MockProvider([Chunk(content="Response", done=True)])
        
        with patch('nodetool.common.chat_websocket_runner.get_provider', return_value=mock_provider):
            await chat_runner.process_messages()
            
        # Verify generation state was reset
        assert not chat_runner.is_generating
        assert not chat_runner.stop_event.is_set()
        
    async def test_state_reset_on_error(self, chat_runner, mock_websocket):
        """Test that generation state is reset on error"""
        chat_runner.websocket = mock_websocket
        chat_runner.is_generating = True
        chat_runner.stop_event.set()
        
        # Mock error during message processing
        error_message = {"text": json.dumps({"role": "user", "content": "test"})}
        mock_websocket.queue_message(error_message)
        
        sent_messages = []
        async def mock_send_message(msg):
            sent_messages.append(msg)
        chat_runner.send_message = mock_send_message
        
        # Mock process_messages to raise error
        async def mock_process_messages():
            raise ValueError("Processing error")
        chat_runner.process_messages = mock_process_messages
        
        with patch.object(chat_runner, 'connect'):
            try:
                await chat_runner.run(mock_websocket)
            except:
                pass  # Expected when receive() returns disconnect
        
        # Verify state was reset
        assert not chat_runner.is_generating
        assert not chat_runner.stop_event.is_set()
        
        # Verify error was sent
        error_messages = [msg for msg in sent_messages if msg.get("type") == "error"]
        assert len(error_messages) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 