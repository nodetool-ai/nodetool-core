import pytest
import os
import json
import uuid
from typing import Any, Dict, AsyncGenerator
from unittest.mock import patch, MagicMock, AsyncMock, call, ANY

from nodetool.agents.agent import Agent, SingleTaskAgent
from nodetool.agents.tools.base import Tool
from nodetool.chat.providers.base import MockProvider
from nodetool.metadata.types import Task, SubTask, Message, ToolCall
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk, TaskUpdate, TaskUpdateEvent


class MockSimpleTool(Tool):
    name: str = "mock_simple_tool"
    description: str = "A simple mock tool."
    input_schema: Dict[str, Any] = {"type": "object", "properties": {}}

    async def process(
        self, context: ProcessingContext, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {"result": "mock tool executed"}


# --- Mock Classes ---


class MockTaskPlanner(MagicMock):
    """Mock for TaskPlanner."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure async methods are AsyncMocks if needed for spec validation or default behavior
        self.create_task = AsyncMock()


class MockTaskExecutor(MagicMock):
    """Mock for TaskExecutor."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use MagicMock, return the async generator *instance*
        self.execute_tasks = MagicMock()
        # get_output_files is often synchronous
        self.get_output_files = MagicMock(return_value=[])


class MockSubTaskContext(MagicMock):
    """Mock for SubTaskContext."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.execute = AsyncMock()


# --- Fixtures ---


@pytest.fixture
def mock_processing_context(tmp_path):
    """Provides a ProcessingContext pointing to a temporary directory."""
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    input_dir = workspace_dir / "input_files"
    input_dir.mkdir()  # Ensure input_files dir exists
    return ProcessingContext(workspace_dir=str(workspace_dir))


@pytest.fixture
def mock_provider():
    """Provides a MockProvider instance."""
    return MockProvider([])  # Initially empty, responses added in tests


@pytest.fixture
def mock_tool():
    """Provides a simple mock tool."""
    return MockSimpleTool()


@pytest.fixture
def basic_agent_config(mock_provider, mock_tool):
    """Provides a basic configuration dictionary for Agent/SingleTaskAgent."""
    return {
        "name": "TestAgent",
        "objective": "Test objective",
        "provider": mock_provider,
        "model": "test-model",
        "tools": [mock_tool],
        "input_files": [],
    }


@pytest.fixture
def sample_task():
    """Provides a sample Task with a couple of subtasks."""
    subtask1 = SubTask(
        content="Subtask 1",
        output_file="sub1.txt",
        output_type="text",
        output_schema='{"type": "string"}',
        input_files=[],
    )
    subtask2 = SubTask(
        content="Subtask 2",
        output_file="sub2.json",
        output_type="json",
        output_schema='{"type": "object"}',
        input_files=["sub1.txt"],
    )
    return Task(title="Sample Task", subtasks=[subtask1, subtask2])


@pytest.fixture(autouse=True)
def mock_dependencies(mocker):
    """Automatically mock external dependencies used by Agent/SingleTaskAgent."""
    mocker.patch("nodetool.agents.agent.shutil.copy")
    mocker.patch("nodetool.agents.agent.os.makedirs")
    mocker.patch(
        "nodetool.agents.agent.get_log_path", return_value="mock_log_path.jsonl"
    )
    mocker.patch("nodetool.agents.agent.Live")  # Mock the rich Live display
    mocker.patch(
        "nodetool.agents.agent.Environment"
    )  # Mock environment variable access
    mocker.patch(
        "nodetool.agents.agent.uuid.uuid4",
        return_value=uuid.UUID("12345678-1234-5678-1234-567812345678"),
    )  # Predictable UUIDs
    mocker.patch(
        "nodetool.agents.agent.clean_and_validate_path",
        side_effect=lambda ws, p, n: p,  # Revert to simple relative path mock
        # side_effect=lambda ws, p, n: os.path.join(ws, p) # Previous refined mock
    )  # Simple mock


# --- Helper Function ---


async def consume_async_generator(agen: AsyncGenerator):
    """Helper to consume an async generator into a list."""
    return [item async for item in agen]


@pytest.mark.asyncio
@patch("nodetool.agents.agent.TaskPlanner", new_callable=MockTaskPlanner)
@patch("nodetool.agents.agent.TaskExecutor", new_callable=MockTaskExecutor)
@patch("nodetool.agents.agent.shutil.copy")
async def test_agent_input_files_copy(
    mock_shutil_copy,
    MockTaskExecutorClass,
    MockTaskPlannerClass,
    basic_agent_config,
    mock_processing_context,
    tmp_path,
    sample_task,
):
    """Test that Agent copies input files to the workspace."""
    # Create dummy input files
    input_file1_path = tmp_path / "input1.txt"
    input_file2_path = tmp_path / "subdir" / "input2.csv"
    input_file2_path.parent.mkdir()
    input_file1_path.write_text("content1")
    input_file2_path.write_text("content2")

    config = {
        **basic_agent_config,
        "input_files": [str(input_file1_path), str(input_file2_path)],
    }
    agent = Agent(**config)  # type: ignore

    # Mock instances are accessed via .return_value
    mock_planner_instance = MockTaskPlannerClass.return_value
    mock_executor_instance = MockTaskExecutorClass.return_value

    # Configure the instances
    mock_planner_instance.create_task.return_value = sample_task
    # mock_executor_instance.execute_tasks = AsyncMock() # No need, already AsyncMock from class def

    async def mock_executor_gen(*args, **kwargs):  # Need to yield at least one thing
        yield TaskUpdate(
            task=sample_task,
            subtask=sample_task.subtasks[0],
            event=TaskUpdateEvent.SUBTASK_STARTED,
        )

    # Use side_effect for AsyncMock with async generator
    mock_executor_instance.execute_tasks.side_effect = mock_executor_gen
    # mock_executor_instance.execute_tasks.return_value = mock_executor_gen() # Incorrect for AsyncMock
    mock_executor_instance.get_output_files.return_value = (
        []
    )  # Configure instance return value

    await consume_async_generator(agent.execute(mock_processing_context))

    # Assert shutil.copy calls
    expected_dest1 = os.path.join(
        mock_processing_context.workspace_dir, "input_files", "input1.txt"
    )
    expected_dest2 = os.path.join(
        mock_processing_context.workspace_dir, "input_files", "input2.csv"
    )
    mock_shutil_copy.assert_has_calls(
        [
            call(str(input_file1_path), expected_dest1),
            call(str(input_file2_path), expected_dest2),
        ],
        any_order=True,
    )

    # Assert class was instantiated with correct args
    relative_input_files = [
        os.path.join("input_files", "input1.txt"),
        os.path.join("input_files", "input2.csv"),
    ]
    MockTaskPlannerClass.assert_called_once_with(
        provider=ANY,
        model=ANY,
        objective=ANY,
        workspace_dir=ANY,
        execution_tools=ANY,
        retrieval_tools=ANY,
        input_files=relative_input_files,  # Check this specifically
        output_schema=ANY,
        enable_retrieval_phase=ANY,
        enable_analysis_phase=ANY,
        enable_data_contracts_phase=ANY,
        use_structured_output=ANY,
    )
    MockTaskExecutorClass.assert_called_once_with(
        provider=ANY,
        model=ANY,
        processing_context=ANY,
        tools=ANY,
        task=ANY,
        system_prompt=ANY,
        input_files=relative_input_files,  # Check this specifically
        max_steps=ANY,
        max_subtask_iterations=ANY,
        max_token_limit=ANY,
    )
    # Assert methods were called on the instances
    mock_planner_instance.create_task.assert_awaited_once()
    mock_executor_instance.execute_tasks.assert_called_once()


@pytest.mark.asyncio
@patch("nodetool.agents.agent.TaskPlanner", new_callable=MockTaskPlanner)
@patch("nodetool.agents.agent.TaskExecutor", new_callable=MockTaskExecutor)
async def test_agent_output_schema_and_type(
    MockTaskExecutorClass,
    MockTaskPlannerClass,
    basic_agent_config,
    mock_processing_context,
    sample_task,
):
    """Test that output_schema and output_type are added to the last subtask."""
    # Create a deep copy for modification in this test
    local_sample_task = Task(
        title=sample_task.title,
        subtasks=[st.copy(deep=True) for st in sample_task.subtasks],
    )

    output_schema = {"type": "object", "properties": {"final_key": {"type": "string"}}}
    output_type = "json"

    config = {
        **basic_agent_config,
        "output_schema": output_schema,
        "output_type": output_type,
    }
    agent = Agent(**config)

    # Mock instances and configure them
    mock_planner_instance = MockTaskPlannerClass.return_value
    mock_executor_instance = MockTaskExecutorClass.return_value
    mock_planner_instance.create_task.return_value = (
        local_sample_task  # Return the copy
    )

    async def mock_executor_gen(*args, **kwargs):  # Need to yield at least one thing
        yield TaskUpdate(
            task=local_sample_task,
            subtask=local_sample_task.subtasks[0],
            event=TaskUpdateEvent.SUBTASK_STARTED,
        )

    # Use side_effect for AsyncMock with async generator
    mock_executor_instance.execute_tasks.side_effect = mock_executor_gen
    # mock_executor_instance.execute_tasks.return_value = mock_executor_gen() # Incorrect

    await consume_async_generator(agent.execute(mock_processing_context))

    # Assertions
    mock_planner_instance.create_task.assert_awaited_once()

    # Check that the last subtask in the local_sample_task object was modified *before* passing to Executor
    assert local_sample_task.subtasks[-1].output_type == output_type
    assert local_sample_task.subtasks[-1].output_schema == json.dumps(output_schema)

    # Check that the modified task was passed to the Executor
    MockTaskExecutorClass.assert_called_once_with(
        provider=ANY,
        model=ANY,
        processing_context=ANY,
        tools=ANY,
        task=local_sample_task,  # Ensure the modified task is passed
        system_prompt=ANY,
        input_files=ANY,
        max_steps=ANY,
        max_subtask_iterations=ANY,
        max_token_limit=ANY,
    )
    mock_executor_instance.execute_tasks.assert_called_once()


@pytest.mark.asyncio
@patch("nodetool.agents.agent.TaskPlanner", new_callable=MockTaskPlanner)
@patch("nodetool.agents.agent.TaskExecutor", new_callable=MockTaskExecutor)
async def test_agent_results_without_finish_task(
    MockTaskExecutorClass,
    MockTaskPlannerClass,
    basic_agent_config,
    mock_processing_context,
    sample_task,
):
    """Test that agent results are taken from executor's output files if finish_task is not called."""
    agent = Agent(**basic_agent_config)

    # Mock instances and configure them
    mock_planner_instance = MockTaskPlannerClass.return_value
    mock_executor_instance = MockTaskExecutorClass.return_value
    mock_planner_instance.create_task.return_value = sample_task
    # mock_executor_instance.execute_tasks = AsyncMock() # Already AsyncMock

    # Simulate execution *without* yielding a finish_task ToolCall
    async def mock_executor_gen(*args, **kwargs):
        yield TaskUpdate(
            task=sample_task,
            subtask=sample_task.subtasks[0],
            event=TaskUpdateEvent.SUBTASK_STARTED,
        )
        yield TaskUpdate(
            task=sample_task,
            subtask=sample_task.subtasks[0],
            event=TaskUpdateEvent.SUBTASK_COMPLETED,
        )
        # No finish_task call

    # Use side_effect for AsyncMock with async generator
    mock_executor_instance.execute_tasks.side_effect = mock_executor_gen
    # mock_executor_instance.execute_tasks.return_value = mock_executor_gen() # Incorrect
    # Mock get_output_files which should be called in this case
    expected_output_files = ["path/to/output1.txt", "another/output.json"]
    mock_executor_instance.get_output_files.return_value = (
        expected_output_files  # Configure instance return
    )

    await consume_async_generator(agent.execute(mock_processing_context))

    # Assertions
    assert agent.get_results() == expected_output_files
    # Assert instance methods were called
    mock_planner_instance.create_task.assert_awaited_once()
    mock_executor_instance.execute_tasks.assert_called_once()
    mock_executor_instance.get_output_files.assert_called_once()  # Assert get_output_files was called


# --- SingleTaskAgent Class Tests ---


@pytest.mark.asyncio
async def test_single_task_agent_initialization(
    basic_agent_config, mock_processing_context
):
    """Test SingleTaskAgent initialization."""
    # Add required args not in basic_agent_config for SingleTaskAgent
    agent = SingleTaskAgent(
        output_type="text", output_schema={"type": "string"}, **basic_agent_config
    )
    # Assertions based on __init__ parameters
    assert agent.name == basic_agent_config["name"]
    assert agent.objective == basic_agent_config["objective"]
    assert agent.provider == basic_agent_config["provider"]
    assert agent.model == basic_agent_config["model"]
    assert agent.tools == basic_agent_config["tools"]
    assert (
        agent.input_files == basic_agent_config["input_files"]
    )  # Should be [] from fixture
    assert agent.output_type == "text"
    assert agent.output_schema == {"type": "string"}
    assert agent.execution_system_prompt is None


@pytest.mark.asyncio
async def test_single_task_agent_plan_single_subtask_fail_validation(
    basic_agent_config, mock_processing_context, mock_provider
):
    """Test planning failure due to validation errors after max retries."""
    objective = "Invalid plan"
    # Simulate responses that always fail validation (e.g., missing required field)
    mock_invalid_definition = {
        "content": "Missing type",
        "output_schema": '{"type": "string"}',  # Missing output_type
    }
    mock_provider.mock_responses = [
        Message(role="assistant", content=json.dumps(mock_invalid_definition)),
        Message(role="assistant", content=json.dumps(mock_invalid_definition)),
        Message(role="assistant", content=json.dumps(mock_invalid_definition)),
    ]

    agent = SingleTaskAgent(
        name="FailAgent",
        output_type="text",
        output_schema=json.loads(mock_invalid_definition["output_schema"]),
        objective=objective,
        provider=mock_provider,
        model="fail-model",
        tools=[],
    )

    with pytest.raises(
        ValueError, match="Failed to plan single subtask after 3 attempts"
    ):
        await agent._plan_single_subtask(context=mock_processing_context, max_retries=3)

    assert agent.task is None
    assert agent.subtask is None
    assert len(mock_provider.call_log) == 3
