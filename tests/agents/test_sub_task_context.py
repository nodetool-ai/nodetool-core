# Create this file: nodetool-core/tests/agents/test_sub_task_context.py
import pytest
import os
import json
import yaml
from typing import Dict, Any

from nodetool.agents.sub_task_context import (
    SubTaskContext,
)
from nodetool.chat.providers.base import MockProvider  # Import the new mock provider
from nodetool.metadata.types import Task, SubTask, Message, ToolCall
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk, TaskUpdate, TaskUpdateEvent
from nodetool.agents.tools.base import Tool


# --- Helper Functions for History Assertion ---


def _format_history_summary(history: list[Message]) -> str:
    """Formats the history list for readable assertion output.

    Creates a multi-line string where each line summarizes a message
    in the history, including role, tool name (if applicable),
    tool calls, and content existence.

    Args:
        history: The list of Message objects from a SubTaskContext.

    Returns:
        A formatted string representing the history summary.
    """
    summary = []
    for i, msg in enumerate(history):
        details = f"role='{msg.role}'"
        if msg.role == "tool" and msg.name:
            details += f", name='{msg.name}'"
        if msg.tool_calls:
            tool_names = [tc.name for tc in msg.tool_calls if tc.name]
            if tool_names:
                details += f", tool_calls=[{', '.join(tool_names)}]"
            else:
                details += f", tool_calls=[{len(msg.tool_calls)} present]"
        if msg.content is not None:
            details += ", content_exists=True"
        else:
            details += ", content_exists=False"
        summary.append(f"  [{i}] {details}")
    return "\n".join(summary)


def _format_expected_summary(structure: list[dict]) -> str:
    """Formats the expected structure list for readable assertion output.

    Creates a multi-line string where each line represents an expected
    message specification dictionary.

    Args:
        structure: The list of dictionaries defining the expected history structure.

    Returns:
        A formatted string representing the expected structure summary.
    """
    summary = []
    for i, spec in enumerate(structure):
        summary.append(f"  [{i}] {spec}")
    return "\n".join(summary)


def assert_history_structure(
    actual_history: list[Message], expected_structure: list[dict]
):
    """
    Asserts that the actual history matches the expected structure.

    Compares the length and attributes of messages in the actual history list
    against a list of expected attribute dictionaries. Provides detailed
    pytest failure messages if mismatches are found.

    Args:
        actual_history: The list of Message objects from the context.
        expected_structure: A list of dictionaries, where each dictionary
                            specifies expected attributes for a message at
                            that position. Supported keys:
                            - 'role' (str): Required. The expected role.
                            - 'tool_name' (str): Optional. Checks message name if role='tool',
                                               or the name of the *first* tool_call
                                               if role='assistant'.
                            - 'content_exists' (bool): Optional. Checks if content is not None.
                            - 'has_tool_calls' (bool): Optional. Checks if tool_calls list
                                                      exists and is not empty.
                            - 'tool_args' (dict): Optional. If role='assistant', checks if the
                                                first tool_call's args match this dict.

    Raises:
        pytest.fail: If the history does not match the expected structure,
                     providing detailed comparison information.
    """
    if len(actual_history) != len(expected_structure):
        pytest.fail(
            f"History length mismatch:\n"
            f"  Actual: {len(actual_history)}\n"
            f"  Expected: {len(expected_structure)}\n\n"
            f"Actual History:\n{_format_history_summary(actual_history)}\n\n"
            f"Expected Structure:\n{_format_expected_summary(expected_structure)}"
        )

    for i, (actual_msg, expected_spec) in enumerate(
        zip(actual_history, expected_structure)
    ):
        errors = []
        spec_role = expected_spec.get("role")
        spec_tool_name = expected_spec.get("tool_name")
        spec_content_exists = expected_spec.get("content_exists")
        spec_has_tool_calls = expected_spec.get("has_tool_calls")
        spec_tool_args = expected_spec.get("tool_args")

        # --- Role check ---
        if actual_msg.role != spec_role:
            errors.append(f"Expected role '{spec_role}', got '{actual_msg.role}'")

        # --- Tool name check ---
        if spec_tool_name:
            actual_tool_name = None
            if actual_msg.role == "tool":
                actual_tool_name = actual_msg.name
            elif actual_msg.role == "assistant" and actual_msg.tool_calls:
                actual_tool_name = actual_msg.tool_calls[
                    0
                ].name  # Check first tool call

            if actual_tool_name != spec_tool_name:
                errors.append(
                    f"Expected tool_name '{spec_tool_name}', got '{actual_tool_name}'"
                )

        # --- Content existence check ---
        actual_content_exists = actual_msg.content is not None
        if spec_content_exists is True and not actual_content_exists:
            errors.append(f"Expected content to exist, but it was None")
        elif spec_content_exists is False and actual_content_exists:
            errors.append(f"Expected content to be None, but it existed")

        # --- Has tool calls check ---
        actual_has_tool_calls = bool(actual_msg.tool_calls)
        if spec_has_tool_calls is True and not actual_has_tool_calls:
            errors.append(f"Expected tool_calls to exist, but list was empty or None")
        elif spec_has_tool_calls is False and actual_has_tool_calls:
            errors.append(f"Expected no tool_calls, but they existed")
        # Also check if has_tool_calls was expected (implicitly True if tool_name is given for assistant)
        elif (
            spec_tool_name
            and actual_msg.role == "assistant"
            and not actual_has_tool_calls
        ):
            errors.append(
                f"Expected tool_calls (for tool_name '{spec_tool_name}'), but list was empty or None"
            )

        # --- Tool args check (for assistant role) ---
        if spec_tool_args and actual_msg.role == "assistant":
            if not actual_msg.tool_calls:
                errors.append(
                    f"Expected tool_args {spec_tool_args}, but no tool_calls found"
                )
            elif actual_msg.tool_calls[0].args != spec_tool_args:
                errors.append(
                    f"Expected tool_args {spec_tool_args}, got {actual_msg.tool_calls[0].args}"
                )

        if errors:
            # Format the actual message details for the error message
            actual_details = f"role='{actual_msg.role}'"
            if actual_msg.role == "tool" and actual_msg.name:
                actual_details += f", name='{actual_msg.name}'"
            if actual_msg.tool_calls:
                tool_names = [tc.name for tc in actual_msg.tool_calls if tc.name]
                actual_details += f", tool_calls=[{', '.join(tool_names)}]"
            actual_details += f", content_exists={actual_content_exists}"

            pytest.fail(
                f"Mismatch at history index {i}:\n"
                f"  Expected Spec:  {expected_spec}\n"
                f"  Actual Message: {actual_details}\n"
                f"  Errors: {'; '.join(errors)}\n\n"
                f"Full Actual History:\n{_format_history_summary(actual_history)}\n\n"
                f"Full Expected Structure:\n{_format_expected_summary(expected_structure)}"
            )


def _format_call_log_summary(call_log: list[dict]) -> str:
    """Formats the call log list for readable assertion output.

    Creates a multi-line string summarizing the calls made to the mock provider,
    including method name, tool names, response format details, and model name.

    Args:
        call_log: The list of call dictionaries recorded by the MockProvider.

    Returns:
        A formatted string representing the call log summary.
    """
    summary = []
    for i, call in enumerate(call_log):
        details = f"method='{call.get('method', 'N/A')}'"
        if "tools" in call:
            tool_names = [t.name for t in call["tools"] if hasattr(t, "name")]
            details += f", tools=[{', '.join(tool_names)}]"
        if "kwargs" in call:
            if (
                "response_format" in call["kwargs"]
                and call["kwargs"]["response_format"]
            ):
                rf_type = call["kwargs"]["response_format"].get("type", "N/A")
                rf_name = (
                    call["kwargs"]["response_format"]
                    .get("json_schema", {})
                    .get("name", "N/A")
                )
                details += f", response_format={{type='{rf_type}', name='{rf_name}'}}"
            if "model" in call["kwargs"]:
                details += f", model='{call['kwargs']['model']}'"
        summary.append(f"  [{i}] {details}")
    return "\n".join(summary)


def _format_expected_call_log_summary(structure: list[dict]) -> str:
    """Formats the expected call log structure list for readable assertion output.

    Creates a multi-line string where each line represents an expected
    call specification dictionary.

    Args:
        structure: The list of dictionaries defining the expected call log structure.

    Returns:
        A formatted string representing the expected structure summary.
    """
    summary = []
    for i, spec in enumerate(structure):
        summary.append(f"  [{i}] {spec}")
    return "\n".join(summary)


def assert_call_log_structure(
    actual_call_log: list[dict], expected_structure: list[dict]
):
    """
    Asserts that the actual provider call log matches the expected structure.

    Compares the length and attributes of calls in the actual call log list
    against a list of expected attribute dictionaries. Provides detailed
    pytest failure messages if mismatches are found.

    Args:
        actual_call_log: The list of call dictionaries from the provider.
        expected_structure: A list of dictionaries, where each dictionary
                            specifies expected attributes for a call at
                            that position. Supported keys:
                            - 'method' (str): Required. The expected method name
                              (e.g., 'generate_message').
                            - 'tool_names' (list[str]): Optional. Checks if *exactly* these
                                                      tool names were present in the call's
                                                      'tools' argument. Order doesn't matter.
                            - 'num_tools' (int): Optional. Checks the number of tools
                                               passed in the 'tools' argument.
                            - 'has_response_format' (bool): Optional. Checks if 'response_format'
                                                           exists in kwargs and is not None/empty.
                            - 'response_format_type' (str): Optional. Checks the 'type' within
                                                            'response_format' (e.g., 'json_schema').
                            - 'response_format_name' (str): Optional. Checks the 'name' within
                                                           'response_format.json_schema'.

    Raises:
        pytest.fail: If the call log does not match the expected structure,
                     providing detailed comparison information.
    """
    if len(actual_call_log) != len(expected_structure):
        pytest.fail(
            f"Call log length mismatch:\n"
            f"  Actual: {len(actual_call_log)}\n"
            f"  Expected: {len(expected_structure)}\n\n"
            f"Actual Call Log:\n{_format_call_log_summary(actual_call_log)}\n\n"
            f"Expected Structure:\n{_format_expected_call_log_summary(expected_structure)}"
        )

    for i, (actual_call, expected_spec) in enumerate(
        zip(actual_call_log, expected_structure)
    ):
        errors = []
        spec_method = expected_spec.get("method")
        spec_tool_names = expected_spec.get("tool_names")
        spec_num_tools = expected_spec.get("num_tools")
        spec_has_rf = expected_spec.get("has_response_format")
        spec_rf_type = expected_spec.get("response_format_type")
        spec_rf_name = expected_spec.get("response_format_name")

        actual_method = actual_call.get("method")
        actual_tools = actual_call.get("tools", [])
        actual_tool_names = sorted([t.name for t in actual_tools if hasattr(t, "name")])
        actual_kwargs = actual_call.get("kwargs", {})
        actual_response_format = actual_kwargs.get("response_format")

        # --- Method check ---
        if actual_method != spec_method:
            errors.append(f"Expected method '{spec_method}', got '{actual_method}'")

        # --- Tool checks ---
        if spec_tool_names is not None:
            expected_tool_names_sorted = sorted(spec_tool_names)
            if actual_tool_names != expected_tool_names_sorted:
                errors.append(
                    f"Expected tool_names {expected_tool_names_sorted}, got {actual_tool_names}"
                )
        if spec_num_tools is not None:
            actual_num_tools = len(actual_tools)
            if actual_num_tools != spec_num_tools:
                errors.append(
                    f"Expected num_tools {spec_num_tools}, got {actual_num_tools}"
                )

        # --- Response format checks ---
        actual_has_rf = bool(actual_response_format)
        if spec_has_rf is True and not actual_has_rf:
            errors.append(
                "Expected 'response_format' in kwargs, but it was missing or None"
            )
        elif spec_has_rf is False and actual_has_rf:
            errors.append("Expected no 'response_format' in kwargs, but it existed")

        if actual_has_rf:
            actual_rf_type = actual_response_format.get("type")
            actual_rf_schema = actual_response_format.get("json_schema", {})
            actual_rf_name = actual_rf_schema.get("name")

            if spec_rf_type and actual_rf_type != spec_rf_type:
                errors.append(
                    f"Expected response_format type '{spec_rf_type}', got '{actual_rf_type}'"
                )
            if spec_rf_name and actual_rf_name != spec_rf_name:
                errors.append(
                    f"Expected response_format name '{spec_rf_name}', got '{actual_rf_name}'"
                )

        if errors:
            pytest.fail(
                f"Mismatch at call log index {i}:\n"
                f"  Expected Spec:  {expected_spec}\n"
                f"  Actual Call:   method='{actual_method}', "
                f"tools={actual_tool_names}, "
                f"response_format={actual_response_format}\n"
                f"  Errors: {'; '.join(errors)}\n\n"
                f"Full Actual Call Log:\n{_format_call_log_summary(actual_call_log)}\n\n"
                f"Full Expected Structure:\n{_format_expected_call_log_summary(expected_structure)}"
            )


# --- Test Fixtures ---


@pytest.fixture
def mock_processing_context(tmp_path):
    """Provides a ProcessingContext pointing to a temporary directory.

    Creates a 'workspace' subdirectory within the pytest temporary directory
    and initializes a ProcessingContext with this path.

    Args:
        tmp_path: The pytest fixture providing a temporary directory path.

    Yields:
        ProcessingContext: An instance configured with a temporary workspace.
    """
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    return ProcessingContext(workspace_dir=str(workspace_dir))


class MockSimpleTool(Tool):
    """A simple mock tool for testing tool execution within SubTaskContext.

    Implements the base Tool interface with minimal functionality.
    Its `process` method returns a dictionary indicating the input it received.

    Attributes:
        name: The identifier for this tool ("mock_tool").
        description: A brief description of the tool.
        input_schema: A JSON schema defining the expected input parameters.
    """

    name: str = "mock_tool"
    description: str = "A simple mock tool."
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {"input": {"type": "string"}},
    }

    async def process(
        self, context: ProcessingContext, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {"output": f"Processed: {params.get('input', '')}"}


@pytest.fixture
def mock_tool_instance(mock_processing_context):
    """Provides an instance of MockSimpleTool.

    Initializes the MockSimpleTool using the workspace directory provided
    by the `mock_processing_context` fixture.

    Args:
        mock_processing_context: The fixture providing a ProcessingContext.

    Returns:
        MockSimpleTool: An instance of the mock tool.
    """
    return MockSimpleTool(mock_processing_context.workspace_dir)


@pytest.fixture
def basic_task():
    """Provides a basic Task object for use in tests.

    Represents a simple parent task containing subtasks.

    Returns:
        Task: A Task instance with a title and description.
    """
    return Task(
        title="Test Task",
        description="A task for testing SubTaskContext",
        subtasks=[],
    )


@pytest.fixture
def execution_subtask():
    """Provides a basic execution SubTask object.

    Represents a standard subtask designed to generate a simple string output
    to a specified file (`output.txt`). This fixture is typically used for
    testing general execution flows, tool usage (excluding finish_task),
    and completion logic within `SubTaskContext`.

    Returns:
        SubTask: A SubTask instance configured for simple execution.
    """
    return SubTask(
        content="Perform a basic execution step.",
        input_files=[],
        output_file="output.txt",
        output_type="string",
        output_schema='{"type": "string"}',
    )


@pytest.fixture
def finish_subtask():
    """Provides a subtask intended for the finish_task tool.

    Represents a subtask specifically designed to be handled by the `finish_task`
    tool, usually for aggregating results from previous subtasks. It specifies
    input files and a markdown output file. This fixture is used to test the
    `use_finish_task=True` mode of `SubTaskContext`.

    Returns:
        SubTask: A SubTask instance configured for aggregation via `finish_task`.
    """
    return SubTask(
        content="Aggregate results.",
        input_files=["input1.txt", "input2.json"],
        output_file="final_report.md",
        output_type="markdown",
        output_schema='{"type": "string", "contentMediaType": "text/markdown"}',
    )


@pytest.fixture
def binary_output_subtask():
    """Provides a subtask with a binary output type.

    Represents a subtask intended to produce a non-text file (e.g., an image).
    While the actual content generation isn't mocked here, this fixture helps
    test how `SubTaskContext` handles subtasks with non-standard `output_type`.

    Returns:
        SubTask: A SubTask instance configured for binary output.
    """
    return SubTask(
        content="Generate a binary file.",
        input_files=[],
        output_file="output.png",
        output_type="png",  # Binary type
        output_schema="",
    )


# --- Test Functions ---


@pytest.mark.asyncio
async def test_simple_execution_completion(
    mock_processing_context, basic_task, execution_subtask
):
    """Test a simple execution flow: user prompt -> assistant response -> finish_subtask."""
    final_content = "This is the final result."
    metadata = {
        "title": "Simple Result",
        "description": "Test output",
        "sources": ["test://source"],
    }

    mock_responses = [
        Message(role="assistant", content="Okay, I will generate the result."),
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call_finish",
                    name="finish_subtask",
                    args={"result": final_content, "metadata": metadata},
                )
            ],
        ),
    ]
    provider = MockProvider(mock_responses)

    context = SubTaskContext(
        task=basic_task,
        subtask=execution_subtask,
        processing_context=mock_processing_context,
        tools=[],
        model="test-model",
        provider=provider,
    )

    updates = [update async for update in context.execute()]

    # --- Assertions ---
    assert execution_subtask.completed
    assert execution_subtask.end_time is not None

    # Check yielded updates
    assert any(
        isinstance(u, TaskUpdate) and u.event == TaskUpdateEvent.SUBTASK_STARTED
        for u in updates
    )
    assert any(
        isinstance(u, Chunk) and u.content == mock_responses[0].content for u in updates
    )
    assert any(isinstance(u, ToolCall) and u.name == "finish_subtask" for u in updates)
    assert any(
        isinstance(u, TaskUpdate) and u.event == TaskUpdateEvent.SUBTASK_COMPLETED
        for u in updates
    )

    # Check provider calls using helper
    assert_call_log_structure(
        provider.call_log,
        [{"method": "generate_message"}, {"method": "generate_message"}],
    )

    # Check history using helper function
    expected_history = [
        {"role": "system"},
        {"role": "user"},
        {"role": "assistant", "content": mock_responses[0].content},
        {"role": "assistant", "tool_calls": ["finish_subtask"]},
        {
            "role": "tool",
            "name": "finish_subtask",
            "tool_call_id": "call_finish",
            "content": {"result": final_content, "metadata": metadata},
        },
    ]
    assert_history_structure(context.history, expected_history)

    # Check output file
    output_path = os.path.join(
        mock_processing_context.workspace_dir, execution_subtask.output_file
    )
    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        content = f.read()
        assert final_content in content  # Simple check for .txt


@pytest.mark.asyncio
async def test_tool_call_and_completion(
    mock_processing_context, basic_task, execution_subtask, mock_tool_instance
):
    """Test execution with a tool call before finishing."""
    tool_input = "some data"
    tool_output = {"output": f"Processed: {tool_input}"}
    final_content = f"Final result based on tool: {tool_output['output']}"
    metadata = {
        "title": "Tool Result",
        "description": "Output using tool",
        "sources": ["tool://mock_tool"],
    }

    mock_responses = [
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(id="call_mock", name="mock_tool", args={"input": tool_input})
            ],
        ),
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call_finish",
                    name="finish_subtask",
                    args={"result": final_content, "metadata": metadata},
                )
            ],
        ),
    ]
    provider = MockProvider(mock_responses)

    context = SubTaskContext(
        task=basic_task,
        subtask=execution_subtask,
        processing_context=mock_processing_context,
        tools=[mock_tool_instance],  # Provide the mock tool
        model="test-model",
        provider=provider,
    )

    updates = [update async for update in context.execute()]

    # --- Assertions ---
    assert execution_subtask.completed
    assert any(isinstance(u, ToolCall) and u.name == "mock_tool" for u in updates)
    assert any(isinstance(u, ToolCall) and u.name == "finish_subtask" for u in updates)
    assert any(
        isinstance(u, TaskUpdate) and u.event == TaskUpdateEvent.SUBTASK_COMPLETED
        for u in updates
    )

    assert len(provider.call_log) == 2

    # Define the expected history structure
    expected_history = [
        {"role": "system"},  # Initial system prompt
        {"role": "user"},  # User prompt derived from subtask
        {
            "role": "assistant",
            "has_tool_calls": True,
            "tool_name": "mock_tool",
        },  # LLM calls mock_tool
        {
            "role": "tool",
            "tool_name": "mock_tool",
            "content_exists": True,
        },  # mock_tool result
        {
            "role": "assistant",
            "has_tool_calls": True,
            "tool_name": "finish_subtask",
        },  # LLM calls finish_subtask
        {
            "role": "tool",
            "tool_name": "finish_subtask",
            "content_exists": True,
        },  # finish_subtask result (args)
    ]
    # Use the new assertion function
    assert_history_structure(context.history, expected_history)

    # Check output file
    output_path = os.path.join(
        mock_processing_context.workspace_dir, execution_subtask.output_file
    )
    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        assert final_content in f.read()


@pytest.mark.asyncio
async def test_max_iterations_reached(
    mock_processing_context, basic_task, execution_subtask
):
    """Test behavior when max_iterations is reached."""
    max_iters = 3
    forced_result = "Forced result due to max iterations."
    forced_metadata = {
        "title": "Max Iterations",
        "description": "Forced completion",
        "sources": [],
    }

    # Simulate responses that *don't* call finish_subtask
    mock_responses = [
        Message(role="assistant", content="Thinking..."),
        Message(role="assistant", content="Still thinking..."),
        Message(role="assistant", content="Still thinking..."),  # Third response
        # Add this fourth response for the forced structured output call
        Message(
            role="assistant",
            content=json.dumps({"result": forced_result, "metadata": forced_metadata}),
        ),
    ]
    provider = MockProvider(mock_responses)

    context = SubTaskContext(
        task=basic_task,
        subtask=execution_subtask,
        processing_context=mock_processing_context,
        tools=[],
        model="test-model",
        provider=provider,
        max_iterations=max_iters,
    )

    updates = [update async for update in context.execute()]

    # --- Assertions ---
    assert context.iterations == max_iters
    assert execution_subtask.completed  # Should be completed by force
    assert any(
        isinstance(u, TaskUpdate) and u.event == TaskUpdateEvent.MAX_ITERATIONS_REACHED
        for u in updates
    )
    # The last call is the forced one, which doesn't yield a ToolCall update in the normal loop
    # but the forced call should be logged by the provider
    # Check provider calls using helper
    assert_call_log_structure(
        provider.call_log,
        [
            {"method": "generate_message"},
            {"method": "generate_message"},
            {"method": "generate_message"},
            {
                "method": "generate_message",
                "has_response_format": True,
                "response_format_type": "json_schema",
                "response_format_name": "finish_subtask",
            },
        ],
    )
    # Check history - includes the forced tool call and result
    expected_history = [
        {"role": "system"},  # Initial system prompt
        {"role": "user"},  # User prompt
        {"role": "assistant", "content": "Thinking..."},  # First response
        {"role": "assistant", "content": "Still thinking..."},  # Second response
        {"role": "assistant", "content": "Still thinking..."},  # Third response
        {"role": "assistant", "content_exists": True},  # Forced JSON response
        {
            "role": "assistant",
            "has_tool_calls": True,
            "tool_name": "finish_subtask",
            "tool_args": {
                "result": forced_result,
                "metadata": forced_metadata,
            },
        },  # Implicit tool call
    ]
    assert_history_structure(context.history, expected_history)

    # Check output file
    output_path = os.path.join(
        mock_processing_context.workspace_dir, execution_subtask.output_file
    )
    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        assert forced_result in f.read()


@pytest.mark.asyncio
async def test_conclusion_stage_transition(
    mock_processing_context, basic_task, execution_subtask
):
    """Test transition to conclusion stage due to token limit."""
    final_content = "Final conclusion."
    metadata = {
        "title": "Conclusion",
        "description": "Reached token limit",
        "sources": [],
    }

    # Simulate responses leading to conclusion stage
    mock_responses = [
        Message(
            role="assistant", content="Generating initial content..."
        ),  # Iteration 1
        # Iteration 2 - Provider response includes the finish call
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="conclude",
                    name="finish_subtask",
                    args={"result": final_content, "metadata": metadata},
                )
            ],
        ),
    ]
    provider = MockProvider(mock_responses)

    # Set a low token limit to force conclusion stage after first response
    # Rough estimate: system + user + first_assistant > 50
    context = SubTaskContext(
        task=basic_task,
        subtask=execution_subtask,
        processing_context=mock_processing_context,
        tools=[
            MockSimpleTool(mock_processing_context.workspace_dir)
        ],  # Include a dummy tool
        model="test-model",
        provider=provider,
        max_token_limit=50,  # Low limit
    )

    updates = [update async for update in context.execute()]

    # --- Assertions ---
    assert context.in_conclusion_stage  # Should have entered conclusion stage
    assert execution_subtask.completed
    assert any(
        isinstance(u, TaskUpdate)
        and u.event == TaskUpdateEvent.ENTERED_CONCLUSION_STAGE
        for u in updates
    )
    assert any(isinstance(u, ToolCall) and u.name == "finish_subtask" for u in updates)
    assert any(
        isinstance(u, TaskUpdate) and u.event == TaskUpdateEvent.SUBTASK_COMPLETED
        for u in updates
    )
    # Check provider calls using helper
    assert_call_log_structure(
        provider.call_log,
        [
            {
                "method": "generate_message",
                "num_tools": 2,
                "tool_names": ["finish_subtask", "read_workspace_file"],
            },
            {
                "method": "generate_message",
                "num_tools": 2,
                "tool_names": ["finish_subtask", "read_workspace_file"],
            },
        ],
    )

    # History should contain the system message about entering conclusion stage
    assert any(
        msg.role == "system"
        and msg.content is not None
        and "ENTER CONCLUSION STAGE NOW" in str(msg.content)
        for msg in context.history
    )

    # Check output file
    output_path = os.path.join(
        mock_processing_context.workspace_dir, execution_subtask.output_file
    )
    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        assert final_content in f.read()


@pytest.mark.asyncio
async def test_finish_task_execution(
    mock_processing_context, basic_task, finish_subtask
):
    """Test the execution flow for a 'finish_task' subtask."""
    # Simulate reading input files
    input1_content = "Data from input 1."
    input2_content = {"key": "value from input 2"}
    input1_path_rel = finish_subtask.input_files[0]
    input2_path_rel = finish_subtask.input_files[1]
    input1_path_abs = os.path.join(
        mock_processing_context.workspace_dir, input1_path_rel
    )
    input2_path_abs = os.path.join(
        mock_processing_context.workspace_dir, input2_path_rel
    )

    os.makedirs(os.path.dirname(input1_path_abs), exist_ok=True)
    with open(input1_path_abs, "w") as f:
        f.write(input1_content)
    os.makedirs(os.path.dirname(input2_path_abs), exist_ok=True)
    with open(input2_path_abs, "w") as f:
        json.dump(input2_content, f)

    # --- Mock LLM interactions ---
    # 1. Call ReadWorkspaceFileTool for input1.txt
    # 2. Call ReadWorkspaceFileTool for input2.json
    # 3. Call finish_task with aggregated result
    final_aggregated_content = (
        f"Aggregated report:\n{input1_content}\nData: {input2_content['key']}"
    )
    metadata = {
        "title": "Final Report",
        "description": "Aggregated results",
        "sources": [f"file://{input1_path_rel}", f"file://{input2_path_rel}"],
    }

    mock_responses = [
        Message(  # Response 1: Ask to read input1.txt
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="read1",
                    name="read_workspace_file",
                    args={"path": input1_path_rel},
                )
            ],
        ),
        Message(  # Response 2: Ask to read input2.json
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="read2",
                    name="read_workspace_file",
                    args={"path": input2_path_rel},
                )
            ],
        ),
        Message(  # Response 3: Call finish_task
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call_finish_agg",
                    name="finish_task",  # Crucially, using finish_task
                    args={"result": final_aggregated_content, "metadata": metadata},
                )
            ],
        ),
    ]
    provider = MockProvider(mock_responses)

    context = SubTaskContext(
        task=basic_task,
        subtask=finish_subtask,  # The subtask configured for finish_task
        processing_context=mock_processing_context,
        tools=[],  # ReadWorkspaceFileTool is added automatically
        model="test-model",
        provider=provider,
        use_finish_task=True,  # Explicitly set for finish task
    )

    updates = [update async for update in context.execute()]

    # --- Assertions ---
    assert finish_subtask.completed
    assert any(
        isinstance(u, ToolCall) and u.name == "read_workspace_file" for u in updates
    )
    assert any(
        isinstance(u, ToolCall) and u.name == "finish_task" for u in updates
    )  # finish_task called
    assert any(
        isinstance(u, TaskUpdate) and u.event == TaskUpdateEvent.SUBTASK_COMPLETED
        for u in updates
    )

    assert len(provider.call_log) == 3  # read1, read2, finish_task

    # Check history for tool calls and results using the helper function
    expected_history = [
        {"role": "system"},  # Initial system prompt
        {"role": "user"},  # User prompt derived from subtask
        {
            "role": "assistant",
            "has_tool_calls": True,
            "tool_name": "read_workspace_file",
        },  # Call read1
        {
            "role": "tool",
            "tool_name": "read_workspace_file",
            "content_exists": True,
        },  # Result read1
        {
            "role": "assistant",
            "has_tool_calls": True,
            "tool_name": "read_workspace_file",
        },  # Call read2
        {
            "role": "tool",
            "tool_name": "read_workspace_file",
            "content_exists": True,
        },  # Result read2
        {
            "role": "assistant",
            "has_tool_calls": True,
            "tool_name": "finish_task",
        },  # Call finish_task
        {
            "role": "tool",
            "tool_name": "finish_task",
            "content_exists": True,
        },  # Result finish_task
    ]
    assert_history_structure(context.history, expected_history)

    # Check output file (markdown)
    output_path = os.path.join(
        mock_processing_context.workspace_dir, finish_subtask.output_file
    )
    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        content = f.read()
        assert "---\n" in content  # Check for YAML frontmatter
        assert f"title: {metadata['title']}" in content
        assert f"description: {metadata['description']}" in content
        assert f"sources:" in content
        assert f"- file://{input1_path_rel}" in content  # Check sources in metadata
        assert "---\n\n" in content
        assert final_aggregated_content in content  # Check main content


# --- Tests for _save_to_output_file ---


def test_save_output_content_md(mock_processing_context, execution_subtask):
    """Test saving markdown content with metadata."""
    subtask = execution_subtask.model_copy(
        update={"output_file": "output.md", "output_type": "markdown"}
    )
    content = "# Title\nSome text."
    metadata = {"title": "My Markdown", "description": "Test", "sources": ["a", "b"]}
    finish_params = {"result": content, "metadata": metadata}

    # Need a context instance to call _save_to_output_file
    provider = MockProvider([])
    context = SubTaskContext(
        Task(title="t", description="d"),
        subtask,
        mock_processing_context,
        [],
        "m",
        provider,
    )
    context._save_to_output_file(finish_params)

    output_path = os.path.join(
        mock_processing_context.workspace_dir, subtask.output_file
    )
    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        lines = f.readlines()
        assert lines[0].strip() == "---"
        # Simple check, YAML order isn't guaranteed
        assert any(f"title: {metadata['title']}" in line for line in lines)
        assert any(f"description: {metadata['description']}" in line for line in lines)
        assert any("sources:" in line for line in lines)
        assert any(f"- {s}" in line for s in metadata["sources"] for line in lines)
        assert "---\n" in lines  # Find the closing separator
        assert content in "".join(lines)


def test_save_output_content_json(mock_processing_context, execution_subtask):
    """Test saving JSON content (as dict) with metadata."""
    subtask = execution_subtask.model_copy(
        update={"output_file": "output.json", "output_type": "json"}
    )
    content = {"key": "value", "nested": [1, 2]}
    metadata = {"title": "My JSON", "description": "Test", "sources": ["c"]}
    finish_params = {"result": content, "metadata": metadata}

    provider = MockProvider([])
    context = SubTaskContext(
        Task(title="t", description="d"),
        subtask,
        mock_processing_context,
        [],
        "m",
        provider,
    )
    context._save_to_output_file(finish_params)

    output_path = os.path.join(
        mock_processing_context.workspace_dir, subtask.output_file
    )
    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        data = json.load(f)
        assert data["key"] == content["key"]
        assert data["nested"] == content["nested"]
        assert data["metadata"]["title"] == metadata["title"]
        assert data["metadata"]["sources"] == metadata["sources"]


def test_save_output_content_json_string(mock_processing_context, execution_subtask):
    """Test saving JSON content (as string) with metadata."""
    subtask = execution_subtask.model_copy(
        update={"output_file": "output.json", "output_type": "json"}
    )
    content_dict = {"key": "value_str", "nested": [3, 4]}
    content_str = json.dumps(content_dict)  # Pass as JSON string
    metadata = {"title": "My JSON String", "description": "Test Str", "sources": ["d"]}
    finish_params = {"result": content_str, "metadata": metadata}

    provider = MockProvider([])
    context = SubTaskContext(
        Task(title="t", description="d"),
        subtask,
        mock_processing_context,
        [],
        "m",
        provider,
    )
    context._save_to_output_file(finish_params)

    output_path = os.path.join(
        mock_processing_context.workspace_dir, subtask.output_file
    )
    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        data = json.load(f)
        # When saved from a string, it might parse it and add metadata,
        # or save as {"result": <string>, "metadata": ...} if parsing fails.
        # Current logic tries to parse, so it should look like the dict test.
        assert data["key"] == content_dict["key"]
        assert data["nested"] == content_dict["nested"]
        assert data["metadata"]["title"] == metadata["title"]


def test_save_output_content_yaml(mock_processing_context, execution_subtask):
    """Test saving YAML content (as dict) with metadata."""
    subtask = execution_subtask.model_copy(
        update={"output_file": "output.yaml", "output_type": "yaml"}
    )
    content = {"key": "yaml_val", "items": ["x", "y"]}
    metadata = {"title": "My YAML", "description": "Test", "sources": ["e"]}
    finish_params = {"result": content, "metadata": metadata}

    provider = MockProvider([])
    context = SubTaskContext(
        Task(title="t", description="d"),
        subtask,
        mock_processing_context,
        [],
        "m",
        provider,
    )
    context._save_to_output_file(finish_params)

    output_path = os.path.join(
        mock_processing_context.workspace_dir, subtask.output_file
    )
    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        data = yaml.safe_load(f)
        assert data["key"] == content["key"]
        assert data["items"] == content["items"]
        assert data["metadata"]["title"] == metadata["title"]


def test_save_output_file_pointer(mock_processing_context, execution_subtask):
    """Test saving output via file pointer."""
    source_rel_path = "source.data"
    source_abs_path = os.path.join(
        mock_processing_context.workspace_dir, source_rel_path
    )
    source_content = "Binary or text data from source."
    metadata = {
        "title": "Pointer Saved",
        "description": "Test Pointer",
        "sources": [f"file://{source_rel_path}"],
    }

    with open(source_abs_path, "w") as f:
        f.write(source_content)

    subtask = execution_subtask.model_copy(
        update={"output_file": "final_output.data"}
    )  # Different name
    finish_params = {
        "result": {"path": source_rel_path},
        "metadata": metadata,
    }  # File pointer

    provider = MockProvider([])
    context = SubTaskContext(
        Task(title="t", description="d"),
        subtask,
        mock_processing_context,
        [],
        "m",
        provider,
    )
    context._save_to_output_file(finish_params)

    output_path = os.path.join(
        mock_processing_context.workspace_dir, subtask.output_file
    )
    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        assert f.read() == source_content
    assert os.path.exists(source_abs_path)  # Source should still exist


def test_save_output_file_pointer_missing_source(
    mock_processing_context, execution_subtask
):
    """Test saving output via file pointer when source is missing."""
    source_rel_path = "non_existent_source.data"
    metadata = {
        "title": "Pointer Error",
        "description": "Missing Source",
        "sources": [],
    }
    subtask = execution_subtask.model_copy(
        update={"output_file": "error_output.json"}
    )  # Save error as JSON
    finish_params = {
        "result": {"path": source_rel_path},
        "metadata": metadata,
    }  # File pointer

    provider = MockProvider([])
    context = SubTaskContext(
        Task(title="t", description="d"),
        subtask,
        mock_processing_context,
        [],
        "m",
        provider,
    )
    context._save_to_output_file(
        finish_params
    )  # Should not raise an error, but write error to file

    output_path = os.path.join(
        mock_processing_context.workspace_dir, subtask.output_file
    )
    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        data = json.load(f)
        assert "error" in data
        assert "not found in workspace" in data["error"]
        assert data["metadata"] == metadata
