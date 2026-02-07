from rich.table import Table

from nodetool.metadata.types import Step, Task, ToolCall
from nodetool.ui.console import TEXTUAL_AVAILABLE, AgentConsole, _LiveContent


def test_start_and_stop_live():
    """Test starting and stopping live display."""
    console = AgentConsole(verbose=True)
    content = _LiveContent(kind="test", title="Test", body="Test content")

    # Start live (will not actually start the live app if textual is not available)
    # but the content should still be set
    console.start_live(content)

    # If textual is available, live should be active
    # If textual is not available, live_active will be False but content is still set
    assert console.current_live_content == content

    # Stop live
    console.stop_live()
    assert not console.is_live_active()
    assert console.current_live_content is None


def test_update_planning_display():
    """Test updating planning display."""
    console = AgentConsole(verbose=True)
    tree = console.create_planning_tree("Plan")
    console.start_live(tree)

    console.update_planning_display("phase1", "Running", "work")
    # Check that the planning node was recorded
    assert "phase1" in console._planning_nodes
    status, _label, content, _is_error = console._planning_nodes["phase1"]
    assert status == "Running"
    assert content == "work"

    console.update_planning_display("phase1", "Success", "done")
    status, _label, content, _is_error = console._planning_nodes["phase1"]
    assert status == "Success"
    assert content == "done"


def test_update_planning_display_without_phase():
    """Test updating planning display without showing phase name."""
    console = AgentConsole(verbose=True)
    tree = console.create_planning_tree("Plan")
    console.start_live(tree)

    console.update_planning_display("phase1", "Running", "work", show_phase=False)
    # Check that the planning node was recorded without phase label
    assert "__planner_status__" in console._planning_nodes
    status, label, content, _is_error = console._planning_nodes["__planner_status__"]
    assert status == "Running"
    assert content == "work"
    assert label == ""  # No phase label

    console.update_planning_display("phase2", "Success", "done", show_phase=False)
    status, _label, content, _is_error = console._planning_nodes["__planner_status__"]
    assert status == "Success"
    assert content == "done"


def test_create_execution_tree():
    """Test creating execution tree."""
    console = AgentConsole(verbose=True)
    sub1 = Step(
        id="sub1",
        instructions="task1",
        output_file="out1.txt",
        input_files=["in1.txt"],
        start_time=1,
        completed=True,
    )
    sub2 = Step(id="sub2", instructions="task2", output_file="out2.txt", completed=True)
    task = Task(title="t", steps=[sub1, sub2])
    call = ToolCall(step_id=sub1.id, name="tool", message="m" * 80)
    tree = console.create_execution_tree("Exec", task, [call])

    # Check that the tree was created
    assert isinstance(tree, _LiveContent)
    assert tree.kind == "execution"
    assert tree.title == "Exec"
    assert console.task == task
    assert console.tool_calls == [call]


def test_running_step_uses_spinner():
    """Test that running steps are rendered correctly."""
    console = AgentConsole(verbose=True)
    done = Step(id="s1", instructions="done step", start_time=1, completed=True)
    running = Step(id="s2", instructions="running step", start_time=1, completed=False)
    pending = Step(id="s3", instructions="pending step", completed=False)
    task = Task(title="t", steps=[done, running, pending])
    tree = console.create_execution_tree("Test", task, [])

    # Check that the tree was created and task is stored
    assert isinstance(tree, _LiveContent)
    assert console.task == task

    # Render the execution body and check the status indicators
    body = console._render_execution_body()
    assert "✓" in body  # Completed step
    assert "○" in body  # Pending step
    # Running step should have a spinner character
    assert any(spinner in body for spinner in ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])


def test_phase_logging():
    """Test that logs are stored separately for each phase."""
    console = AgentConsole(verbose=True)

    # Test setting current phase
    console.set_current_phase("Analysis")
    assert console.current_phase == "Analysis"
    assert "Analysis" in console.phase_logs

    # Test logging to current phase
    console.log_to_phase("info", "Starting analysis")
    assert len(console.get_phase_logs("Analysis")) == 1
    assert console.get_phase_logs("Analysis")[0]["message"] == "Starting analysis"
    assert console.get_phase_logs("Analysis")[0]["level"] == "info"

    # Test switching phases
    console.set_current_phase("Data Flow")
    console.log_to_phase("debug", "Processing data flow")

    # Verify logs are stored separately
    analysis_logs = console.get_phase_logs("Analysis")
    data_flow_logs = console.get_phase_logs("Data Flow")

    assert len(analysis_logs) == 1
    assert len(data_flow_logs) == 1
    assert analysis_logs[0]["message"] == "Starting analysis"
    assert data_flow_logs[0]["message"] == "Processing data flow"

    # Test logging to specific phase
    console.log_to_phase("warning", "Analysis warning", "Analysis")
    assert len(console.get_phase_logs("Analysis")) == 2
    assert console.get_phase_logs("Analysis")[1]["level"] == "warning"

    # Test get all phase logs
    all_logs = console.get_all_phase_logs()
    assert "Analysis" in all_logs
    assert "Data Flow" in all_logs
    assert len(all_logs["Analysis"]) == 2
    assert len(all_logs["Data Flow"]) == 1


def test_phase_logging_via_standard_methods():
    """Test that standard logging methods (info, debug, etc.) also log to phases."""
    console = AgentConsole(verbose=True)

    # Set a phase and use standard logging methods
    console.set_current_phase("Plan Creation")

    console.info("This is an info message")
    console.debug("This is a debug message")
    console.warning("This is a warning")
    console.error("This is an error")

    # Verify all messages were logged to the phase
    phase_logs = console.get_phase_logs("Plan Creation")
    assert len(phase_logs) == 4

    messages = [log["message"] for log in phase_logs]
    levels = [log["level"] for log in phase_logs]

    assert "This is an info message" in messages
    assert "This is a debug message" in messages
    assert "This is a warning" in messages
    assert "This is an error" in messages

    assert "info" in levels
    assert "debug" in levels
    assert "warning" in levels
    assert "error" in levels


def test_clear_phase_logs():
    """Test clearing phase logs."""
    console = AgentConsole(verbose=True)

    console.set_current_phase("Test Phase")
    console.log_to_phase("info", "Test message")

    assert len(console.get_phase_logs("Test Phase")) == 1

    # Clear specific phase
    console.clear_phase_logs("Test Phase")
    assert len(console.get_phase_logs("Test Phase")) == 0

    # Add logs to multiple phases
    console.set_current_phase("Phase 1")
    console.log_to_phase("info", "Phase 1 message")
    console.set_current_phase("Phase 2")
    console.log_to_phase("info", "Phase 2 message")

    # Clear all phases
    console.clear_phase_logs()
    assert len(console.get_all_phase_logs()) == 0
