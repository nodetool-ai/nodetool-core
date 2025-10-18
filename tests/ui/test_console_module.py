from rich.table import Table

from nodetool.ui.console import AgentConsole
from nodetool.metadata.types import Task, SubTask, ToolCall


class FakeLive:
    def __init__(
        self, content, console=None, refresh_per_second=4, vertical_overflow="visible"
    ):
        self.content = content
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def update(self, content):
        self.content = content

    @property
    def is_started(self):
        return self.started


def test_start_and_stop_live(monkeypatch):
    console = AgentConsole(verbose=True)
    monkeypatch.setattr("nodetool.ui.console.Live", FakeLive)
    table = Table()
    console.start_live(table)
    assert isinstance(console.live, FakeLive)
    assert console.live.is_started
    assert console.current_table is table
    console.stop_live()
    assert console.live is None
    assert console.current_table is None


def test_update_planning_display(monkeypatch):
    console = AgentConsole(verbose=True)
    monkeypatch.setattr("nodetool.ui.console.Live", FakeLive)
    tree = console.create_planning_tree("Plan")
    console.start_live(tree)
    console.update_planning_display("phase1", "Running", "work")
    node = console.phase_nodes["phase1"]
    assert "phase1" in node.label
    console.update_planning_display("phase1", "Success", "done")
    assert "Success" in node.label


def test_create_execution_table(monkeypatch):
    console = AgentConsole(verbose=True)
    sub1 = SubTask(
        id="sub1",
        content="task1",
        output_file="out1.txt",
        input_files=["in1.txt"],
        start_time=1,
        completed=True,
    )
    sub2 = SubTask(id="sub2", content="task2", output_file="out2.txt", completed=True)
    task = Task(title="t", subtasks=[sub1, sub2])
    call = ToolCall(subtask_id=sub1.id, name="tool", message="m" * 80)
    tree = console.create_execution_tree("Exec", task, [call])
    node1 = console.subtask_nodes[sub1.id]
    assert "task1" in node1.label
    # Check that tool calls are displayed with truncated message
    assert len(node1.children) > 0  # Should have tool calls
    tool_section = node1.children[0]  # Should be the "Tools" section
    assert len(tool_section.children) > 0  # Should have at least one tool call
    tool_call_label = str(tool_section.children[0].label)
    assert "..." in tool_call_label  # message truncated
    assert len(tree.children) == 2


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
