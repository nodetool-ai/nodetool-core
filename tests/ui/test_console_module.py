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
    tree = console.create_planning_table("Plan")
    console.start_live(tree)
    console.update_planning_display("phase1", "Running", "work")
    node = console.phase_nodes["phase1"]
    assert "phase1" in node.label
    console.update_planning_display("phase1", "Success", "done")
    assert "Success" in node.label


def test_create_execution_table(monkeypatch):
    console = AgentConsole(verbose=True)
    sub1 = SubTask(
        content="task1", output_file="out1.txt", input_files=["in1.txt"], start_time=1
    )
    sub2 = SubTask(content="task2", output_file="out2.txt", completed=True)
    task = Task(title="t", subtasks=[sub1, sub2])
    call = ToolCall(subtask_id=sub1.id, name="tool", message="m" * 60)
    tree = console.create_execution_table("Exec", task, [call])
    node1 = console.subtask_nodes[sub1.id]
    assert "task1" in node1.label
    assert "..." in node1.label  # message truncated
    assert len(tree.children) == 2
