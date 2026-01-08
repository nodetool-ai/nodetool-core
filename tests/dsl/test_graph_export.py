from nodetool.dsl.export import graph_to_dsl_py
from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.types.api_graph import Edge as ApiEdge
from nodetool.types.api_graph import Graph as ApiGraph
from nodetool.types.api_graph import Node as ApiNode


def test_export_minimal_single_node():
    n1 = ApiNode(id="1", type="apple.notes.ReadNotes", data={})
    g = ApiGraph(nodes=[n1], edges=[])

    code = graph_to_dsl_py(g)

    # Imports
    assert "from nodetool.dsl.apple.notes import ReadNotes" in code
    # Instantiation and graph assembly
    assert "read_notes_1 = ReadNotes(" in code
    assert "workflow = graph(read_notes_1)" in code


def test_export_with_connection_and_props():
    n1 = ApiNode(id="1", type="apple.notes.ReadNotes", data={})
    n2 = ApiNode(id="2", type="apple.notes.CreateNote", data={"title": "T"})
    e = ApiEdge(id="e1", source="1", sourceHandle="output", target="2", targetHandle="content")
    g = ApiGraph(nodes=[n1, n2], edges=[e])

    code = graph_to_dsl_py(g)

    # Two imports grouped by module
    assert (
        "from nodetool.dsl.apple.notes import CreateNote, ReadNotes" in code
        or "from nodetool.dsl.apple.notes import ReadNotes, CreateNote" in code
    )

    # ReadNotes defined before CreateNote (topological)
    first_idx = code.find("read_notes_1 = ReadNotes(")
    second_idx = code.find("create_note_1 = CreateNote(")
    assert first_idx != -1 and second_idx != -1 and first_idx < second_idx

    # Connected handle and preserved literal property
    assert "create_note_1 = CreateNote(title='T', instructions=read_notes_1.output)" in code
    assert "workflow = graph(read_notes_1, create_note_1)" in code


def test_export_dynamic_outputs_and_properties():
    n = ApiNode(
        id="1",
        type="apple.notes.CreateNote",
        data={"title": "Draft"},
        dynamic_properties={"tag": "personal"},
        dynamic_outputs={
            "branch_a": TypeMetadata(type="str"),
            "branch_b": TypeMetadata(type="int"),
        },
    )
    g = ApiGraph(nodes=[n], edges=[])

    code = graph_to_dsl_py(g)

    # dynamic_outputs rendered as dict literals
    assert "dynamic_outputs={'branch_a': {'type': 'str'}, 'branch_b': {'type': 'int'}}" in code
    # dynamic property included
    assert "tag='personal'" in code
    # regular prop included
    assert "title='Draft'" in code
