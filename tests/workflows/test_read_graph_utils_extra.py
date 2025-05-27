from nodetool.workflows import read_graph


def test_is_comfy_widget():
    assert read_graph.is_comfy_widget([1, 2])
    assert read_graph.is_comfy_widget("INT")
    assert not read_graph.is_comfy_widget("custom")


def test_get_x_y_from_pos():
    assert read_graph.get_x_y_from_pos([4, 10]) == (2, 5)
    assert read_graph.get_x_y_from_pos({"0": 8}) == (4, 4)


def test_convert_graph_basic(monkeypatch):
    inp = {
        "nodes": [
            {"id": 1, "type": "A", "widgets_values": [5], "pos": [0, 0]},
            {
                "id": 2,
                "type": "B",
                "inputs": [{"name": "x", "link": 0}],
                "widgets_values": [1, 2],
            },
        ],
        "links": [[0, 1, "out"]],
    }

    monkeypatch.setattr(
        read_graph, "get_widget_names", lambda t: ["a"] if t == "A" else ["x", "y"]
    )
    out = read_graph.convert_graph(inp)
    assert out["1"]["data"]["a"] == 5
    assert out["2"]["data"]["x"] == ["1", "out"]
