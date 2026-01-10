import pytest

from nodetool.dsl.graph import graph_result, run_graph
from nodetool.types.api_graph import Graph
from nodetool.workflows.processing_context import AssetOutputMode
from nodetool.workflows.types import OutputUpdate


@pytest.mark.asyncio
async def test_run_graph_passes_asset_output_mode(monkeypatch):
    captured_context = None

    async def fake_run_workflow(request, context=None):
        nonlocal captured_context
        captured_context = context
        yield OutputUpdate(
            node_id="node-1",
            node_name="Result",
            output_name="value",
            value="ok",
            output_type="text",
        )

    monkeypatch.setattr("nodetool.dsl.graph.run_workflow", fake_run_workflow)

    graph = Graph(nodes=[], edges=[])
    result = await run_graph(
        graph,
        user_id="u1",
        auth_token="token",
        asset_output_mode=AssetOutputMode.DATA_URI,
    )

    assert result == {"Result": "ok"}
    assert captured_context is not None
    assert captured_context.asset_output_mode == AssetOutputMode.DATA_URI
    assert captured_context.user_id == "u1"
    assert captured_context.auth_token == "token"


@pytest.mark.asyncio
async def test_run_graph_without_mode_uses_default(monkeypatch):
    captured_context = object()

    async def fake_run_workflow(request, context=None):
        nonlocal captured_context
        captured_context = context
        yield OutputUpdate(
            node_id="node-1",
            node_name="Result",
            output_name="value",
            value="ok",
            output_type="text",
        )

    monkeypatch.setattr("nodetool.dsl.graph.run_workflow", fake_run_workflow)

    graph = Graph(nodes=[], edges=[])
    await run_graph(graph)

    # When no mode is supplied run_graph should defer to run_workflow defaults.
    assert captured_context is None


@pytest.mark.asyncio
async def test_graph_result_allows_asset_mode(monkeypatch):
    async def fake_run_graph_async(g, **kwargs):
        assert kwargs.get("asset_output_mode") == AssetOutputMode.WORKSPACE
        return {"Output": {"type": "image", "path": "/tmp/example.png"}}

    monkeypatch.setattr("nodetool.dsl.graph.run_graph_async", fake_run_graph_async)

    sentinel = Graph(nodes=[], edges=[])

    def fake_graph(node):
        assert node == "example"
        return sentinel

    monkeypatch.setattr("nodetool.dsl.graph.graph", fake_graph)

    result = await graph_result("example", asset_output_mode=AssetOutputMode.WORKSPACE)
    assert result == {"Output": {"type": "image", "path": "/tmp/example.png"}}
