"""
Tests for the ComfyUI graph converter (``comfy_graph_converter.py``).

Covers:
- Basic graph→prompt conversion
- Handle-based slot resolution (output_N, input_N)
- Metadata-based input/output name mapping
- Fallback slot index behavior
- Internal data key filtering
- has_comfy_nodes detection
- Reverse conversion (prompt→graph)
"""

import pytest

from nodetool.types.api_graph import Edge, Graph, Node
from nodetool.workflows.comfy_graph_converter import (
    graph_to_prompt,
    has_comfy_nodes,
    prompt_to_graph,
)

# ---------------------------------------------------------------------------
# graph_to_prompt
# ---------------------------------------------------------------------------


class TestGraphToPrompt:
    """Test suite for NodeTool Graph → ComfyUI prompt conversion."""

    def test_basic_conversion(self):
        """Nodes with comfy. prefix are included; non-comfy nodes excluded."""
        graph = Graph(
            nodes=[
                Node(id="n1", type="comfy.KSampler", data={"steps": 20, "cfg": 8}),
                Node(id="n2", type="nodetool.input.FloatInput", data={"value": 1.0}),
            ],
            edges=[],
        )
        prompt = graph_to_prompt(graph)
        assert "n1" in prompt
        assert "n2" not in prompt
        assert prompt["n1"]["class_type"] == "KSampler"
        assert prompt["n1"]["inputs"]["steps"] == 20
        assert prompt["n1"]["inputs"]["cfg"] == 8

    def test_edge_conversion_with_output_n_handle(self):
        """Edges with output_N sourceHandle should resolve to numeric index N."""
        graph = Graph(
            nodes=[
                Node(id="loader", type="comfy.CheckpointLoaderSimple", data={"ckpt_name": "model.safetensors"}),
                Node(id="sampler", type="comfy.KSampler", data={"steps": 20}),
            ],
            edges=[
                Edge(source="loader", sourceHandle="output_0", target="sampler", targetHandle="model"),
            ],
        )
        prompt = graph_to_prompt(graph)
        assert prompt["sampler"]["inputs"]["model"] == ["loader", 0]

    def test_edge_conversion_with_semantic_handle(self):
        """Edges with semantic source handle fall back to index 0 when no metadata."""
        graph = Graph(
            nodes=[
                Node(id="loader", type="comfy.CheckpointLoaderSimple", data={}),
                Node(id="sampler", type="comfy.KSampler", data={}),
            ],
            edges=[
                Edge(source="loader", sourceHandle="model", target="sampler", targetHandle="model"),
            ],
        )
        prompt = graph_to_prompt(graph)
        # Without metadata, semantic handles fall back to 0
        assert prompt["sampler"]["inputs"]["model"] == ["loader", 0]

    def test_edge_with_metadata_output_resolution(self):
        """Output slot resolved via _comfy_metadata.outputs name lookup."""
        graph = Graph(
            nodes=[
                Node(
                    id="loader",
                    type="comfy.CheckpointLoaderSimple",
                    data={
                        "ckpt_name": "model.safetensors",
                        "_comfy_metadata": {
                            "outputs": [
                                {"name": "MODEL"},
                                {"name": "CLIP"},
                                {"name": "VAE"},
                            ]
                        },
                    },
                ),
                Node(id="encode", type="comfy.CLIPTextEncode", data={"text": "hello"}),
            ],
            edges=[
                Edge(source="loader", sourceHandle="CLIP", target="encode", targetHandle="clip"),
            ],
        )
        prompt = graph_to_prompt(graph)
        assert prompt["encode"]["inputs"]["clip"] == ["loader", 1]

    def test_edge_with_metadata_input_name_resolution(self):
        """Input name resolved via _comfy_metadata.inputs for index-style handles."""
        graph = Graph(
            nodes=[
                Node(id="src", type="comfy.SomeNode", data={}),
                Node(
                    id="dst",
                    type="comfy.OtherNode",
                    data={
                        "_comfy_metadata": {
                            "inputs": [
                                {"name": "alpha"},
                                {"name": "beta"},
                                {"name": "gamma"},
                            ]
                        }
                    },
                ),
            ],
            edges=[
                Edge(source="src", sourceHandle="output_0", target="dst", targetHandle="input_2"),
            ],
        )
        prompt = graph_to_prompt(graph)
        # input_2 should resolve to "gamma"
        assert "gamma" in prompt["dst"]["inputs"]
        assert prompt["dst"]["inputs"]["gamma"] == ["src", 0]

    def test_internal_metadata_keys_excluded(self):
        """Keys like _comfy_metadata should not appear in prompt inputs."""
        graph = Graph(
            nodes=[
                Node(
                    id="n1",
                    type="comfy.KSampler",
                    data={
                        "steps": 20,
                        "_comfy_metadata": {"outputs": [{"name": "LATENT"}]},
                    },
                ),
            ],
            edges=[],
        )
        prompt = graph_to_prompt(graph)
        assert "_comfy_metadata" not in prompt["n1"]["inputs"]
        assert prompt["n1"]["inputs"]["steps"] == 20

    def test_multiple_edges_to_same_node(self):
        """Multiple edges targeting different inputs on the same node."""
        graph = Graph(
            nodes=[
                Node(id="loader", type="comfy.CheckpointLoaderSimple", data={}),
                Node(id="encode", type="comfy.CLIPTextEncode", data={"text": "test"}),
                Node(id="sampler", type="comfy.KSampler", data={"steps": 20}),
            ],
            edges=[
                Edge(source="loader", sourceHandle="output_0", target="sampler", targetHandle="model"),
                Edge(source="encode", sourceHandle="output_0", target="sampler", targetHandle="positive"),
            ],
        )
        prompt = graph_to_prompt(graph)
        assert prompt["sampler"]["inputs"]["model"] == ["loader", 0]
        assert prompt["sampler"]["inputs"]["positive"] == ["encode", 0]

    def test_empty_graph(self):
        """Empty graph produces empty prompt."""
        prompt = graph_to_prompt(Graph(nodes=[], edges=[]))
        assert prompt == {}

    def test_no_comfy_nodes(self):
        """Graph with no comfy nodes produces empty prompt."""
        graph = Graph(
            nodes=[
                Node(id="n1", type="nodetool.math.Add", data={"a": 1, "b": 2}),
            ],
            edges=[],
        )
        prompt = graph_to_prompt(graph)
        assert prompt == {}

    def test_edge_overrides_data_value(self):
        """Edge connection should override any static value in data for the same key."""
        graph = Graph(
            nodes=[
                Node(id="src", type="comfy.Source", data={}),
                Node(id="dst", type="comfy.Dest", data={"model": "static_value"}),
            ],
            edges=[
                Edge(source="src", sourceHandle="output_0", target="dst", targetHandle="model"),
            ],
        )
        prompt = graph_to_prompt(graph)
        assert prompt["dst"]["inputs"]["model"] == ["src", 0]


# ---------------------------------------------------------------------------
# has_comfy_nodes
# ---------------------------------------------------------------------------


class TestHasComfyNodes:
    def test_with_comfy_nodes(self):
        graph = Graph(
            nodes=[Node(id="n1", type="comfy.KSampler", data={})],
            edges=[],
        )
        assert has_comfy_nodes(graph) is True

    def test_without_comfy_nodes(self):
        graph = Graph(
            nodes=[Node(id="n1", type="nodetool.math.Add", data={})],
            edges=[],
        )
        assert has_comfy_nodes(graph) is False

    def test_empty_graph(self):
        assert has_comfy_nodes(Graph(nodes=[], edges=[])) is False

    def test_mixed_nodes(self):
        graph = Graph(
            nodes=[
                Node(id="n1", type="nodetool.input.FloatInput", data={}),
                Node(id="n2", type="comfy.SaveImage", data={}),
            ],
            edges=[],
        )
        assert has_comfy_nodes(graph) is True


# ---------------------------------------------------------------------------
# prompt_to_graph (reverse conversion)
# ---------------------------------------------------------------------------


class TestPromptToGraph:
    def test_basic_reverse(self):
        """Simple prompt converts back to a graph with comfy. prefixed types."""
        prompt = {
            "1": {
                "class_type": "KSampler",
                "inputs": {"steps": 20, "cfg": 8},
            }
        }
        graph = prompt_to_graph(prompt)
        assert len(graph.nodes) == 1
        assert graph.nodes[0].id == "1"
        assert graph.nodes[0].type == "comfy.KSampler"
        assert graph.nodes[0].data == {"steps": 20, "cfg": 8}
        assert len(graph.edges) == 0

    def test_connection_tuples_become_edges(self):
        """Connection tuples [source_id, index] become edges."""
        prompt = {
            "loader": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "model.safetensors"},
            },
            "sampler": {
                "class_type": "KSampler",
                "inputs": {
                    "steps": 20,
                    "model": ["loader", 0],
                },
            },
        }
        graph = prompt_to_graph(prompt)
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1

        edge = graph.edges[0]
        assert edge.source == "loader"
        assert edge.sourceHandle == "output_0"
        assert edge.target == "sampler"
        assert edge.targetHandle == "model"

        # The connection tuple should NOT appear in data
        sampler_node = next(n for n in graph.nodes if n.id == "sampler")
        assert "model" not in sampler_node.data
        assert sampler_node.data["steps"] == 20

    def test_roundtrip_consistency(self):
        """Converting graph→prompt→graph preserves structure."""
        original = Graph(
            nodes=[
                Node(id="n1", type="comfy.CheckpointLoaderSimple", data={"ckpt_name": "model.safetensors"}),
                Node(id="n2", type="comfy.KSampler", data={"steps": 20}),
            ],
            edges=[
                Edge(source="n1", sourceHandle="output_0", target="n2", targetHandle="model"),
            ],
        )
        prompt = graph_to_prompt(original)
        roundtripped = prompt_to_graph(prompt)

        assert len(roundtripped.nodes) == 2
        assert len(roundtripped.edges) == 1

        n2 = next(n for n in roundtripped.nodes if n.id == "n2")
        assert n2.type == "comfy.KSampler"
        assert n2.data["steps"] == 20

        edge = roundtripped.edges[0]
        assert edge.source == "n1"
        assert edge.target == "n2"
        assert edge.targetHandle == "model"


if __name__ == "__main__":
    pytest.main([__file__])
