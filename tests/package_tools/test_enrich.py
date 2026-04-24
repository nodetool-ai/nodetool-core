"""Enrich tests (mocked HF fetch)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from nodetool.metadata.node_metadata import NodeMetadata
from nodetool.package_tools.enrich import enrich_nodes_with_model_info


class _FakeModelInfo:
    def __init__(self) -> None:
        self.pipeline_tag = "text-generation"
        self.tags = ["foo", "bar"]
        self.downloads = 123
        self.likes = 45
        self.trending_score = 7.2
        self.siblings = []
        self.cardData = None


def _make_node_with_repo(repo_id: str) -> NodeMetadata:
    from nodetool.types.model import UnifiedModel

    model = UnifiedModel(
        id=repo_id,
        repo_id=repo_id,
        type=None,
        name=repo_id,
        path=None,
        downloaded=False,
    )
    return NodeMetadata(
        title="T",
        description="D",
        namespace="ns",
        node_type="ns.T",
        recommended_models=[model],
    )


@pytest.mark.asyncio
async def test_enrich_populates_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch(_model_id: str) -> Any:
        return _FakeModelInfo()

    def fake_has_model_index(_info: Any) -> bool:
        return False

    def fake_type(_repo_map: Any, _repo_id: str, _info: Any) -> str | None:
        return "language_model"

    import nodetool.integrations.huggingface.huggingface_models as hm

    monkeypatch.setattr(hm, "fetch_model_info", fake_fetch)
    monkeypatch.setattr(hm, "has_model_index", fake_has_model_index)
    monkeypatch.setattr(hm, "model_type_from_model_info", fake_type)

    node = _make_node_with_repo("owner/repo")
    ok, failed = await enrich_nodes_with_model_info([node])
    assert (ok, failed) == (1, 0)

    m = node.recommended_models[0]
    assert m.pipeline_tag == "text-generation"
    assert m.tags == ["foo", "bar"]
    assert m.downloads == 123
    assert m.likes == 45


@pytest.mark.asyncio
async def test_enrich_no_models_is_noop() -> None:
    node = NodeMetadata(title="T", description="D", namespace="ns", node_type="ns.T")
    ok, failed = await enrich_nodes_with_model_info([node])
    assert (ok, failed) == (0, 0)


def test_enrich_sync_wrapper() -> None:
    node = NodeMetadata(title="T", description="D", namespace="ns", node_type="ns.T")
    ok, failed = asyncio.run(enrich_nodes_with_model_info([node]))
    assert (ok, failed) == (0, 0)
