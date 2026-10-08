"""Cached text-to-video repos are recognized by their diffusers pipeline class."""

import json

import pytest

from nodetool.integrations.huggingface import huggingface_models


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "class_name, listed",
    [
        ("HunyuanVideo15Pipeline", True),
        ("HunyuanVideo15ImageToVideoPipeline", True),
        ("WanPipeline", True),
        ("FluxPipeline", False),
    ],
)
async def test_video_repos_are_listed_by_pipeline_class(
    monkeypatch, tmp_path, class_name, listed
):
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": class_name}))

    async def cached_files():
        yield "org/repo", tmp_path, tmp_path, ["model_index.json"]

    monkeypatch.setattr(huggingface_models, "iter_cached_model_files", cached_files)

    models = await huggingface_models.get_text_to_video_models_from_hf_cache()

    assert [m.id for m in models] == (["org/repo"] if listed else [])
