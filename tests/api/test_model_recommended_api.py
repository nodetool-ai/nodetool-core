"""Tests for /api/models/recommended* endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from nodetool.types.model import UnifiedModel

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


def test_recommended_language_models_include_gguf_when_server_check_disabled_by_default(
    client: TestClient,
    headers: dict[str, str],
    monkeypatch,
) -> None:
    import nodetool.workflows.recommended_models as recommended_mod

    calls: list[bool] = []

    async def fake_get_recommended_language_models(
        system: str | None = None,
        check_servers: bool = True,
    ) -> list[UnifiedModel]:
        calls.append(check_servers)
        if check_servers:
            return []
        return [
            UnifiedModel(
                id="ggml-org/model:file.gguf",
                name="GGUF Model",
                type="llama_cpp_model",
                repo_id="ggml-org/model",
                path="file.gguf",
            )
        ]

    monkeypatch.setattr(
        recommended_mod,
        "get_recommended_language_models",
        fake_get_recommended_language_models,
    )

    response = client.get("/api/models/recommended/language", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert calls == [False]
    assert len(data) == 1
    assert data[0]["type"] == "llama_cpp_model"
    assert data[0]["path"].endswith(".gguf")


def test_recommended_language_models_respect_check_servers_query_param(
    client: TestClient,
    headers: dict[str, str],
    monkeypatch,
) -> None:
    import nodetool.workflows.recommended_models as recommended_mod

    calls: list[bool] = []

    async def fake_get_recommended_language_models(
        system: str | None = None,
        check_servers: bool = True,
    ) -> list[UnifiedModel]:
        calls.append(check_servers)
        return []

    monkeypatch.setattr(
        recommended_mod,
        "get_recommended_language_models",
        fake_get_recommended_language_models,
    )

    response = client.get(
        "/api/models/recommended/language?check_servers=true",
        headers=headers,
    )
    assert response.status_code == 200
    assert calls == [True]


@pytest.mark.parametrize(
    ("endpoint", "function_name"),
    [
        ("/api/models/recommended/image", "get_recommended_image_models"),
        ("/api/models/recommended/image/text-to-image", "get_recommended_text_to_image_models"),
        ("/api/models/recommended/image/image-to-image", "get_recommended_image_to_image_models"),
        ("/api/models/recommended/language", "get_recommended_language_models"),
        ("/api/models/recommended/language/text-generation", "get_recommended_language_text_generation_models"),
        ("/api/models/recommended/language/embedding", "get_recommended_language_embedding_models"),
        ("/api/models/recommended/asr", "get_recommended_asr_models"),
        ("/api/models/recommended/tts", "get_recommended_tts_models"),
        ("/api/models/recommended/video/text-to-video", "get_recommended_text_to_video_models"),
        ("/api/models/recommended/video/image-to-video", "get_recommended_image_to_video_models"),
    ],
)
def test_recommended_typed_endpoints_default_check_servers_false(
    client: TestClient,
    headers: dict[str, str],
    monkeypatch,
    endpoint: str,
    function_name: str,
) -> None:
    import nodetool.workflows.recommended_models as recommended_mod

    calls: list[bool] = []

    async def fake_recommended_fn(system: str | None = None, check_servers: bool = True) -> list[UnifiedModel]:
        calls.append(check_servers)
        return [
            UnifiedModel(
                id="org/model",
                name="Model",
                type="hf.text_generation",
                repo_id="org/model",
            )
        ]

    monkeypatch.setattr(recommended_mod, function_name, fake_recommended_fn)

    response = client.get(endpoint, headers=headers)
    assert response.status_code == 200
    assert calls == [False]


@pytest.mark.parametrize(
    ("endpoint", "function_name"),
    [
        ("/api/models/recommended/image", "get_recommended_image_models"),
        ("/api/models/recommended/image/text-to-image", "get_recommended_text_to_image_models"),
        ("/api/models/recommended/image/image-to-image", "get_recommended_image_to_image_models"),
        ("/api/models/recommended/language", "get_recommended_language_models"),
        ("/api/models/recommended/language/text-generation", "get_recommended_language_text_generation_models"),
        ("/api/models/recommended/language/embedding", "get_recommended_language_embedding_models"),
        ("/api/models/recommended/asr", "get_recommended_asr_models"),
        ("/api/models/recommended/tts", "get_recommended_tts_models"),
        ("/api/models/recommended/video/text-to-video", "get_recommended_text_to_video_models"),
        ("/api/models/recommended/video/image-to-video", "get_recommended_image_to_video_models"),
    ],
)
def test_recommended_typed_endpoints_respect_check_servers_true(
    client: TestClient,
    headers: dict[str, str],
    monkeypatch,
    endpoint: str,
    function_name: str,
) -> None:
    import nodetool.workflows.recommended_models as recommended_mod

    calls: list[bool] = []

    async def fake_recommended_fn(system: str | None = None, check_servers: bool = True) -> list[UnifiedModel]:
        calls.append(check_servers)
        return []

    monkeypatch.setattr(recommended_mod, function_name, fake_recommended_fn)

    response = client.get(f"{endpoint}?check_servers=true", headers=headers)
    assert response.status_code == 200
    assert calls == [True]


def test_recommended_all_endpoint_respects_check_servers_query_param(
    client: TestClient,
    headers: dict[str, str],
    monkeypatch,
) -> None:
    import nodetool.api.model as model_api

    calls: list[bool] = []

    async def fake_recommended_models(_user: str, check_servers: bool = False) -> list[UnifiedModel]:
        calls.append(check_servers)
        return []

    monkeypatch.setattr(model_api, "recommended_models", fake_recommended_models)

    response = client.get("/api/models/recommended?check_servers=true", headers=headers)
    assert response.status_code == 200
    assert calls == [True]
