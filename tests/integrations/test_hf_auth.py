import pytest

from nodetool.integrations.huggingface import hf_auth, huggingface_models


@pytest.mark.asyncio
async def test_hf_token_uses_live_environment_on_rotation(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "token-one")
    assert await hf_auth.get_hf_token("user") == "token-one"

    monkeypatch.setenv("HF_TOKEN", "token-two")
    assert await hf_auth.get_hf_token("user") == "token-two"
    assert huggingface_models.get_hf_token is hf_auth.get_hf_token


@pytest.mark.asyncio
async def test_hf_token_falls_back_to_user_secret(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    async def fake_get_secret(key: str, user_id: str) -> str:
        assert key == "HF_TOKEN"
        assert user_id == "user"
        return "secret-token"

    monkeypatch.setattr(hf_auth, "get_secret", fake_get_secret)
    assert await hf_auth.get_hf_token("user") == "secret-token"
