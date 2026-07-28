"""Regression tests for get_value's handling of DEFAULT_ENV keys mapped to None.

``default_env.get(key, default)`` returned ``None`` for the many keys that are
explicitly ``None`` in DEFAULT_ENV, silently shadowing a caller-supplied
default.
"""

import pytest

from nodetool.config.settings import get_value

DEFAULT_ENV_SAMPLE = {
    "ASSET_DOMAIN": None,
    "ASSET_BUCKET": "images",
}


def test_caller_default_wins_over_none_valued_default_env(monkeypatch):
    monkeypatch.delenv("ASSET_DOMAIN", raising=False)
    assert get_value("ASSET_DOMAIN", {}, DEFAULT_ENV_SAMPLE, "cdn.example.com") == "cdn.example.com"


def test_none_default_env_without_caller_default_is_none(monkeypatch):
    monkeypatch.delenv("ASSET_DOMAIN", raising=False)
    assert get_value("ASSET_DOMAIN", {}, DEFAULT_ENV_SAMPLE, None) is None


def test_non_none_default_env_still_wins_over_caller_default(monkeypatch):
    monkeypatch.delenv("ASSET_BUCKET", raising=False)
    assert get_value("ASSET_BUCKET", {}, DEFAULT_ENV_SAMPLE, "other") == "images"


def test_settings_and_env_still_take_priority(monkeypatch):
    monkeypatch.delenv("ASSET_DOMAIN", raising=False)
    assert get_value("ASSET_DOMAIN", {"ASSET_DOMAIN": "from-settings"}, DEFAULT_ENV_SAMPLE, "d") == "from-settings"

    monkeypatch.setenv("ASSET_DOMAIN", "from-env")
    assert get_value("ASSET_DOMAIN", {"ASSET_DOMAIN": "from-settings"}, DEFAULT_ENV_SAMPLE, "d") == "from-env"


def test_unknown_key_without_default_raises(monkeypatch):
    monkeypatch.delenv("TOTALLY_UNKNOWN_SETTING", raising=False)
    with pytest.raises(Exception):
        get_value("TOTALLY_UNKNOWN_SETTING", {}, DEFAULT_ENV_SAMPLE)
