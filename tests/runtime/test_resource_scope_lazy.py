"""Regression tests for lazy resource inheritance and post-exit safety.

``ResourceScope.__init__`` used to eagerly resolve the parent's asset storage
(which mkdirs the asset folder) and HTTP client, so nested scopes paid for
resources they never used. ``__aexit__`` also swallowed contextvar reset
failures, leaving an exited scope reachable and willing to build a brand new
owned ``httpx.AsyncClient`` that nobody would ever close.
"""

import contextvars

import pytest

from nodetool.runtime.resources import ResourceScope, _current_scope


@pytest.mark.no_setup
async def test_nested_scope_does_not_eagerly_build_resources(monkeypatch):
    import nodetool.storage.file_storage as file_storage_module

    def _boom(*args, **kwargs):
        raise AssertionError("FileStorage must not be constructed eagerly")

    async with ResourceScope() as parent:
        parent._asset_storage = object()
        parent._temp_storage = object()

        monkeypatch.setattr(file_storage_module, "FileStorage", _boom)

        child = ResourceScope()
        assert child._asset_storage is None
        assert child._temp_storage is None
        assert child._memory_uri_cache is None
        assert child._http_client is None

        # Resolved from the parent only when actually requested.
        assert child.get_asset_storage() is parent._asset_storage
        assert child.get_temp_storage() is parent._temp_storage
        assert child.get_memory_uri_cache() is parent.get_memory_uri_cache()
        assert child._owns_http_client is False


@pytest.mark.no_setup
async def test_nested_scope_shares_parent_http_client():
    async with ResourceScope() as parent:
        child = ResourceScope()
        assert child.get_http_client() is parent.get_http_client()
        assert child._owns_http_client is False
        assert parent._owns_http_client is True


@pytest.mark.no_setup
async def test_exited_scope_refuses_to_build_new_resources():
    scope = ResourceScope()
    async with scope:
        pass

    for getter in (
        scope.get_http_client,
        scope.get_asset_storage,
        scope.get_temp_storage,
        scope.get_memory_uri_cache,
    ):
        with pytest.raises(RuntimeError, match="already exited"):
            getter()

    assert scope._owns_http_client is False


@pytest.mark.no_setup
async def test_mismatched_context_reset_is_logged_not_silent(caplog):
    scope = ResourceScope()
    await scope.__aenter__()
    # Simulate a token created in a different context (an async generator
    # resumed by a different task).
    scope._token = contextvars.copy_context().run(lambda: _current_scope.set(scope))

    with caplog.at_level("WARNING"):
        await scope.__aexit__(None, None, None)

    assert any("different context" in record.message for record in caplog.records)
    _current_scope.set(None)
