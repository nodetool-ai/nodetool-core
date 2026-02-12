# Parallel Test Worker Crash Issue

**Insight**: Some tests fail when run with pytest-xdist parallel execution but pass individually

**Rationale**: The tests `test_admin_endpoints_require_token_in_production_when_configured` and `test_terminal_ws_echoes_input` experience worker crashes during parallel execution. This is likely due to:
1. Shared state/资源 conflict between workers
2. Async event loop issues across worker processes
3. Database file locking in SQLite tests

**Example**:
```bash
# Both fail with parallel execution (pytest -n auto)
pytest tests/api/test_server_e2e.py::TestAdminAuthMiddleware::test_admin_endpoints_require_token_in_production_when_configured \
       tests/api/test_terminal_websocket.py::test_terminal_ws_echoes_input

# Both pass when run sequentially
pytest tests/api/test_server_e2e.py::TestAdminAuthMiddleware::test_admin_endpoints_require_token_in_production_when_configured \
       tests/api/test_terminal_websocket.py::test_terminal_ws_echoes_input -v
```

**Impact**: 2 tests show flaky behavior in CI/CD but tests are valid when run in isolation

**Files**: `tests/api/test_server_e2e.py`, `tests/api/test_terminal_websocket.py`

**Date**: 2026-02-12
