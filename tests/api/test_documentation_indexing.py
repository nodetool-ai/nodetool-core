import os
from unittest.mock import patch

from nodetool.api.server import create_app
from nodetool.common.environment import Environment


def test_create_app_starts_indexing_process():
    # ensure production environment
    prev_env = Environment.get_env()
    Environment.set_env("production")
    try:
        with patch("nodetool.api.server.Process") as mock_process:
            create_app()
            mock_process.assert_called_once()
            mock_process.return_value.start.assert_called_once()
    finally:
        Environment.set_env(prev_env or "development")

