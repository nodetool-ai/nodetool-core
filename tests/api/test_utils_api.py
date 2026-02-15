from unittest.mock import MagicMock, patch

import pytest

from nodetool.api import utils


@pytest.mark.asyncio
async def test_current_user_local():
    mock_request = MagicMock()
    mock_request.state.user_id = None
    mock_request.headers = {}
    with patch.object(utils.Environment, "enforce_auth", return_value=False):
        user = await utils.current_user(mock_request)
    assert user == "1"
