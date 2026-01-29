from unittest.mock import MagicMock

import pytest

from nodetool.api import utils


@pytest.mark.asyncio
async def test_current_user_local():
    mock_request = MagicMock()
    mock_request.state.user_id = None
    mock_request.headers = {}
    user = await utils.current_user(mock_request)
    assert user == "1"
