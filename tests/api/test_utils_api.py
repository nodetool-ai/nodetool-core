import pytest

from nodetool.api import utils


@pytest.mark.asyncio
async def test_current_user_local():
    user = await utils.get_current_user_direct()
    assert user == "1"
