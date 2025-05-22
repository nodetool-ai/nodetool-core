import pytest
from nodetool.api import utils


@pytest.mark.asyncio
async def test_current_user_local():
    user = await utils.current_user()
    assert user == "1"
