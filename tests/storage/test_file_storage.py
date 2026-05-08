import pytest
import os
from pathlib import Path
from io import BytesIO
from nodetool.storage.file_storage import FileStorage

@pytest.mark.asyncio
async def test_file_storage_path_traversal(tmp_path):
    storage = FileStorage(str(tmp_path))
    with pytest.raises(ValueError, match="Path traversal attempt detected"):
        await storage.upload("../evil.txt", BytesIO(b"evil"))
