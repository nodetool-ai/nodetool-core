import asyncio
import json
import os
from unittest.mock import MagicMock
from nodetool.messaging.regular_chat_processor import RegularChatProcessor, detect_mime_type
from nodetool.providers.base import BaseProvider
from nodetool.metadata.types import AssetRef

async def test_fix():
    print("Testing detect_mime_type...")
    assert detect_mime_type(b'\x89PNG\r\n\x1a\n') == 'image/png'
    assert detect_mime_type(b'\xff\xd8') == 'image/jpeg'
    assert detect_mime_type(b'ID3') == 'audio/mpeg'
    assert detect_mime_type(b'some junk') == 'application/octet-stream'
    print("detect_mime_type passed.")

    # Mock provider
    provider = MagicMock(spec=BaseProvider)
    processor = RegularChatProcessor(provider)

    # Test _process_tool_result with bytes
    print("Testing _process_tool_result with bytes...")
    
    # Payload with nested bytes
    payload = {
        "message": "Here is an image",
        "image_data": b'\x89PNG\r\n\x1a\nfakeimagecontent',
        "audio_data": b'ID3fakeaudiocontent',
        "plain_list": [1, 2, b'simplebytes']
    }

    result = await processor._process_tool_result(payload)
    
    print("Result:", json.dumps(result, indent=2))
    
    assert result["message"] == "Here is an image"
    assert result["image_data"]["type"] == "image"
    assert result["image_data"]["uri"].startswith("file://")
    assert result["image_data"]["uri"].endswith(".png")
    
    assert result["audio_data"]["type"] == "audio"
    assert result["audio_data"]["uri"].startswith("file://")
    
    assert result["plain_list"][2]["type"] == "asset" # simplebytes -> application/octet-stream -> AssetRef -> type="asset"
    
    print("Verification successful!")

if __name__ == "__main__":
    asyncio.run(test_fix())
