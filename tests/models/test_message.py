import pytest

from nodetool.models.message import Message


@pytest.mark.asyncio
async def test_find_message(user_id: str):
    message = await Message.create(
        user_id=user_id,
        thread_id="th1",
    )

    found_message = await Message.get(message.id)

    if found_message:
        assert message.id == found_message.id
    else:
        pytest.fail("Message not found")

    # Test finding a message that does not exist in the database
    not_found_message = await Message.get("invalid_id")
    assert not_found_message is None


@pytest.mark.asyncio
async def test_paginate_messages(user_id: str):
    await Message.create(user_id=user_id, thread_id="th1")

    messages, last_key = await Message.paginate(thread_id="th1", limit=10)
    assert len(messages) > 0
    assert last_key == ""


@pytest.mark.asyncio
async def test_create_message(user_id: str):
    message = await Message.create(
        user_id=user_id,
        thread_id="th1",
    )

    assert await Message.get(message.id) is not None


@pytest.mark.asyncio
async def test_create_message_image_content(user_id: str):
    message = await Message.create(
        user_id=user_id,
        thread_id="th1",
        instructions=[
            {
                "type": "image_url",
                "image": {"type": "image", "uri": "https://example.com/image.jpg"},
            }
        ],
    )

    assert await Message.get(message.id) is not None
    assert message.content is not None
    assert isinstance(message.content, list)
    assert len(message.content) == 1
    assert message.content[0].type == "image_url"
    assert message.content[0].image.uri == "https://example.com/image.jpg"


@pytest.mark.asyncio
async def test_create_message_mixed_content(user_id: str):
    message = await Message.create(
        user_id=user_id,
        thread_id="th1",
        instructions=[
            {"type": "text", "text": "Hello"},
            {
                "type": "image_url",
                "image": {"type": "image", "uri": "https://example.com/image.jpg"},
            },
        ],
    )

    assert await Message.get(message.id) is not None

    assert message.content is not None
    assert isinstance(message.content, list)
    assert len(message.content) == 2
    assert message.content[0].type == "text"
    assert message.content[0].text == "Hello"
    assert message.content[1].type == "image_url"
    assert message.content[1].image.uri == "https://example.com/image.jpg"


# ============================================================================
# Encryption Tests
# ============================================================================


@pytest.mark.asyncio
class TestMessageEncryption:
    """Tests for message content encryption at rest."""

    async def test_content_is_encrypted_in_db(self, user_id: str):
        """Test that message content is encrypted when stored in the database."""
        content = "This is a secret message"
        message = await Message.create(
            user_id=user_id,
            thread_id="encryption_test_thread",
            content=content,
            role="user",
        )

        # The encrypted_content field should be set and different from plaintext
        assert message.encrypted_content is not None
        assert message.encrypted_content != content
        # Content should still be available in memory
        assert message.content == content

    async def test_content_decryption(self, user_id: str):
        """Test that message content can be decrypted correctly."""
        content = "This is a secret message to decrypt"
        message = await Message.create(
            user_id=user_id,
            thread_id="decryption_test_thread",
            content=content,
            role="user",
        )

        # Decrypt and verify
        decrypted = await message.get_decrypted_content()
        assert decrypted == content

    async def test_message_loaded_from_db_can_decrypt(self, user_id: str):
        """Test that messages loaded from database can decrypt their content."""
        content = "Message loaded from database"
        message = await Message.create(
            user_id=user_id,
            thread_id="db_load_test_thread",
            content=content,
            role="assistant",
        )

        # Load from database (simulating fresh retrieval)
        loaded = await Message.get(message.id)
        assert loaded is not None

        # Decrypt content from loaded message
        decrypted = await loaded.get_decrypted_content()
        assert decrypted == content

    async def test_dict_content_encryption(self, user_id: str):
        """Test that dictionary content is properly encrypted and decrypted."""
        content = {"key": "value", "nested": {"data": 123}}
        message = await Message.create(
            user_id=user_id,
            thread_id="dict_content_test",
            content=content,
            role="tool",
        )

        # Verify encryption occurred
        assert message.encrypted_content is not None
        assert message.encrypted_content != str(content)

        # Verify decryption works
        decrypted = await message.get_decrypted_content()
        assert decrypted == content

    async def test_list_content_encryption(self, user_id: str):
        """Test that list content (MessageContent) is properly encrypted and decrypted."""
        content = [
            {"type": "text", "text": "Hello, world!"},
            {"type": "text", "text": "This is encrypted."},
        ]
        message = await Message.create(
            user_id=user_id,
            thread_id="list_content_test",
            content=content,
            role="user",
        )

        # Verify encryption occurred
        assert message.encrypted_content is not None

        # Verify decryption works
        decrypted = await message.get_decrypted_content()
        assert decrypted is not None
        assert len(decrypted) == 2

    async def test_null_content_handling(self, user_id: str):
        """Test that null content is handled correctly."""
        message = await Message.create(
            user_id=user_id,
            thread_id="null_content_test",
            content=None,
            role="user",
        )

        # Encrypted content should be None when content is None
        assert message.encrypted_content is None

        # Decryption should return None
        decrypted = await message.get_decrypted_content()
        assert decrypted is None

    async def test_update_content(self, user_id: str):
        """Test updating message content with encryption."""
        original_content = "Original content"
        new_content = "Updated content"

        message = await Message.create(
            user_id=user_id,
            thread_id="update_content_test",
            content=original_content,
            role="user",
        )

        original_encrypted = message.encrypted_content

        # Update the content
        await message.update_content(new_content)

        # Verify encryption changed
        assert message.encrypted_content != original_encrypted
        assert message.content == new_content

        # Verify decryption works
        decrypted = await message.get_decrypted_content()
        assert decrypted == new_content

    async def test_user_isolation(self, user_id: str):
        """Test that messages from different users have different encryption."""
        content = "Same content for both users"
        user1 = "user_isolation_1"
        user2 = "user_isolation_2"

        message1 = await Message.create(
            user_id=user1,
            thread_id="isolation_test",
            content=content,
            role="user",
        )

        message2 = await Message.create(
            user_id=user2,
            thread_id="isolation_test",
            content=content,
            role="user",
        )

        # Encrypted values should be different due to user-specific key derivation
        assert message1.encrypted_content != message2.encrypted_content

        # But both should decrypt to the same content
        assert await message1.get_decrypted_content() == content
        assert await message2.get_decrypted_content() == content

    async def test_same_content_different_ciphertext(self, user_id: str):
        """Test that encrypting the same content produces different ciphertext (IV randomization)."""
        content = "Same content to encrypt"

        message1 = await Message.create(
            user_id=user_id,
            thread_id="ciphertext_test_1",
            content=content,
            role="user",
        )

        message2 = await Message.create(
            user_id=user_id,
            thread_id="ciphertext_test_2",
            content=content,
            role="user",
        )

        # Fernet uses random IV, so encrypted values should be different
        assert message1.encrypted_content != message2.encrypted_content

        # But both should decrypt to the same value
        assert await message1.get_decrypted_content() == content
        assert await message2.get_decrypted_content() == content
