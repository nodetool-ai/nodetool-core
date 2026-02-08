import pytest
from fastapi.testclient import TestClient

from nodetool.metadata.types import Message as APIMessage
from nodetool.models.message import Message
from nodetool.models.thread import Thread
from nodetool.types.message_types import MessageList


@pytest.mark.asyncio
async def test_create_message(client: TestClient, thread: Thread, headers: dict[str, str], user_id: str):
    message = APIMessage(thread_id=thread.id, role="user", content="Hello")
    json = message.model_dump()
    response = client.post("/api/messages/", json=json, headers=headers)
    assert response.status_code == 200

    m = await Message.get(response.json()["id"])
    assert m is not None
    assert m.content == "Hello"


@pytest.mark.asyncio
async def test_create_message_no_thread(client: TestClient, headers: dict[str, str], user_id: str):
    message = APIMessage(role="user", content="Hello")
    json = message.model_dump()
    response = client.post("/api/messages/", json=json, headers=headers)
    assert response.status_code == 200

    m = await Message.get(response.json()["id"])
    assert m is not None
    assert m.content == "Hello"
    assert m.thread_id is not None


@pytest.mark.asyncio
async def test_get_messages(client: TestClient, message: Message, thread: Thread, headers: dict[str, str]):
    response = client.get(
        "/api/messages/",
        headers=headers,
        params={"thread_id": thread.id, "reverse": "0"},
    )
    assert response.status_code == 200
    message_list = MessageList(**response.json())
    assert len(message_list.messages) == 1
    assert message_list.messages[0].id == message.id


@pytest.mark.asyncio
async def test_get_messages_reverse(client: TestClient, message: Message, thread: Thread, headers: dict[str, str]):
    # create second message
    last_message = await Message.create(
        user_id=message.user_id,
        thread_id=message.thread_id,
        role="user",
        instructions="Last",
    )
    response = client.get(
        "/api/messages/",
        headers=headers,
        params={"thread_id": thread.id, "reverse": "1"},
    )
    assert response.status_code == 200
    message_list = MessageList(**response.json())
    assert len(message_list.messages) == 2
    assert message_list.messages[0].id == last_message.id
    assert message_list.messages[1].id == message.id


@pytest.mark.asyncio
async def test_search_messages_fts(client: TestClient, thread: Thread, headers: dict[str, str], user_id: str):
    await Message.create(user_id=user_id, thread_id=thread.id, role="user", content="Find me please")
    response = client.get(
        "/api/messages/search",
        headers=headers,
        params={"query": "find", "thread_id": thread.id},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["messages"]
    assert payload["messages"][0]["content"] == "Find me please"


@pytest.mark.asyncio
async def test_search_messages_similar_returns_empty_during_tests(
    client: TestClient, thread: Thread, headers: dict[str, str], user_id: str
):
    await Message.create(user_id=user_id, thread_id=thread.id, role="user", content="Semantic search text")
    response = client.get(
        "/api/messages/similar",
        headers=headers,
        params={"query": "semantic", "thread_id": thread.id},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["messages"] == []
