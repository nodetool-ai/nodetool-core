#!/usr/bin/env python

import pytest
from fastapi.testclient import TestClient
from nodetool.models.thread import Thread
from nodetool.models.message import Message
from nodetool.types.thread import ThreadCreateRequest, ThreadUpdateRequest


def test_create_thread(client: TestClient, headers: dict[str, str], user_id: str):
    """Test creating a new thread."""
    request = ThreadCreateRequest(title="Test Thread")
    response = client.post("/api/threads/", json=request.model_dump(), headers=headers)
    
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Test Thread"
    assert data["user_id"] == user_id
    assert "id" in data
    assert "created_at" in data
    assert "updated_at" in data
    
    # Verify in database
    thread = Thread.get(data["id"])
    assert thread is not None
    assert thread.title == "Test Thread"
    assert thread.user_id == user_id


def test_create_thread_default_title(client: TestClient, headers: dict[str, str], user_id: str):
    """Test creating a thread without providing a title."""
    request = ThreadCreateRequest()
    response = client.post("/api/threads/", json=request.model_dump(), headers=headers)
    
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "New Thread"
    assert data["user_id"] == user_id


def test_get_thread(client: TestClient, headers: dict[str, str], thread):
    """Test getting a single thread by ID."""
    response = client.get(f"/api/threads/{thread.id}", headers=headers)
    
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == thread.id
    assert data["title"] == thread.title
    assert data["user_id"] == thread.user_id


def test_get_thread_not_found(client: TestClient, headers: dict[str, str]):
    """Test getting a non-existent thread."""
    response = client.get("/api/threads/non-existent-id", headers=headers)
    
    assert response.status_code == 404
    assert response.json()["detail"] == "Thread not found"


def test_get_thread_unauthorized(client: TestClient, headers: dict[str, str]):
    """Test getting a thread owned by another user."""
    # Create a thread for a different user
    other_thread = Thread.create(user_id="other-user", title="Other User's Thread")
    
    response = client.get(f"/api/threads/{other_thread.id}", headers=headers)
    
    assert response.status_code == 404
    assert response.json()["detail"] == "Thread not found"


def test_list_threads(client: TestClient, headers: dict[str, str], user_id: str):
    """Test listing threads with pagination."""
    # Create multiple threads
    threads = []
    for i in range(5):
        thread = Thread.create(user_id=user_id, title=f"Thread {i}")
        threads.append(thread)
    
    response = client.get("/api/threads/", headers=headers, params={"limit": 3})
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["threads"]) == 3
    assert "next" in data
    
    # Test pagination with cursor
    response2 = client.get("/api/threads/", headers=headers, params={"limit": 3, "cursor": data["next"]})
    assert response2.status_code == 200
    data2 = response2.json()
    assert len(data2["threads"]) == 2
    
    # Verify no duplicate threads
    ids1 = [t["id"] for t in data["threads"]]
    ids2 = [t["id"] for t in data2["threads"]]
    assert len(set(ids1) & set(ids2)) == 0


def test_list_threads_reverse(client: TestClient, headers: dict[str, str], user_id: str):
    """Test listing threads in reverse order."""
    # Create threads with specific titles to verify order
    thread1 = Thread.create(user_id=user_id, title="First Thread")
    thread2 = Thread.create(user_id=user_id, title="Second Thread")
    thread3 = Thread.create(user_id=user_id, title="Third Thread")
    
    response = client.get("/api/threads/", headers=headers, params={"reverse": True})
    
    assert response.status_code == 200
    data = response.json()
    # In reverse order, newest (Third) should come first
    assert data["threads"][0]["title"] == "Third Thread"
    assert data["threads"][1]["title"] == "Second Thread"
    assert data["threads"][2]["title"] == "First Thread"


def test_list_threads_only_user_threads(client: TestClient, headers: dict[str, str], user_id: str):
    """Test that listing threads only returns threads for the current user."""
    # Create threads for current user
    my_thread = Thread.create(user_id=user_id, title="My Thread")
    
    # Create threads for other user
    Thread.create(user_id="other-user", title="Other Thread 1")
    Thread.create(user_id="other-user", title="Other Thread 2")
    
    response = client.get("/api/threads/", headers=headers)
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["threads"]) == 1
    assert data["threads"][0]["id"] == my_thread.id
    assert data["threads"][0]["title"] == "My Thread"


def test_update_thread(client: TestClient, headers: dict[str, str], thread):
    """Test updating a thread's title."""
    request = ThreadUpdateRequest(title="Updated Title")
    response = client.put(f"/api/threads/{thread.id}", json=request.model_dump(), headers=headers)
    
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Updated Title"
    assert data["id"] == thread.id
    
    # Verify in database
    updated_thread = Thread.get(thread.id)
    assert updated_thread.title == "Updated Title"
    assert updated_thread.updated_at > thread.updated_at


def test_update_thread_not_found(client: TestClient, headers: dict[str, str]):
    """Test updating a non-existent thread."""
    request = ThreadUpdateRequest(title="Updated Title")
    response = client.put("/api/threads/non-existent-id", json=request.model_dump(), headers=headers)
    
    assert response.status_code == 404
    assert response.json()["detail"] == "Thread not found"


def test_update_thread_unauthorized(client: TestClient, headers: dict[str, str]):
    """Test updating a thread owned by another user."""
    other_thread = Thread.create(user_id="other-user", title="Other User's Thread")
    
    request = ThreadUpdateRequest(title="Hacked Title")
    response = client.put(f"/api/threads/{other_thread.id}", json=request.model_dump(), headers=headers)
    
    assert response.status_code == 404
    assert response.json()["detail"] == "Thread not found"
    
    # Verify thread was not updated
    thread = Thread.get(other_thread.id)
    assert thread.title == "Other User's Thread"


def test_delete_thread(client: TestClient, headers: dict[str, str], thread):
    """Test deleting a thread."""
    thread_id = thread.id
    response = client.delete(f"/api/threads/{thread_id}", headers=headers)
    
    assert response.status_code == 200
    
    # Verify thread is deleted
    deleted_thread = Thread.get(thread_id)
    assert deleted_thread is None


def test_delete_thread_with_messages(client: TestClient, headers: dict[str, str], thread, user_id: str):
    """Test deleting a thread also deletes its messages."""
    # Create messages in the thread
    message1 = Message.create(
        user_id=user_id,
        thread_id=thread.id,
        role="user",
        content="Test message 1"
    )
    message2 = Message.create(
        user_id=user_id,
        thread_id=thread.id,
        role="assistant",
        content="Test response"
    )
    
    # Delete the thread
    response = client.delete(f"/api/threads/{thread.id}", headers=headers)
    assert response.status_code == 200
    
    # Verify thread and messages are deleted
    assert Thread.get(thread.id) is None
    assert Message.get(message1.id) is None
    assert Message.get(message2.id) is None


def test_delete_thread_not_found(client: TestClient, headers: dict[str, str]):
    """Test deleting a non-existent thread."""
    response = client.delete("/api/threads/non-existent-id", headers=headers)
    
    assert response.status_code == 404
    assert response.json()["detail"] == "Thread not found"


def test_delete_thread_unauthorized(client: TestClient, headers: dict[str, str]):
    """Test deleting a thread owned by another user."""
    other_thread = Thread.create(user_id="other-user", title="Other User's Thread")
    
    response = client.delete(f"/api/threads/{other_thread.id}", headers=headers)
    
    assert response.status_code == 404
    assert response.json()["detail"] == "Thread not found"
    
    # Verify thread was not deleted
    thread = Thread.get(other_thread.id)
    assert thread is not None


def test_thread_message_isolation(client: TestClient, headers: dict[str, str], thread, user_id: str):
    """Test that deleting a thread only deletes messages belonging to the same user."""
    # Create a message from the current user
    my_message = Message.create(
        user_id=user_id,
        thread_id=thread.id,
        role="user",
        content="My message"
    )
    
    # Create a message from another user (simulating shared thread scenario)
    other_message = Message.create(
        user_id="other-user",
        thread_id=thread.id,
        role="user",
        content="Other user's message"
    )
    
    # Delete the thread
    response = client.delete(f"/api/threads/{thread.id}", headers=headers)
    assert response.status_code == 200
    
    # Verify only the current user's message is deleted
    assert Message.get(my_message.id) is None
    assert Message.get(other_message.id) is not None  # Other user's message remains