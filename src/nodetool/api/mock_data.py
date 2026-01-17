"""
Mock data generator for testing and development.

Provides functions to populate the database with realistic test data including
threads, messages, workflows, assets, and collections.
"""

import asyncio
import io
import json
import random
import uuid
from datetime import datetime, timedelta
from typing import Any

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    MessageImageContent,
    MessageTextContent,
    Provider,
    ToolCall,
)
from nodetool.models.asset import Asset
from nodetool.models.base_model import create_time_ordered_uuid
from nodetool.models.message import Message
from nodetool.models.thread import Thread
from nodetool.models.workflow import Workflow
from nodetool.runtime.resources import require_scope

log = get_logger(__name__)

# Sample data for generating realistic content
SAMPLE_THREADS = [
    "Image Generation Project",
    "Text Processing Workflow",
    "Audio Transcription Task",
    "Video Analysis Pipeline",
    "Data Transformation",
    "API Integration Demo",
    "Machine Learning Experiment",
    "Creative Writing Assistant",
    "Code Review Helper",
    "Document Summarization",
]

SAMPLE_USER_MESSAGES = [
    "Can you help me create an image of a sunset?",
    "I need to process some text data",
    "How do I transcribe this audio file?",
    "Analyze this video for objects",
    "Transform this CSV into JSON",
    "Connect to the weather API",
    "Train a simple model on this dataset",
    "Write a creative story about space",
    "Review this Python code for bugs",
    "Summarize this document for me",
]

SAMPLE_ASSISTANT_RESPONSES = [
    "I'll help you create that image. Here's what I've generated...",
    "I've processed your text data successfully. Here are the results...",
    "I've transcribed your audio file. Here's the transcript...",
    "I've analyzed your video and found several objects...",
    "I've transformed your CSV file into JSON format...",
    "I've connected to the API and retrieved the weather data...",
    "I've trained a model on your dataset with 95% accuracy...",
    "Here's a creative story about space exploration...",
    "I've reviewed your code and found a few potential issues...",
    "Here's a summary of your document...",
]

SAMPLE_WORKFLOW_NAMES = [
    "Image Generator",
    "Text Processor",
    "Audio Transcriber",
    "Video Analyzer",
    "Data Transformer",
    "API Client",
    "Model Trainer",
    "Story Writer",
    "Code Reviewer",
    "Document Summarizer",
]

SAMPLE_WORKFLOW_DESCRIPTIONS = [
    "Generate images using AI models",
    "Process and analyze text data",
    "Transcribe audio files to text",
    "Analyze video content for objects and scenes",
    "Transform data between different formats",
    "Connect to external APIs and process responses",
    "Train machine learning models on custom datasets",
    "Generate creative writing using AI",
    "Review code for bugs and improvements",
    "Summarize long documents into key points",
]


async def generate_mock_threads(user_id: str = "1", count: int = 5) -> list[Thread]:
    """Generate mock threads with realistic titles."""
    threads = []
    for i in range(count):
        thread_id = create_time_ordered_uuid()
        created_at = datetime.now() - timedelta(days=random.randint(0, 30))

        thread = await Thread.create(
            user_id=user_id,
            id=thread_id,
            title=random.choice(SAMPLE_THREADS),
            created_at=created_at,
            updated_at=created_at + timedelta(hours=random.randint(1, 24)),
        )
        threads.append(thread)
        log.info(f"Created mock thread: {thread.title} ({thread.id})")

    return threads


async def generate_mock_messages(thread_id: str, user_id: str = "1", count: int = 10) -> list[Message]:
    """Generate mock messages for a thread."""
    messages = []
    created_at = datetime.now() - timedelta(hours=count)

    for i in range(count):
        # Alternate between user and assistant messages
        is_user = i % 2 == 0

        if is_user:
            content: Any = [MessageTextContent(text=random.choice(SAMPLE_USER_MESSAGES))]
            role = "user"
            provider = None
            model = None
        else:
            content = [MessageTextContent(text=random.choice(SAMPLE_ASSISTANT_RESPONSES))]
            role = "assistant"
            provider = Provider.Fake
            model = "fake-model-v1"

        message = await Message.create(
            thread_id=thread_id,
            user_id=user_id,
            role=role,
            content=content,
            provider=provider,
            model=model,
            created_at=created_at + timedelta(minutes=i * 5),
        )
        messages.append(message)

    log.info(f"Created {len(messages)} mock messages for thread {thread_id}")
    return messages


async def generate_mock_workflows(user_id: str = "1", count: int = 5) -> list[Workflow]:
    """Generate mock workflows with example graphs."""
    workflows = []

    for i in range(count):
        name = random.choice(SAMPLE_WORKFLOW_NAMES)
        description = random.choice(SAMPLE_WORKFLOW_DESCRIPTIONS)

        # Create a simple graph structure
        graph = {
            "nodes": [
                {
                    "id": str(uuid.uuid4()),
                    "type": "nodetool.input.StringInput",
                    "data": {"name": "input", "value": "Hello World"},
                    "position": {"x": 100, "y": 100},
                },
                {
                    "id": str(uuid.uuid4()),
                    "type": "nodetool.output.Output",
                    "data": {"name": "output"},
                    "position": {"x": 400, "y": 100},
                },
            ],
            "edges": [],
        }

        workflow = await Workflow.create(
            user_id=user_id,
            name=name,
            description=description,
            graph=graph,
            tags=["mock", "example"],
            access="private",
        )
        workflows.append(workflow)
        log.info(f"Created mock workflow: {name} ({workflow.id})")

    return workflows


async def generate_mock_assets(user_id: str = "1", count: int = 10) -> list[Asset]:
    """Generate mock assets including files and folders."""
    assets = []
    storage = require_scope().get_asset_storage()

    # Create a root folder
    folder = await Asset.create(
        user_id=user_id,
        name="Mock Data",
        content_type="folder",
        parent_id=None,
    )
    assets.append(folder)
    log.info(f"Created mock folder: {folder.name} ({folder.id})")

    # Create some image assets
    try:
        from PIL import Image

        for i in range(count):
            asset_id = create_time_ordered_uuid()
            asset_name = f"mock_image_{i}.png"

            # Create a simple test image
            img = Image.new(
                "RGB", (100, 100), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            )
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            # Generate file name for storage
            file_name = f"{user_id}/{asset_id}.png"

            # Upload to storage
            await storage.upload(file_name, img_bytes)

            # Create asset record
            asset = await Asset.create(
                id=asset_id,
                user_id=user_id,
                name=asset_name,
                content_type="image/png",
                file_name=file_name,
                parent_id=folder.id,
            )
            assets.append(asset)

        log.info(f"Created {count} mock image assets")
    except ImportError:
        log.warning("PIL not available, skipping image asset generation")

    # Create some text file assets
    for i in range(3):
        asset_id = create_time_ordered_uuid()
        asset_name = f"mock_text_{i}.txt"

        # Create a simple text file
        text_content = f"This is mock text file number {i}\n" * 10
        text_bytes = io.BytesIO(text_content.encode("utf-8"))

        # Generate file name for storage
        file_name = f"{user_id}/{asset_id}.txt"

        # Upload to storage
        await storage.upload(file_name, text_bytes)

        # Create asset record
        asset = await Asset.create(
            id=asset_id,
            user_id=user_id,
            name=asset_name,
            content_type="text/plain",
            file_name=file_name,
            parent_id=folder.id,
        )
        assets.append(asset)

    log.info(f"Created 3 mock text file assets")
    return assets


async def generate_mock_collections(user_id: str = "1", count: int = 2) -> list[str]:
    """Generate mock ChromaDB collections with sample documents."""
    try:
        from nodetool.integrations.vectorstores.chroma.async_chroma_client import (
            get_async_chroma_client,
        )
    except ImportError:
        log.warning("ChromaDB not available, skipping mock collections")
        return []

    collections = []

    try:
        chroma_client = await get_async_chroma_client()

        for i in range(count):
            collection_name = f"mock_collection_{i}"

            # Create collection
            try:
                collection = await chroma_client.create_collection(
                    name=collection_name,
                    metadata={"user_id": user_id, "mock": True},
                )
            except Exception:
                # Collection might already exist
                collection = await chroma_client.get_collection(name=collection_name)

            # Add sample documents
            documents = [
                "This is a sample document about artificial intelligence.",
                "Machine learning is a subset of AI focused on learning from data.",
                "Neural networks are inspired by biological neurons.",
                "Deep learning uses multiple layers of neural networks.",
                "Natural language processing helps computers understand text.",
            ]

            await collection.add(
                documents=documents,
                ids=[f"doc_{i}_{j}" for j in range(len(documents))],
                metadatas=[{"source": "mock", "index": j} for j in range(len(documents))],
            )

            collections.append(collection_name)
            log.info(f"Created mock collection: {collection_name} with {len(documents)} documents")

    except Exception as e:
        log.error(f"Failed to create mock collections: {e}")

    return collections


async def populate_mock_data(user_id: str = "1") -> dict[str, Any]:
    """
    Populate the database with mock data for testing.

    Args:
        user_id: User ID to associate with all mock data

    Returns:
        Dictionary with counts of created items
    """
    log.info(f"Populating mock data for user {user_id}")

    # Generate threads
    threads = await generate_mock_threads(user_id=user_id, count=5)

    # Generate messages for each thread
    all_messages = []
    for thread in threads[:3]:  # Only add messages to first 3 threads
        messages = await generate_mock_messages(thread_id=thread.id, user_id=user_id, count=10)
        all_messages.extend(messages)

    # Generate workflows
    workflows = await generate_mock_workflows(user_id=user_id, count=5)

    # Generate assets
    assets = await generate_mock_assets(user_id=user_id, count=10)

    # Generate collections
    collections = await generate_mock_collections(user_id=user_id, count=2)

    result = {
        "threads": len(threads),
        "messages": len(all_messages),
        "workflows": len(workflows),
        "assets": len(assets),
        "collections": len(collections),
    }

    log.info(f"Mock data populated: {result}")
    return result
