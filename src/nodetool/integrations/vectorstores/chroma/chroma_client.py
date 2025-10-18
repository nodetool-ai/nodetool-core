"""
ChromaDB Client Module

This module provides utilities for interacting with ChromaDB, a vector database for storing and querying embeddings.
It handles both local and remote ChromaDB instances, with support for:

- Client initialization and configuration
- Collection management with automatic embedding function selection
- Document splitting and chunking
- Multi-tenant support for remote instances
- Integration with both Ollama and SentenceTransformer embedding models

The module automatically configures embedding functions based on collection metadata and
environment settings, defaulting to SentenceTransformer when no specific model is specified.

Key Functions:
    - get_chroma_client(): Initialize a ChromaDB client (local or remote)
    - get_collection(): Retrieve a collection with appropriate embedding function
    - split_document(): Split text into chunks for vector storage
    - get_all_collections(): Retrieve all collections with configured embedding functions
"""

import chromadb
from chromadb.config import Settings, DEFAULT_DATABASE, DEFAULT_TENANT
from urllib.parse import urlparse
from nodetool.config.logging_config import get_logger


from nodetool.config.environment import Environment
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
    SentenceTransformerEmbeddingFunction,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document
from typing import List

from nodetool.metadata.types import TextChunk

log = get_logger(__name__)


def get_chroma_client(
    user_id: str | None = None,
):
    """
    Get a ChromaDB client instance.

    Args:
        user_id (str | None): User ID for tenant creation. Required for remote connections.
        existing_client (ClientAPI | None): Existing ChromaDB client to reuse if provided.

    Returns:
        ClientAPI: ChromaDB client instance
    """
    url = Environment.get_chroma_url()
    token = Environment.get_chroma_token()

    if url is not None:
        parsed_url = urlparse(url)

        # Only create tenant if user_id is provided
        if user_id:
            admin = chromadb.AdminClient(
                settings=Settings(
                    chroma_api_impl="chromadb.api.fastapi.FastAPI",
                    chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
                    chroma_client_auth_credentials=token,
                    chroma_server_host=parsed_url.hostname,
                    chroma_server_http_port=parsed_url.port,
                    anonymized_telemetry=False,
                ),
            )
            tenant = f"tenant_{user_id}"
            try:
                admin.get_tenant(tenant)
            except Exception:
                log.info(f"Creating tenant {tenant}")
                admin.create_tenant(tenant)
                log.info(f"Creating database {DEFAULT_DATABASE}")
                admin.create_database(DEFAULT_DATABASE, tenant)

        return chromadb.HttpClient(
            host=parsed_url.hostname,
            port=parsed_url.port,
            settings=Settings(
                chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
                chroma_client_auth_credentials=token,
                anonymized_telemetry=False,
            ),
            tenant=f"tenant_{user_id}" if user_id else DEFAULT_TENANT,
            database=DEFAULT_DATABASE,
        )
    else:
        return chromadb.PersistentClient(
            path=Environment.get_chroma_path(),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE,
            settings=Settings(anonymized_telemetry=False),
        )


def get_collection(name: str) -> chromadb.Collection:
    """
    Get a collection by name.
    Automatically handles embedding model selection.
    Raises an error if the collection doesn't have embedding model information.

    Args:
        name: The name of the collection.

    Returns:
        The collection.
    """
    if not name:
        raise ValueError("Collection name cannot be empty")

    client = get_chroma_client()
    return client.get_collection(name=name)


DEFAULT_SEPARATORS = [
    "\n\n",
    "\n",
    ".",
]


def split_document(
    text: str,
    source_id: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 1000,
    separators: List[str] = DEFAULT_SEPARATORS,
) -> List[TextChunk]:
    """
    Split text using markdown headers and/or chunk size.

    Args:
        text: Text content to split
        chunk_size: Optional size for further chunking
        chunk_overlap: Overlap between chunks when using chunk_size
        separators: List of separators to use for splitting, in order of preference

    Returns:
        List of dictionaries containing split text and metadata
    """
    splits = []

    # Split by headers if specified
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
        add_start_index=True,
    )

    splits = splitter.split_documents([Document(page_content=text)])

    # Normalize output format
    return [
        TextChunk(
            text=doc.page_content,
            source_id=source_id,
            start_index=doc.metadata["start_index"],
        )
        for doc in splits
    ]


def get_all_collections() -> List[chromadb.Collection]:
    """
    Get all collections from the ChromaDB instance.
    Automatically handles embedding model selection for each collection.

    Returns:
        List[Collection]: List of ChromaDB collections with appropriate embedding functions
    """
    client = get_chroma_client()
    collections = client.list_collections()

    ollama_url = Environment.get("OLLAMA_API_URL")
    result = []

    for collection in collections:
        model = collection.metadata.get("embedding_model")
        print(model)
        if model:
            embedding_function = OllamaEmbeddingFunction(
                url=f"{ollama_url}/api/embeddings", model_name=model, timeout=300
            )
            try:
                embedding_function(["test"])
            except Exception as e:
                log.error(
                    f"Failed to connect or use Ollama model '{model}' for collection '{collection.name}': {e}"
                )
                raise ValueError(
                    f"Ollama model '{model}' for collection '{collection.name}' not available at {ollama_url}. Error: {e}"
                )
        else:
            embedding_function = SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2",
            )

        result.append(
            client.get_collection(
                name=collection.name, embedding_function=embedding_function
            )  # type: ignore
        )

    return result
