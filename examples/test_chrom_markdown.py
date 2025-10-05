# test_chroma_markdown_split_index.py
"""
End-to-end test script for the `ChromaMarkdownSplitAndIndexTool`.

This script demonstrates how to use the `ChromaMarkdownSplitAndIndexTool`
to split Markdown content based on headers and chunk size, and then index
the resulting chunks into a ChromaDB collection.

It performs the following steps:
1. Initializes an in-memory ChromaDB client and creates a test collection.
2. Initializes the `ChromaMarkdownSplitAndIndexTool`.
3. Defines sample Markdown content.
4. Runs the tool's `process` method to split and index the content.
5. Verifies the indexing by retrieving the documents directly from ChromaDB
   and comparing the count.
"""
import asyncio
import chromadb
from nodetool.agents.tools.chroma_tools import ChromaMarkdownSplitAndIndexTool
from nodetool.integrations.vectorstores.chroma.async_chroma_client import get_async_chroma_client
from nodetool.workflows.processing_context import (
    ProcessingContext,
)  # Assuming a basic context is needed


async def run_test():
    """Runs an end-to-end test for ChromaMarkdownSplitAndIndexTool."""

    # 1. Initialize ChromaDB (in-memory for testing)
    print("Initializing ChromaDB client...")
    client = await get_async_chroma_client()
    collection_name = "test_markdown_collection"
    # Ensure the collection is clean for the test
    try:
        await client.delete_collection(name=collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception:
        print(f"Collection {collection_name} does not exist, creating new one.")
        pass  # Collection doesn't exist, which is fine
    collection = await client.create_collection(name=collection_name)
    print(f"Created collection: {collection_name}")


    # 2. Initialize the Tool
    # Use a dummy workspace directory for the test
    workspace_dir = "/tmp/nodetool_test_workspace"
    tool = ChromaMarkdownSplitAndIndexTool(collection=collection)
    print("Initialized ChromaMarkdownSplitAndIndexTool.")

    # 3. Define Test Markdown Content
    test_markdown = """
# Document Title

This is the introduction.

## Section 1

Content for the first section. It contains some text that might be split if the chunk size is small enough.

### Subsection 1.1

Details within the first section.

## Section 2

Content for the second section. This section has bullet points:
*   Item 1
*   Item 2

This section is also longer to test splitting within a header section. Let's add more text here to ensure it potentially exceeds the default chunk size and forces a split based on the RecursiveCharacterTextSplitter logic applied after header splitting. More filler text. Even more filler text.

# Another H1 Header (Edge Case)

Testing how it handles multiple H1 headers.
"""
    document_id = "test_doc_123"
    print(f"\nTest Markdown (Document ID: {document_id}):")
    print("--------------------")
    print(test_markdown)
    print("--------------------")

    # 4. Prepare Parameters and Context
    params = {
        "text": test_markdown,
        "document_id": document_id,
        # Optional: Override defaults if needed
        # "chunk_size": 200,
        # "chunk_overlap": 50,
        "metadata": {"source_type": "test_file", "author": "test_script"},
    }
    # Create a placeholder ProcessingContext if required by your tool implementation
    context = ProcessingContext(workspace_dir=workspace_dir)

    # 5. Run the Tool's Process Method
    print("\nRunning tool.process()...")
    result = await tool.process(context=context, params=params)
    print("\nTool Process Result:")
    print(result)

    # 6. Verification (Query ChromaDB)
    print("\nVerifying results by querying ChromaDB...")
    if result.get("status") == "success":
        indexed_count = result.get("indexed_count", 0)
        print(f"Tool reported {indexed_count} chunks indexed.")

        # Give ChromaDB a moment to process embeddings if necessary
        await asyncio.sleep(1)

        # Retrieve all documents from the collection for verification
        retrieved_docs = await collection.get(
            include=["metadatas", "documents"]
        )

        print(
            f"\nRetrieved {len(retrieved_docs.get('ids', []))} documents from collection '{collection_name}':"
        )
        if retrieved_docs and retrieved_docs.get("ids"):
            print(retrieved_docs)
            assert retrieved_docs["documents"] is not None
            for i, doc_id in enumerate(retrieved_docs["ids"]):
                assert retrieved_docs["documents"][i] is not None
                print(f"\n--- Document ID: {doc_id} ---")
                print(f"Content:\n{retrieved_docs['documents'][i]}")
                print("-------------------------")

            # Basic assertion
            if len(retrieved_docs.get("ids", [])) == indexed_count:
                print(
                    "\n✅ Verification PASSED: Retrieved count matches reported indexed count."
                )
            else:
                print(
                    f"\n❌ Verification FAILED: Retrieved count ({len(retrieved_docs.get('ids', []))}) does not match reported indexed count ({indexed_count})."
                )

        else:
            print(
                "\n❌ Verification FAILED: No documents retrieved from the collection."
            )

    else:
        print("\n❌ Tool execution failed or reported an error.")
        print(f"Error details: {result.get('error')}")

    # 7. Clean up (optional)
    # client.delete_collection(name=collection_name)
    # print(f"\nCleaned up collection: {collection_name}")


if __name__ == "__main__":
    asyncio.run(run_test())
