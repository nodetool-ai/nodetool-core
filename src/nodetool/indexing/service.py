"""
Collection indexing service shared by API and lightweight server routes.
"""

from __future__ import annotations

from typing import Optional

from nodetool.common.chroma_client import get_collection
from nodetool.indexing.ingestion import default_ingestion_workflow, find_input_nodes
from nodetool.metadata.types import Collection, FilePath
from nodetool.types.job import JobUpdate
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow


async def index_file_to_collection(
    name: str,
    file_path: str,
    mime_type: str,
    token: str = "local_token",
) -> Optional[str]:
    """Index a file into the specified collection.

    If the collection metadata contains a workflow ID, the workflow is executed
    with `CollectionInput` and `FileInput` populated. Otherwise, the default
    ingestion workflow is used.

    Args:
        name: Collection name
        file_path: Temporary file path to index
        mime_type: File MIME type
        token: Auth token for workflow execution (when needed)

    Returns:
        None on success; an error message string if workflow execution failed.
    """
    collection = get_collection(name)

    if collection.metadata and (workflow_id := collection.metadata.get("workflow")):
        processing_context = ProcessingContext(
            user_id="1",
            auth_token=token,
            workflow_id=workflow_id,
        )
        req = RunJobRequest(
            workflow_id=workflow_id,
            user_id="1",
            auth_token=token,
        )
        workflow = await processing_context.get_workflow(workflow_id)
        req.graph = workflow.graph
        req.params = {}

        collection_input, file_input = find_input_nodes(req.graph.model_dump())
        if collection_input:
            req.params[collection_input] = Collection(name=name)
        if file_input:
            req.params[file_input] = FilePath(path=file_path)

        async for msg in run_workflow(req):
            if isinstance(msg, JobUpdate):
                if msg.status == "completed":
                    break
                elif msg.status == "failed":
                    return msg.error or "Indexing workflow failed"
        return None

    # Fallback to default ingestion when no workflow is configured
    default_ingestion_workflow(collection, file_path, mime_type)
    return None


