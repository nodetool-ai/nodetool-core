"""
Example: Using node_id and job_id for asset tracking

This example demonstrates how to use the new node_id and job_id fields
when creating assets from within a node's process method.
"""

from io import BytesIO

from nodetool.metadata.types import ImageRef
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class ExampleAutoSaveNode(BaseNode):
    """
    Example node that saves its output as an asset with proper tracking.

    Set _auto_save_asset = True to indicate this node automatically saves assets.
    """

    _auto_save_asset = True  # Indicates this node automatically saves assets

    async def process(self, context: ProcessingContext) -> ImageRef:
        """
        Process the node and automatically save output as an asset with tracking.

        The asset will be tagged with:
        - workflow_id: from context.workflow_id
        - job_id: from context.job_id
        - node_id: from self.id
        """
        # Generate or process your data
        image_data = BytesIO(b"example image data")

        # Create asset with full tracking information
        await context.create_asset(
            name=f"output_{self.id}.png",
            content_type="image/png",
            content=image_data,
            node_id=self.id,  # Tag with node that created it
        )

        # Note: workflow_id and job_id are automatically included by create_asset
        # from the ProcessingContext

        # Return the asset as an ImageRef
        return await context.image_from_io(image_data)


# Example: Querying assets by node or job
async def query_assets_by_node(user_id: str, node_id: str):
    """Query all assets created by a specific node."""
    from nodetool.models.asset import Asset

    assets, _ = await Asset.paginate(
        user_id=user_id,
        node_id=node_id,
        limit=100
    )
    return assets


async def query_assets_by_job(user_id: str, job_id: str):
    """Query all assets created during a specific job execution."""
    from nodetool.models.asset import Asset

    assets, _ = await Asset.paginate(
        user_id=user_id,
        job_id=job_id,
        limit=100
    )
    return assets


async def query_assets_by_workflow(user_id: str, workflow_id: str):
    """Query all assets created for a specific workflow."""
    from nodetool.models.asset import Asset

    assets, _ = await Asset.paginate(
        user_id=user_id,
        workflow_id=workflow_id,
        limit=100
    )
    return assets
