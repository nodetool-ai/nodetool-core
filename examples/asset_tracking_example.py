"""
Example: Using node_id and job_id for asset tracking with automatic saving

This example demonstrates:
1. Automatic asset saving when _auto_save_asset = True
2. Manual asset creation with tracking information
3. Querying assets by node, job, or workflow
"""

from io import BytesIO

from nodetool.metadata.types import ImageRef
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class AutoSaveImageNode(BaseNode):
    """
    Example node that automatically saves its output assets.

    With _auto_save_asset = True, any AssetRef in the output will be
    automatically saved to storage with node_id, job_id, and workflow_id tracking.
    """

    _auto_save_asset = True  # Enable automatic asset saving

    async def process(self, context: ProcessingContext) -> ImageRef:
        """
        Process and return an ImageRef - it will be saved automatically.

        The system will:
        1. Detect the ImageRef in the output
        2. Save it to storage
        3. Tag it with node_id (self.id), job_id, and workflow_id
        4. Update the ImageRef with the saved asset_id
        """
        # Generate or process your image data
        image_data = BytesIO(b"example image data")

        # Just return the ImageRef - no need to call create_asset!
        return await context.image_from_io(image_data)


class ManualSaveImageNode(BaseNode):
    """
    Example node that manually saves assets with tracking.

    For more control over the saving process, you can manually create assets.
    """

    async def process(self, context: ProcessingContext) -> ImageRef:
        """
        Manually save an asset with full control over the process.
        """
        # Generate or process your data
        image_data = BytesIO(b"example image data")

        # Manually create asset with tracking information
        asset = await context.create_asset(
            name=f"output_{self.id}.png",
            content_type="image/png",
            content=image_data,
            node_id=self.id,  # Tag with node that created it
        )
        # Note: workflow_id and job_id are automatically included by create_asset

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
