# Asset Tracking with Node and Job IDs

This document explains the asset tracking feature that allows associating assets with specific nodes and jobs during workflow execution.

## Overview

Assets in NodeTool can now be tagged with:
- **`node_id`**: The ID of the node that created the asset
- **`job_id`**: The ID of the job execution that created the asset
- **`workflow_id`**: The ID of the workflow (already existed, now complemented by node/job IDs)

This enables better tracking and organization of assets created during workflow execution.

## Database Schema

The `nodetool_assets` table now includes:
```sql
ALTER TABLE nodetool_assets ADD COLUMN node_id TEXT DEFAULT NULL;
ALTER TABLE nodetool_assets ADD COLUMN job_id TEXT DEFAULT NULL;
```

## API Usage

### Creating Assets with Tracking

When creating assets via the API:

```bash
curl -X POST http://localhost:7777/api/assets/ \
  -H "Authorization: Bearer <token>" \
  -F "file=@image.jpg" \
  -F 'json={"name":"output.jpg","content_type":"image/jpeg","node_id":"node_123","job_id":"job_456"}'
```

### Querying Assets

You can now filter assets by node_id, job_id, or workflow_id:

```bash
# Get all assets created by a specific node
GET /api/assets?node_id=node_123

# Get all assets from a specific job execution
GET /api/assets?job_id=job_456

# Get all assets for a specific workflow
GET /api/assets?workflow_id=workflow_789

# Combine multiple filters
GET /api/assets?workflow_id=workflow_789&node_id=node_123&job_id=job_456
```

## Python API Usage

### From Within a Node

Nodes can create assets with automatic tracking:

```python
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

class MyNode(BaseNode):
    _auto_save_asset = True  # Indicates this node saves assets
    
    async def process(self, context: ProcessingContext):
        # Create asset with full tracking
        asset = await context.create_asset(
            name=f"output_{self.id}.png",
            content_type="image/png",
            content=my_image_data,
            node_id=self.id,  # Current node ID
        )
        # workflow_id and job_id are automatically added by create_asset
        return asset
```

### Querying Assets

```python
from nodetool.models.asset import Asset

# Query by node
assets, _ = await Asset.paginate(user_id=user_id, node_id=node_id)

# Query by job
assets, _ = await Asset.paginate(user_id=user_id, job_id=job_id)

# Query by workflow
assets, _ = await Asset.paginate(user_id=user_id, workflow_id=workflow_id)

# Combine filters
assets, _ = await Asset.paginate(
    user_id=user_id,
    workflow_id=workflow_id,
    node_id=node_id,
    job_id=job_id
)
```

## Use Cases

### 1. Debugging Workflow Outputs
Quickly find all assets created by a specific node to debug its behavior:
```python
assets, _ = await Asset.paginate(user_id=user_id, node_id="problematic_node_id")
```

### 2. Job Result Tracking
Retrieve all assets generated during a specific job execution:
```python
assets, _ = await Asset.paginate(user_id=user_id, job_id="job_12345")
```

### 3. Workflow Output Management
Find all assets associated with a workflow:
```python
assets, _ = await Asset.paginate(user_id=user_id, workflow_id="workflow_789")
```

### 4. Performance Analysis
Analyze which nodes are creating the most assets or largest files by querying assets grouped by node_id.

## BaseNode Auto-Save Flag

The `_auto_save_asset` class variable on `BaseNode` can be set to `True` to indicate that a node automatically saves its outputs as assets. This is primarily for documentation and metadata purposes:

```python
class ImageProcessingNode(BaseNode):
    _auto_save_asset = True  # Documents that this node saves assets
    
    async def process(self, context: ProcessingContext):
        # Process and save output
        ...
```

You can check if a node auto-saves assets:

```python
if MyNode.auto_save_asset():
    print("This node automatically saves assets")
```

## Migration

The database migration `20260124_000000_add_node_job_fields_to_assets.py` adds the new columns. It runs automatically when the application starts.

## Testing

See test examples in:
- `tests/models/test_asset.py` - Model-level tests for filtering
- `tests/api/test_asset_api.py` - API endpoint tests
- `examples/asset_tracking_example.py` - Usage examples

## Notes

- All three tracking fields (`workflow_id`, `node_id`, `job_id`) are optional
- Assets created outside of workflow execution (e.g., manually uploaded) will have `NULL` for these fields
- The `ProcessingContext.create_asset()` method automatically includes `workflow_id` and `job_id` from the context
- Nodes should explicitly pass `node_id=self.id` when creating assets if they want to track which node created them
