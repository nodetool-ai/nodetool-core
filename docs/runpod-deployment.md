# RunPod Deployment Guide for NodeTool Workflows

This guide explains how to deploy NodeTool workflows to RunPod serverless infrastructure for scalable, GPU-accelerated execution.

## Overview

The RunPod deployment system allows you to:
- Deploy specific NodeTool workflows as serverless functions
- Execute workflows with GPU acceleration (RTX A4000/A5000 class)
- Auto-scale based on demand (0-3 workers)
- Minimize costs with automatic worker termination
- Run workflows in isolated, containerized environments

## Architecture

The deployment consists of two main components:

### 1. Deployment Script (`deploy_to_runpod.py`)
- Fetches workflows from the NodeTool database
- Embeds complete workflow data into Docker images
- Builds specialized containers for RunPod execution
- Creates RunPod templates and endpoints

### 2. RunPod Handler (`src/nodetool/api/runpod_handler.py`)
- Processes workflow execution requests on RunPod
- Supports both embedded and dynamic workflows
- Streams real-time execution updates
- Handles errors and provides detailed logging

## Prerequisites

### Software Requirements
- Docker installed and running
- Python 3.11+ with NodeTool dependencies
- Access to NodeTool database
- RunPod account and API key

### Environment Setup
```bash
# Install RunPod SDK
pip install runpod

# Set RunPod API key
export RUNPOD_API_KEY="your_runpod_api_key_here"

# Ensure NodeTool environment is configured
# (Database connections, S3 credentials, etc.)
```

## Deployment Process

### Step 1: Prepare Your Workflow
Ensure your workflow is saved in the NodeTool database and accessible to the deployment user:

```python
# Example: Create or verify workflow exists
from nodetool.models.workflow import Workflow

workflow = Workflow.find(user_id="your_user_id", workflow_id="your_workflow_id")
if workflow:
    print(f"Found workflow: {workflow.name}")
else:
    print("Workflow not found or not accessible")
```

### Step 2: Deploy to RunPod
Run the deployment script with your workflow details:

```bash
python deploy_to_runpod.py \
    --workflow-id "your_workflow_id" \
    --user-id "your_user_id"
```

#### Command Options
- `--workflow-id`: Required. The unique ID of the workflow to deploy
- `--user-id`: Required. User ID for workflow access permissions
- `--skip-build`: Optional. Skip Docker image building
- `--skip-deploy`: Optional. Skip RunPod resource creation

### Step 3: Push to Registry
After building, push your Docker image to a public registry:

```bash
# Tag for your registry
docker tag nodetool-runpod:latest yourusername/nodetool-runpod:latest

# Push to registry
docker push yourusername/nodetool-runpod:latest
```

### Step 4: Create RunPod Resources
Uncomment the deployment code in the script to create RunPod templates and endpoints:

```python
# In deploy_to_runpod.py, uncomment these lines:
push_to_registry()
template_id = create_runpod_template()
endpoint_id = create_runpod_endpoint(template_id)
```

## Docker Image Structure

The deployed Docker image contains:

```
/app/
├── embedded_workflow.json      # Complete workflow data
├── runpod_handler.py          # Execution handler
├── venv/                      # Python virtual environment
└── [NodeTool dependencies]    # All required packages
```

### Environment Variables
- `EMBEDDED_WORKFLOW_PATH=/app/embedded_workflow.json`
- `PYTHONPATH=/app`
- `VIRTUAL_ENV=/app/venv`

## Usage

### Executing Workflows
Once deployed, send requests to your RunPod endpoint:

```python
import runpod

# For embedded workflows (minimal input required)
job = runpod.run_sync(
    endpoint_id="your_endpoint_id",
    job_input={
        "auth_token": "optional_auth_token"  # Only if API access needed
    }
)

# For dynamic workflows (full input required)
job = runpod.run_sync(
    endpoint_id="your_endpoint_id",
    job_input={
        "workflow_id": "workflow_id",
        "user_id": "user_id",
        "auth_token": "optional_auth_token",
        "graph": {...}  # Optional workflow graph
    }
)
```

### Response Format
The handler streams responses with job updates:

```json
{
    "job_id": "unique_job_id",
    "status": "running|completed|failed",
    "progress": 0.75,
    "message": "Processing node: image_generation",
    "results": {...},
    "error": null
}
```

## Configuration

### RunPod Template Settings
- **Container Disk**: 20GB for dependencies and temporary files
- **GPU Type**: AMPERE_16 (RTX A4000/A5000 equivalent)
- **Scaling**: 0-3 workers with queue delay trigger
- **Idle Timeout**: 5 seconds for cost optimization
- **Regions**: US locations for optimal performance

### Workflow Settings
Embedded workflows include all database fields:
- `id`, `name`, `description`
- `graph` (nodes and edges)
- `settings` and `run_mode`
- `user_id` and access permissions
- `tags` and metadata

## Troubleshooting

### Common Issues

#### 1. Workflow Not Found
```
Error: Workflow workflow_123 not found or not accessible
```
**Solution**: Verify workflow ID and user permissions.

#### 2. Docker Build Failed
```
Command failed: docker build ...
```
**Solutions**:
- Ensure Docker is running
- Check available disk space
- Verify base Dockerfile exists

#### 3. RunPod API Errors
```
Failed to create template: Unauthorized
```
**Solutions**:
- Verify RUNPOD_API_KEY is set correctly
- Check API key permissions
- Ensure sufficient RunPod credits

#### 4. Image Registry Issues
```
docker push failed: authentication required
```
**Solutions**:
- Login to your registry: `docker login`
- Verify image name matches registry format
- Check registry permissions

### Debugging Tips

#### Local Testing
Test the handler locally before deployment:

```python
# In runpod_handler.py, uncomment the test code:
import os
os.environ["EMBEDDED_WORKFLOW_PATH"] = "/path/to/workflow.json"

async def main():
    test_job = {"input": {}, "id": "test_job_123"}
    async for msg in async_generator_handler(test_job):
        print(msg)

asyncio.run(main())
```

#### Log Analysis
Check RunPod logs for execution details:
- Processing context initialization
- Workflow loading and validation
- Node execution progress
- Error messages and stack traces

#### Resource Monitoring
Monitor RunPod dashboard for:
- Worker scaling behavior
- Execution times and costs
- GPU utilization
- Queue depth and wait times

## Best Practices

### Workflow Design
- **Optimize for GPU**: Use GPU-accelerated nodes when possible
- **Minimize I/O**: Reduce external API calls and file operations
- **Error Handling**: Include robust error handling in custom nodes
- **Resource Management**: Clean up temporary files and memory

### Deployment Strategy
- **Version Control**: Tag Docker images with workflow versions
- **Testing**: Test workflows locally before deployment
- **Monitoring**: Set up alerts for failed executions
- **Cost Control**: Monitor usage and adjust scaling parameters

### Security Considerations
- **API Keys**: Never embed secrets in Docker images
- **Access Control**: Use proper user permissions for workflows
- **Network Security**: Restrict outbound connections if needed
- **Data Privacy**: Ensure sensitive data handling compliance

## Cost Optimization

### Scaling Configuration
```python
# Adjust these parameters in create_runpod_endpoint():
workers_min=0,        # Start with zero workers
workers_max=3,        # Limit maximum concurrent workers
idle_timeout=5,       # Terminate workers quickly
scaler_value=4        # Scale up after 4 seconds in queue
```

### Resource Sizing
- Use minimum required GPU type for your workload
- Optimize container disk size (20GB default)
- Consider volume storage for large datasets

### Usage Patterns
- **Batch Processing**: Group multiple workflows together
- **Off-Peak Execution**: Schedule non-urgent workflows during cheaper hours
- **Resource Pooling**: Share endpoints across similar workflows

## Advanced Configuration

### Custom Base Images
Modify the Dockerfile for specialized requirements:

```dockerfile
# Add custom dependencies
RUN pip install your-custom-package

# Configure system settings
ENV YOUR_ENV_VAR=value

# Custom initialization scripts
COPY init-script.sh /app/
RUN chmod +x /app/init-script.sh
```

### Multi-Workflow Deployment
Deploy multiple workflows to the same endpoint:

```python
# Modify job_data based on input parameters
if "workflow_name" in job_data:
    workflow_mapping = {
        "image_gen": "workflow_id_1",
        "text_analysis": "workflow_id_2"
    }
    workflow_id = workflow_mapping.get(job_data["workflow_name"])
```

### Integration Patterns
- **API Gateway**: Use RunPod behind an API gateway
- **Webhook Triggers**: Set up webhook-based execution
- **Queue Systems**: Integrate with job queue systems
- **Monitoring**: Add custom metrics and logging

## Support and Resources

### Documentation
- [RunPod Documentation](https://docs.runpod.io/)
- [NodeTool Core Documentation](../README.md)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

### Community
- NodeTool Discord/Slack channels
- RunPod community forums
- GitHub issues and discussions

### Professional Support
- NodeTool enterprise support
- RunPod business support plans
- Custom deployment consulting

## Changelog

### Version 1.0.0
- Initial RunPod deployment implementation
- Embedded workflow support
- Comprehensive documentation
- Example configurations and troubleshooting guides