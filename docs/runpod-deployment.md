# RunPod Deployment Guide for NodeTool Workflows

This guide explains how to deploy NodeTool workflows to RunPod serverless infrastructure for scalable, GPU-accelerated execution using the integrated CLI commands.

## Overview

The RunPod deployment system allows you to:
- Deploy specific NodeTool workflows as serverless functions
- Execute workflows with GPU acceleration (various GPU types supported)
- Auto-scale based on demand (0-3 workers)
- Minimize costs with automatic worker termination
- Run workflows in isolated, containerized environments
- Test deployed workflows with integrated CLI tools

## Architecture

The deployment consists of several integrated components:

### 1. CLI Deployment Command (`nodetool deploy`)
- Integrated deployment command in the main CLI
- Fetches workflows from the NodeTool database
- Embeds complete workflow data into Docker images
- Builds specialized containers for RunPod execution
- Creates RunPod templates and endpoints automatically

### 2. CLI Testing Command (`nodetool test-runpod`)
- Integrated testing command for deployed workflows
- Real-time execution monitoring and progress tracking
- Rich console output with colored status messages
- Automatic result saving with timestamps

### 3. RunPod Handler (`src/nodetool/deploy/runpod_handler.py`)
- Processes workflow execution requests on RunPod
- Supports embedded workflow execution
- Streams real-time execution updates
- Handles errors and provides detailed logging

## Prerequisites

### Software Requirements
- Docker installed and running
- Python 3.11+ with NodeTool Core installed
- Access to NodeTool database
- RunPod account and API key
- Docker Hub account (or other registry) for image storage

### Environment Setup
```bash
# Install NodeTool Core with RunPod dependencies
pip install -r requirements-dev.txt

# Set RunPod API key
export RUNPOD_API_KEY="your_runpod_api_key_here"

# Login to Docker registry (for image pushing)
docker login

# Ensure NodeTool environment is configured
# (Database connections, S3 credentials, etc.)
```

## Deployment Process

### Step 1: Prepare Your Workflow
Ensure your workflow is saved in the NodeTool database and accessible:

```python
# Example: Create or verify workflow exists
from nodetool.models.workflow import Workflow

workflow = Workflow.get("your_workflow_id")
if workflow:
    print(f"Found workflow: {workflow.name}")
else:
    print("Workflow not found or not accessible")
```

### Step 2: Deploy to RunPod
Use the integrated CLI command to deploy your workflow:

```bash
# Basic deployment
nodetool deploy --workflow-id "your_workflow_id"

# Deployment with specific Docker username
nodetool deploy --workflow-id "your_workflow_id" --docker-username "yourusername"

# Deployment with specific GPU types
nodetool deploy --workflow-id "your_workflow_id" --gpu-types AMPERE_24 ADA_48_PRO
```

#### Available Options
```bash
# View all deployment options
nodetool deploy --help

# List available GPU types and options
nodetool deploy --list-all-options
```

#### Key Command Options
- `--workflow-id`: Required. The unique ID of the workflow to deploy
- `--docker-username`: Docker Hub username (auto-detected from docker login if not provided)
- `--tag`: Custom image tag (auto-generated if not provided)
- `--gpu-types`: GPU types for endpoint (e.g., AMPERE_24, ADA_48_PRO)
- `--platform`: Docker build platform (default: linux/amd64)
- `--registry`: Docker registry URL (default: docker.io)

### Step 3: Automated Process
The CLI handles the entire deployment pipeline automatically:

1. **Workflow Fetching**: Retrieves workflow data from database
2. **Docker Building**: Creates specialized RunPod container
3. **Registry Push**: Uploads image to Docker registry
4. **Template Creation**: Creates or updates RunPod template
5. **Endpoint Setup**: Creates serverless endpoint with auto-scaling

The deployment will output the endpoint ID for testing and usage.

## Docker Image Structure

The deployed Docker image contains:

```
/app/
├── workflow.json              # Complete embedded workflow data
├── handler.py                 # RunPod execution handler (from runpod_handler.py)
├── .venv/                     # Python virtual environment with dependencies
└── [NodeTool Core]            # Complete NodeTool installation
```

### Environment Variables
- `WORKSPACE_DIR=/app`
- `PYTHONPATH=/app`

## Testing Deployed Workflows

### Using the CLI Test Command
After deployment, test your workflow using the integrated test command:

```bash
# Basic test with no parameters
nodetool test-runpod --endpoint-id YOUR_ENDPOINT_ID

# Test with inline JSON parameters
nodetool test-runpod \
  --endpoint-id YOUR_ENDPOINT_ID \
  --params-json '{"text": "Hello World", "count": 3}'

# Test with parameter file
nodetool test-runpod \
  --endpoint-id YOUR_ENDPOINT_ID \
  --params test_params.json

# Test with custom timeout and output file
nodetool test-runpod \
  --endpoint-id YOUR_ENDPOINT_ID \
  --timeout 120 \
  --output my_results.json
```

### Using RunPod SDK Directly
You can also use the RunPod SDK directly for programmatic access:

```python
import runpod

# Configure API key
runpod.api_key = "your_runpod_api_key"

# Create endpoint instance
endpoint = runpod.Endpoint("your_endpoint_id")

# Execute workflow
job = endpoint.run({
    "param1": "value1",
    "param2": "value2"
})

# Get results
result = job.output()
print(result)
```

### Test Output Example
The CLI test command provides rich, colored output:

```
🧪 Testing RunPod workflow...
Endpoint ID: abc123def456
Parameters: {
  "text": "Hello World"
}
Timeout: 60 seconds
🚀 Starting workflow execution...
Job status: IN_QUEUE
Job status: IN_PROGRESS (elapsed: 1s)
Job status: COMPLETED (elapsed: 3s)
✅ Job completed successfully!
Execution completed in 3 seconds

📊 Job Results:
{
  "id": "12345-67890-abcdef",
  "status": "COMPLETED",
  "output": {
    "result": "Hello World processed successfully"
  }
}

💾 Results saved to: runpod_result_20241220_143022.json
✅ Test completed successfully!
```

## Configuration

### GPU Types Available
The deployment supports various GPU types with different memory and performance characteristics:

#### Ada Lovelace Architecture
- `ADA_24`: L4, RTX 4000 series (24GB)
- `ADA_32_PRO`: Professional Ada cards (32GB)
- `ADA_48_PRO`: L40, L40S, RTX 6000 Ada (48GB)
- `ADA_80_PRO`: High-end Ada professional cards (80GB)

#### Ampere Architecture
- `AMPERE_16`: RTX 3060, A2000, A4000 (16GB)
- `AMPERE_24`: RTX 3070/3080/3090, A4500, A5000 (24GB)
- `AMPERE_48`: A40, RTX A6000 (48GB)
- `AMPERE_80`: A100 (80GB)

#### Hopper Architecture
- `HOPPER_141`: H200 (141GB)

### RunPod Template Settings
- **Container Disk**: 20GB for dependencies and temporary files
- **GPU Types**: Configurable via `--gpu-types` parameter
- **Scaling**: 0-3 workers with queue delay trigger
- **Idle Timeout**: 5 seconds for cost optimization
- **Auto-scaling**: Based on queue depth and request volume

### Workflow Settings
Embedded workflows include all database fields:
- `id`, `name`, `description`
- `graph` (nodes and edges)
- `settings` and `run_mode`
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

#### Using CLI Test Command
Start debugging with the integrated test command:

```bash
# Test with shorter timeout for debugging
nodetool test-runpod --endpoint-id YOUR_ID --timeout 30

# Save results for analysis
nodetool test-runpod --endpoint-id YOUR_ID --output debug_results.json

# Test with verbose parameter logging
nodetool test-runpod \
  --endpoint-id YOUR_ID \
  --params-json '{"debug": true, "verbose": true}'
```

#### View Available Options
```bash
# List all deployment options
nodetool deploy --list-all-options

# Get help for deployment
nodetool deploy --help

# Get help for testing
nodetool test-runpod --help
```

#### Log Analysis
Check RunPod logs for execution details:
- Workflow loading and validation
- Node execution progress
- Error messages and stack traces
- Resource utilization metrics

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

### Security Considerations
- **API Keys**: Never embed secrets in Docker images
- **Access Control**: Use proper user permissions for workflows
- **Network Security**: Restrict outbound connections if needed
- **Data Privacy**: Ensure sensitive data handling compliance

## Cost Optimization

### GPU Type Selection
Choose the appropriate GPU type for your workload:

```bash
# For basic workflows - lower cost
nodetool deploy --workflow-id YOUR_ID --gpu-types AMPERE_16

# For image generation - balanced performance/cost
nodetool deploy --workflow-id YOUR_ID --gpu-types AMPERE_24

# For large models - high performance
nodetool deploy --workflow-id YOUR_ID --gpu-types AMPERE_80
```

### Scaling Configuration
The deployment automatically configures optimal scaling:
- **Workers Min**: 0 (start with zero workers)
- **Workers Max**: 3 (limit maximum concurrent workers)
- **Idle Timeout**: 5 seconds (terminate workers quickly)
- **Auto-scaling**: Based on queue depth and request volume

### Resource Sizing
- Use minimum required GPU type for your workload
- Container disk: 20GB (optimized for NodeTool dependencies)
- Consider workload memory requirements when selecting GPU types

### Usage Patterns
- **Batch Processing**: Group multiple requests together
- **Off-Peak Execution**: Schedule non-urgent workflows during cheaper hours
- **Resource Pooling**: Use the same endpoint for similar workflows
- **Testing**: Use `--timeout` to avoid long-running failed jobs


### Integration Patterns
- **API Gateway**: Use RunPod behind an API gateway
- **Webhook Triggers**: Set up webhook-based execution
- **Queue Systems**: Integrate with job queue systems
- **Monitoring**: Add custom metrics and logging

## Quick Reference

### Essential Commands
```bash
# Deploy a workflow
nodetool deploy --workflow-id YOUR_WORKFLOW_ID

# Test deployed workflow
nodetool test-runpod --endpoint-id YOUR_ENDPOINT_ID

# Deploy with specific GPU type
nodetool deploy --workflow-id YOUR_ID --gpu-types AMPERE_24

# Test with parameters
nodetool test-runpod \
  --endpoint-id YOUR_ID \
  --params-json '{"param": "value"}'

# View all options
nodetool deploy --list-all-options
```

### File Locations
- **Deployment Scripts**: `src/nodetool/deploy/`
- **Dockerfile**: `src/nodetool/deploy/Dockerfile`
- **Handler**: `src/nodetool/deploy/runpod_handler.py`
- **Testing Guide**: `src/nodetool/deploy/runpod_testing_guide.md`

## Support and Resources

### Documentation
- [RunPod Documentation](https://docs.runpod.io/)
- [NodeTool Core Documentation](../README.md)
- [RunPod Testing Guide](../src/nodetool/deploy/runpod_testing_guide.md)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

### Getting Help
```bash
# CLI help
nodetool deploy --help
nodetool test-runpod --help

# List all deployment options
nodetool deploy --list-all-options
```