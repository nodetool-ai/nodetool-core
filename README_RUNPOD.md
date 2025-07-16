# RunPod Deployment for NodeTool

Deploy NodeTool workflows to RunPod serverless infrastructure for scalable, GPU-accelerated execution.

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install runpod
   export RUNPOD_API_KEY="your_api_key"
   ```

2. **Deploy a workflow**:
   ```bash
   python deploy_to_runpod.py --workflow-id YOUR_WORKFLOW_ID --user-id YOUR_USER_ID
   ```

3. **Push to registry**:
   ```bash
   docker tag nodetool-runpod:latest yourusername/nodetool-runpod:latest
   docker push yourusername/nodetool-runpod:latest
   ```

4. **Execute workflows**:
   ```python
   import runpod
   job = runpod.run_sync(
       endpoint_id="your_endpoint_id",
       job_input={"auth_token": "optional"}
   )
   ```

## Features

- ✅ **Embedded Workflows**: Bake workflows into Docker images
- ✅ **GPU Acceleration**: RTX A4000/A5000 class GPUs
- ✅ **Auto-scaling**: 0-3 workers based on demand
- ✅ **Real-time Updates**: Stream execution progress
- ✅ **Cost Optimization**: 5-second idle timeout
- ✅ **Error Handling**: Comprehensive logging and debugging

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   NodeTool DB   │───▶│  Deploy Script   │───▶│  Docker Image   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Job Results   │◀───│  RunPod Handler  │◀───│  RunPod Cloud   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Files

- **`deploy_to_runpod.py`**: Main deployment script
- **`src/nodetool/api/runpod_handler.py`**: Serverless execution handler  
- **`docs/runpod-deployment.md`**: Comprehensive documentation

## Documentation

For detailed setup, configuration, and troubleshooting, see [docs/runpod-deployment.md](docs/runpod-deployment.md).

## Requirements

- Docker
- Python 3.11+
- NodeTool database access
- RunPod API key