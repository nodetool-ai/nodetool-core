# HuggingFace Inference Endpoints Deployment

This guide explains how to deploy NodeTool workflows to [HuggingFace Inference Endpoints](https://huggingface.co/inference-endpoints), a fully managed service for deploying AI models and custom containers.

## Overview

HuggingFace Inference Endpoints provides:
- Serverless or dedicated infrastructure
- Auto-scaling based on traffic
- Built-in monitoring and logging
- Easy integration with HuggingFace Hub

NodeTool supports deploying workflows as custom containers to HuggingFace Inference Endpoints.

## Prerequisites

1. **HuggingFace Account**: You need a HuggingFace account with an API token that has write access.

2. **Docker**: Docker must be installed and running on your machine.

3. **Container Registry**: Access to a container registry (Docker Hub, etc.) where you can push images.

## Configuration

### 1. Set Environment Variables

```bash
# HuggingFace API token with write access
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# Optional: Docker Hub username if not already logged in
export DOCKER_USERNAME=your-dockerhub-username
```

### 2. Create Deployment Configuration

Initialize a deployment configuration if you haven't already:

```bash
nodetool deploy init
```

Then edit `~/.config/nodetool/deployment.yaml` to add a HuggingFace deployment:

```yaml
version: "1.0"
defaults:
  chat_provider: openai
  default_model: gpt-4o-mini
  log_level: INFO
  auth_provider: local

deployments:
  my-hf-endpoint:
    type: huggingface
    enabled: true
    namespace: your-hf-username  # or organization name
    endpoint_name: nodetool-workflow
    image:
      registry: docker.io
      repository: your-dockerhub-username/nodetool-hf
      tag: latest
      build:
        platform: linux/amd64
        dockerfile: Dockerfile.hf
    resources:
      instance_size: small  # small, medium, large, xlarge
      instance_type: intel-icl  # CPU type or GPU type
      min_replica: 0  # Scale to zero when idle
      max_replica: 1
    region: us-east-1  # us-east-1, eu-west-1, etc.
    vendor: aws  # aws, gcp, azure
    task: custom
    custom_image: true
    environment:
      NODETOOL_WORKFLOW_ID: your-workflow-id  # Optional: default workflow
      LOG_LEVEL: INFO
```

## Deployment

### Basic Deployment

Deploy using the configuration from `deployment.yaml`:

```bash
nodetool deploy hf my-hf-endpoint
```

### Deployment Options

```bash
# Skip Docker build (use existing image)
nodetool deploy hf my-hf-endpoint --skip-build

# Skip pushing to registry (image already in registry)
nodetool deploy hf my-hf-endpoint --skip-build --skip-push

# Skip endpoint creation (just build and push)
nodetool deploy hf my-hf-endpoint --skip-endpoint

# Build without Docker cache
nodetool deploy hf my-hf-endpoint --no-cache

# Custom chat configuration
nodetool deploy hf my-hf-endpoint --chat-provider anthropic --default-model claude-3-opus-20240229
```

## Instance Types

### CPU Instances

| Size   | vCPU | Memory | Use Case |
|--------|------|--------|----------|
| small  | 1    | 2GB    | Light workloads |
| medium | 2    | 4GB    | Standard workloads |
| large  | 4    | 8GB    | Heavy processing |
| xlarge | 8    | 16GB   | Very heavy workloads |

### GPU Instances

| Type       | GPU      | Memory | Use Case |
|------------|----------|--------|----------|
| nvidia-a10g | A10G    | 24GB   | General inference |
| nvidia-a100 | A100    | 40GB   | Large models |
| nvidia-t4   | T4      | 16GB   | Cost-effective |

## Regions

Available regions depend on the vendor:

### AWS
- `us-east-1` (N. Virginia)
- `eu-west-1` (Ireland)
- `ap-southeast-1` (Singapore)

### GCP
- `us-central1`
- `europe-west4`

### Azure
- `eastus`
- `westeurope`

## API Usage

Once deployed, your endpoint can be called via the HuggingFace Inference API:

```python
import requests

API_URL = "https://your-endpoint-url.endpoints.huggingface.cloud"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Run a workflow
result = query({
    "inputs": {
        "prompt": "Hello, world!"
    },
    "workflow_id": "your-workflow-id"
})
print(result)
```

### Request Format

```json
{
  "inputs": {
    "param1": "value1",
    "param2": "value2"
  },
  "workflow_id": "optional-workflow-id",
  "parameters": {
    "additional": "params"
  }
}
```

### Response Format

```json
{
  "results": {
    "output_name": "output_value"
  },
  "status": "success",
  "workflow_id": "workflow-id"
}
```

## Monitoring

### Dashboard

Access your endpoint dashboard at:
```
https://ui.endpoints.huggingface.co/{namespace}/endpoints/{endpoint-name}
```

### Logs

View logs through the HuggingFace dashboard or use the API:

```bash
curl -H "Authorization: Bearer $HF_TOKEN" \
  "https://api.endpoints.huggingface.cloud/v2/endpoint/{namespace}/{endpoint}/logs"
```

## Troubleshooting

### Common Issues

**1. Endpoint fails to start**
- Check that the Docker image is accessible
- Verify environment variables are set correctly
- Check the endpoint logs for errors

**2. Authentication errors**
- Ensure `HF_TOKEN` has write access
- Verify the token is set in your environment

**3. Build failures**
- Ensure Docker is running
- Check that you're authenticated to your registry
- Try building with `--no-cache`

**4. Timeout errors**
- Increase the instance size
- Check if your workflow is too resource-intensive

### Getting Help

- Check the [HuggingFace documentation](https://huggingface.co/docs/inference-endpoints/)
- View endpoint logs in the HuggingFace dashboard
- Open an issue on the NodeTool GitHub repository

## Cost Considerations

HuggingFace Inference Endpoints pricing depends on:
- Instance type (CPU vs GPU)
- Instance size
- Running time
- Number of replicas

Tips to reduce costs:
- Use `min_replica: 0` to scale to zero when idle
- Choose the smallest instance that meets your needs
- Use CPU instances for non-ML workloads

## Security

### Environment Variables

Sensitive data should be passed as environment variables:
- API keys
- Database credentials
- Authentication tokens

### Access Control

HuggingFace Inference Endpoints support three access levels:
- **Protected**: Requires HF token authentication
- **Private**: Only accessible within your organization
- **Public**: Open access (use with caution)

The default is `protected`, requiring authentication for all requests.
