# Deployment

### RunPod serverless

- Basic: deploy a workflow by ID

```bash
nodetool deploy-runpod --workflow-id <workflow-id> --name my-workflow
```

- Multiple workflows

```bash
nodetool deploy-runpod \
  --workflow-id abc123 \
  --workflow-id def456 \
  --workflow-id ghi789 \
  --name multi-workflow
```

- Select GPUs and regions

```bash
nodetool deploy-runpod --workflow-id <workflow-id> \
  --gpu-types "NVIDIA GeForce RTX 4090" \
  --gpu-types "NVIDIA L40S" \
  --data-centers US-CA-2 \
  --data-centers US-GA-1 \
  --name gpu-workflow
```

- Scaling and startup

```bash
nodetool deploy-runpod --workflow-id <workflow-id> \
  --workers-min 0 \
  --workers-max 3 \
  --idle-timeout 60 \
  --flashboot \
  --name scaled-workflow
```

- Docker build options

```bash
nodetool deploy-runpod --workflow-id <workflow-id> \
  --docker-username <dockerhub-username> \
  --docker-registry docker.io \
  --tag v1 \
  --platform linux/amd64 \
  --name my-workflow
```

- Discover available options

```bash
nodetool deploy-runpod --list-gpu-types
nodetool deploy-runpod --list-cpu-flavors
nodetool deploy-runpod --list-data-centers
nodetool deploy-runpod --list-all-options
```

- Run locally in Docker (no cloud deploy)

```bash
nodetool deploy-runpod --workflow-id <workflow-id> --local-docker --name my-workflow
```

### Google Cloud Run

- Basic: deploy a workflow service

```bash
nodetool deploy-gcp --workflow-id <workflow-id> --service-name my-workflow
```

- Region and resources

```bash
nodetool deploy-gcp --workflow-id <workflow-id> \
  --service-name my-workflow \
  --region us-west1 \
  --cpu 2 \
  --memory 4Gi \
  --min-instances 0 \
  --max-instances 3 \
  --concurrency 80 \
  --timeout 3600
```

- Docker build options and local run

```bash
nodetool deploy-gcp --workflow-id <workflow-id> \
  --service-name my-workflow \
  --docker-username <dockerhub-username> \
  --docker-registry docker.io \
  --tag v1 \
  --platform linux/amd64

# Run locally in Docker
nodetool deploy-gcp --workflow-id <workflow-id> --service-name my-workflow --local-docker
```

### Test a deployed endpoint (RunPod)

```bash
# No parameters
nodetool test-runpod --endpoint-id <endpoint-id>

# With parameters from JSON file
nodetool test-runpod --endpoint-id <endpoint-id> --params params.json

# With inline JSON params
nodetool test-runpod --endpoint-id <endpoint-id> --params-json '{"text":"Hello World"}'
```

### Admin ops via HTTP API server

```bash
# Download a HuggingFace repo via API server
nodetool admin download-hf --repo-id microsoft/DialoGPT-small --server-url http://localhost:8000

# Download an Ollama model via API server
nodetool admin download-ollama --model-name llama3.2:latest --server-url http://localhost:8000

# Scan HF cache and show size via API server
nodetool admin scan-cache --server-url http://localhost:8000
nodetool admin cache-size --server-url http://localhost:8000
```
