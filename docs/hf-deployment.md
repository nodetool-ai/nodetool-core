# Hugging Face Inference Endpoint Deployment

This guide explains how to deploy NodeTool workflows as Hugging Face Inference Endpoints.

```bash
nodetool deploy-hf \
  --workflow-id YOUR_WORKFLOW_ID \
  --endpoint-name my-endpoint \
  --repository username/my-repo \
  --docker-username mydockeruser
```

The command builds a Docker image containing the workflow and uploads it to your Docker registry. An Inference Endpoint is then created using that image.

The deployed container uses `hf_inference.py` which runs the workflow when Hugging Face calls the `predict` function.
