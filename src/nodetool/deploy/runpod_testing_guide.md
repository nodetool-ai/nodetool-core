# RunPod Workflow Testing Guide

This guide explains how to test your deployed NodeTool workflows on RunPod serverless infrastructure.

## Prerequisites

1. **Deployed Workflow**: You must have already deployed a workflow using `deploy_to_runpod.py`
2. **RunPod API Key**: Get this from your [RunPod account settings](https://www.runpod.io/console/user/settings)
3. **Endpoint ID**: This is returned when you deploy your workflow

**Note**: When deploying workflows, Docker images are built with `--platform linux/amd64` to ensure compatibility with RunPod's Linux servers. This may take longer on ARM-based systems (Apple Silicon) due to cross-platform emulation.

**Template Management**: The deployment script automatically deletes existing RunPod templates with the same name before creating new ones. Use `--no-force-recreate` if you want to fail instead of overwriting existing templates.

## Quick Start

### 1. Test Connectivity

First, verify that your endpoint is reachable:

```bash
python test_runpod_workflow.py --endpoint-id YOUR_ENDPOINT_ID --test-connectivity
```

### 2. Basic Test with No Parameters

Test a workflow that doesn't require specific input parameters:

```bash
python test_runpod_workflow.py --endpoint-id YOUR_ENDPOINT_ID
```

### 3. Test with Inline Parameters

Pass parameters directly via command line:

```bash
python test_runpod_workflow.py \
  --endpoint-id YOUR_ENDPOINT_ID \
  --params-json '{"text": "Hello World", "count": 3}'
```

### 4. Test with Parameter File

Create a JSON file with your parameters and reference it:

```bash
python test_runpod_workflow.py \
  --endpoint-id YOUR_ENDPOINT_ID \
  --params examples/test_params_basic.json
```

## Environment Setup

Set your RunPod API key as an environment variable:

```bash
export RUNPOD_API_KEY="your-api-key-here"
```

Or pass it directly:

```bash
python test_runpod_workflow.py \
  --endpoint-id YOUR_ENDPOINT_ID \
  --api-key YOUR_API_KEY
```

## Parameter Examples

### Basic Workflow Parameters

For simple workflows (see `examples/test_params_basic.json`):

```json
{
  "text": "Hello, NodeTool workflow!",
  "count": 5,
  "enabled": true
}
```

### Image Generation Parameters

For image generation workflows (see `examples/test_params_image.json`):

```json
{
  "prompt": "A beautiful sunset over a mountain landscape, digital art",
  "width": 512,
  "height": 512,
  "num_inference_steps": 20,
  "guidance_scale": 7.5,
  "seed": 42
}
```

### Complex Workflow Parameters

For workflows with multiple inputs:

```json
{
  "image_url": "https://example.com/image.jpg",
  "text_input": "Analyze this image",
  "options": {
    "temperature": 0.7,
    "max_tokens": 150
  },
  "output_format": "json"
}
```

## Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--endpoint-id` | RunPod endpoint ID (required) | `--endpoint-id abc123def456` |
| `--api-key` | RunPod API key | `--api-key your-key-here` |
| `--params` | JSON file with parameters | `--params test_params.json` |
| `--params-json` | Inline JSON parameters | `--params-json '{"key": "value"}'` |
| `--output` | Output file for results | `--output results.json` |
| `--timeout` | Timeout in minutes | `--timeout 15` |
| `--poll-interval` | Status check interval (seconds) | `--poll-interval 3` |
| `--test-connectivity` | Only test connectivity | `--test-connectivity` |

## Understanding Results

### Successful Execution

```
🧪 Testing RunPod workflow...
Endpoint ID: abc123def456
Timeout: 10 minutes
🔍 Testing connectivity to endpoint abc123def456...
✅ Endpoint is reachable
🚀 Starting workflow execution...
Endpoint: https://api.runpod.ai/v2/abc123def456/run
Parameters: {
  "text": "Hello World"
}
✅ Job started successfully!
Job ID: 12345-67890-abcdef
Status: IN_QUEUE
⏳ Waiting for job completion (timeout: 10 minutes)...
Status: IN_PROGRESS
Status: COMPLETED
✅ Job completed successfully!
Execution time: 2500ms
Delay time: 100ms

📊 Job Results:
Job ID: 12345-67890-abcdef
Status: COMPLETED
Execution Time: 2500ms
Delay Time: 100ms

🎯 Workflow Output:
  result: "Hello World processed successfully"
  output_url: "https://temp-bucket.s3.amazonaws.com/result.jpg"

💾 Results saved to: runpod_result_20241220_143022.json
✅ Test completed successfully!
```

### Error Handling

The script handles various error scenarios:

- **Authentication failures**: Invalid API key
- **Network issues**: Connection timeouts
- **Workflow errors**: Runtime failures in your workflow
- **Timeouts**: Jobs that take too long to complete

## Troubleshooting

### Common Issues

1. **401 Unauthorized**
   - Check your RunPod API key
   - Verify the key has access to the endpoint

2. **404 Not Found**
   - Verify the endpoint ID is correct
   - Check that the endpoint is still active

3. **Timeout**
   - Increase timeout with `--timeout 20`
   - Check RunPod console for endpoint status

4. **Workflow Errors**
   - Review the error details in the output
   - Check that your workflow parameters match what the workflow expects

5. **Platform/Architecture Issues**
   - On Apple Silicon Macs: Docker build may be slower due to `--platform linux/amd64` emulation
   - If build fails with platform errors, ensure Docker Desktop has cross-platform builds enabled
       - For faster builds on ARM systems, consider using a cloud build service or Linux AMD64 machine
    - Advanced users can override platform with `--platform` flag, but `linux/amd64` is required for RunPod

### Debugging Tips

1. **Test connectivity first**:
   ```bash
   python test_runpod_workflow.py --endpoint-id YOUR_ID --test-connectivity
   ```

2. **Use shorter timeouts for debugging**:
   ```bash
   python test_runpod_workflow.py --endpoint-id YOUR_ID --timeout 2
   ```

3. **Save results for analysis**:
   ```bash
   python test_runpod_workflow.py --endpoint-id YOUR_ID --output debug_results.json
   ```

4. **Check the RunPod console**:
   - Visit [RunPod Serverless Console](https://www.runpod.io/console/serverless)
   - Monitor your endpoint logs and metrics

## Advanced Usage

### Automated Testing

Create a test script that runs multiple test cases:

```bash
#!/bin/bash
ENDPOINT_ID="your-endpoint-id"

echo "Testing basic parameters..."
python test_runpod_workflow.py --endpoint-id $ENDPOINT_ID --params examples/test_params_basic.json

echo "Testing image generation..."
python test_runpod_workflow.py --endpoint-id $ENDPOINT_ID --params examples/test_params_image.json

echo "Testing with custom output..."
python test_runpod_workflow.py --endpoint-id $ENDPOINT_ID --params custom_params.json --output custom_results.json
```

### Performance Testing

Test with different timeout and polling settings:

```bash
# Quick polling for fast workflows
python test_runpod_workflow.py --endpoint-id YOUR_ID --poll-interval 1 --timeout 5

# Longer timeout for complex workflows
python test_runpod_workflow.py --endpoint-id YOUR_ID --poll-interval 10 --timeout 30
```

## Integration with CI/CD

You can integrate the test script into your CI/CD pipeline:

```yaml
# GitHub Actions example
name: Test RunPod Deployment
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Test deployed workflow
        env:
          RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}
        run: |
          python test_runpod_workflow.py \
            --endpoint-id ${{ secrets.ENDPOINT_ID }} \
            --params examples/test_params_basic.json \
            --timeout 5
```

## Next Steps

1. **Monitor Performance**: Use RunPod console to monitor your endpoint's performance
2. **Scale Testing**: Test with multiple concurrent requests
3. **Update Workflows**: Use deployment script to update your workflows
4. **Cost Optimization**: Adjust auto-scaling settings based on usage patterns

For more information, see:
- [RunPod Serverless Documentation](https://docs.runpod.io/serverless/overview)
- [NodeTool Documentation](https://docs.nodetool.ai/) 