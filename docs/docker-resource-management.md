# Docker Resource Management

This guide explains how to configure and manage resource constraints for Docker-based job execution in multi-user production deployments.

## Overview

In a multi-user AI workflow engine, it's crucial to prevent individual jobs from monopolizing system resources. Nodetool's Docker job execution supports fine-grained resource constraints for:

- **CPU**: Limit CPU cores per job
- **Memory (RAM)**: Limit system memory per job
- **GPU/VRAM**: Allocate specific GPUs and limit VRAM usage per job

## Configuration

### Environment Variables (Recommended)

Set default limits for all Docker jobs:

```bash
# Memory limit (default: 2g)
export DOCKER_MEM_LIMIT="4g"

# CPU limit in cores (default: 2.0)
export DOCKER_CPU_LIMIT="2.0"

# GPU device IDs (comma-separated, empty = no GPUs)
export DOCKER_GPU_DEVICES="0,1"

# GPU memory limit per device
export DOCKER_GPU_MEMORY_LIMIT="8g"
```

### Programmatic Configuration

Override defaults per-job:

```python
from nodetool.workflows.docker_job_execution import DockerJobExecution

job = await DockerJobExecution.create_and_start(
    request=run_job_request,
    context=processing_context,
    mem_limit="4g",          # 4GB RAM
    cpu_limit=2.0,           # 2 CPU cores
    gpu_device_ids=[0],      # Use only GPU 0
    gpu_memory_limit="8g",   # 8GB VRAM per GPU
)
```

## Resource Limit Strategies

### Single GPU Server

For a server with one GPU, use a queue-based approach to prevent concurrent GPU jobs:

```bash
# Only allow CPU jobs by default
export DOCKER_GPU_DEVICES=""

# Allocate GPU to specific jobs via API/scheduler
# Jobs request GPU access and wait in queue
```

### Multi-GPU Server

Distribute GPUs across concurrent jobs:

```bash
# Example: 4 GPUs available, run up to 4 concurrent jobs

# Job 1 gets GPU 0
DOCKER_GPU_DEVICES="0" nodetool run job1.json

# Job 2 gets GPU 1
DOCKER_GPU_DEVICES="1" nodetool run job2.json

# Job 3 gets GPUs 2,3
DOCKER_GPU_DEVICES="2,3" nodetool run job3.json
```

### Mixed CPU/GPU Workloads

```bash
# CPU-only jobs (lightweight)
export DOCKER_MEM_LIMIT="1g"
export DOCKER_CPU_LIMIT="1.0"
export DOCKER_GPU_DEVICES=""  # No GPU

# GPU jobs (resource-intensive)
export DOCKER_MEM_LIMIT="8g"
export DOCKER_CPU_LIMIT="4.0"
export DOCKER_GPU_DEVICES="0"
export DOCKER_GPU_MEMORY_LIMIT="12g"
```

## Multi-User Scheduling Patterns (NOT IMPLEMTED YET)

### 1. Fair Share Scheduling

Simple round-robin GPU allocation:

```python
# Simple round-robin GPU allocation
gpu_queue = [0, 1, 2, 3]  # 4 GPUs
current_gpu = 0

async def schedule_job(request):
    gpu_id = gpu_queue[current_gpu % len(gpu_queue)]
    current_gpu += 1

    return await DockerJobExecution.create_and_start(
        request=request,
        context=context,
        mem_limit="4g",
        cpu_limit="2.0",
        gpu_device_ids=[gpu_id],
        gpu_memory_limit="10g"
    )
```

### 2. Priority-Based Scheduling

Allocate resources based on user tier:

````python
# Higher priority = more resources
RESOURCE_TIERS = {
    "premium": {
        "mem_limit": "16g",
        "cpu_limit": 8.0,
        "gpu_device_ids": [0, 1],  # 2 GPUs
        "gpu_memory_limit": "24g"
    },
    "standard": {
        "mem_limit": "8g",
        "cpu_limit": 4.0,
        "gpu_device_ids": [2],  # 1 GPU
        "gpu_memory_limit": "12g"
    },
    "free": {
        "mem_limit": "2g",
        "cpu_limit": 1.0,
        "gpu_device_ids": [],  # No GPU
        "gpu_memory_limit": None
    }
}

### 3. Queue-Based GPU Access

Jobs wait for available GPUs:

```python
import asyncio
from collections import deque

class GPUScheduler:
    def __init__(self, num_gpus=4):
        self.gpu_queues = {i: deque() for i in range(num_gpus)}
        self.gpu_available = {i: True for i in range(num_gpus)}

    async def request_gpu(self, gpu_id=None):
        """Request a GPU, wait if not available."""
        if gpu_id is None:
            # Find first available GPU
            gpu_id = next((i for i, avail in self.gpu_available.items() if avail), None)

        if gpu_id is None or not self.gpu_available[gpu_id]:
            # Wait in queue
            future = asyncio.Future()
            self.gpu_queues[gpu_id or 0].append(future)
            gpu_id = await future

        self.gpu_available[gpu_id] = False
        return gpu_id

    def release_gpu(self, gpu_id):
        """Release GPU and assign to next job in queue."""
        if self.gpu_queues[gpu_id]:
            # Give to next waiting job
            future = self.gpu_queues[gpu_id].popleft()
            future.set_result(gpu_id)
        else:
            self.gpu_available[gpu_id] = True

# Usage
scheduler = GPUScheduler(num_gpus=4)

async def run_job_with_gpu(request):
    gpu_id = await scheduler.request_gpu()
    try:
        job = await DockerJobExecution.create_and_start(
            request=request,
            context=context,
            gpu_device_ids=[gpu_id],
            gpu_memory_limit="12g"
        )
        # Wait for job completion
        while job.is_running():
            await asyncio.sleep(1)
    finally:
        scheduler.release_gpu(gpu_id)
````

## Production Scheduler Example

Complete production-ready scheduler:

```python
class ProductionScheduler:
    def __init__(self):
        self.max_concurrent_jobs = 10
        self.max_gpu_jobs = 4
        self.active_jobs = 0
        self.active_gpu_jobs = 0
        self.job_queue = asyncio.Queue()
        self.gpu_pool = GPUPool(gpus=[0, 1, 2, 3])

    async def submit_job(self, request, user_tier="free"):
        """Submit job and wait for resources."""
        await self.job_queue.put((request, user_tier))

    async def scheduler_worker(self):
        """Background worker that processes queued jobs."""
        while True:
            # Wait for queued job
            request, user_tier = await self.job_queue.get()

            # Wait for slot
            while self.active_jobs >= self.max_concurrent_jobs:
                await asyncio.sleep(1)

            # Get resource allocation
            limits = self.get_limits_for_tier(user_tier)

            # Allocate GPU if needed
            if limits["gpu_device_ids"]:
                while self.active_gpu_jobs >= self.max_gpu_jobs:
                    await asyncio.sleep(1)
                gpu_id = await self.gpu_pool.acquire()
                limits["gpu_device_ids"] = [gpu_id]
                self.active_gpu_jobs += 1

            self.active_jobs += 1

            # Start job
            asyncio.create_task(self.run_job(request, limits))

    async def run_job(self, request, limits):
        """Run job with allocated resources."""
        try:
            job = await DockerJobExecution.create_and_start(
                request=request,
                context=ProcessingContext(),
                **limits
            )

            # Wait for completion
            while job.is_running():
                await asyncio.sleep(1)

        finally:
            # Release resources
            self.active_jobs -= 1
            if limits.get("gpu_device_ids"):
                await self.gpu_pool.release(limits["gpu_device_ids"][0])
                self.active_gpu_jobs -= 1
```

## Docker GPU Configuration

### Enable GPU Support

Ensure Docker has GPU support enabled (requires nvidia-docker):

```bash
# Check Docker can see GPUs
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Test with nodetool image
docker run --rm --gpus all nodetool nvidia-smi
```

### Hard vs Soft Limits

- **CPU/Memory**: Docker enforces these as **hard limits**
- **GPU Memory**: PyTorch/TensorFlow respect environment variables as **soft limits**

For stricter GPU memory control, consider:

1. **MPS (Multi-Process Service)**: NVIDIA's GPU sharing mechanism
2. **MIG (Multi-Instance GPU)**: Partition A100/H100 GPUs into isolated instances
3. **Kubernetes**: More advanced resource management and scheduling

## Production Configuration Example

```bash
# /etc/nodetool/environment

# Default limits for all jobs
export DOCKER_MEM_LIMIT="4g"
export DOCKER_CPU_LIMIT="2.0"
export DOCKER_GPU_DEVICES=""  # Disabled by default

# GPU allocation handled by scheduler
# Scheduler overrides these per-job based on:
# - User tier (free/standard/premium)
# - Current GPU availability
# - Job priority/queue position
```

## Best Practices

1. **Start Conservative**: Begin with lower limits and increase based on monitoring
2. **Monitor Metrics**: Track resource usage, queue lengths, job completion times
3. **User Feedback**: Log resource constraints so users understand limits
4. **Graceful Degradation**: Fall back to CPU when GPUs unavailable
5. **Cost Tracking**: Associate resource usage with user accounts for billing
6. **Auto-scaling**: Consider cloud auto-scaling for elastic capacity

## Troubleshooting

### Job OOM (Out of Memory)

```bash
# Increase memory limit
export DOCKER_MEM_LIMIT="8g"
```

### CUDA Out of Memory

```bash
# Reduce GPU memory allocation
export DOCKER_GPU_MEMORY_LIMIT="4g"

# Or use smaller batch sizes in workflow
```

### GPU Not Found

```bash
# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Verify nvidia-docker installed
dpkg -l | grep nvidia-docker
```

### Performance Issues

```bash
# Monitor container stats
docker stats

# Check for CPU throttling
docker inspect <container_id> | grep -A 5 "CpuStats"

# Check for memory pressure
docker inspect <container_id> | grep -A 5 "MemoryStats"
```

## Related Documentation

- [Docker Execution Guide](docker-execution.md) - Main documentation
- [Docker Testing Guide](docker-testing.md) - Testing and debugging
- [Deployment Guide](deployment.md) - Production deployment
- [Docker Resource Constraints](https://docs.docker.com/config/containers/resource_constraints/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
