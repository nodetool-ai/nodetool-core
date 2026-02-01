# Self-Hosted Deployment Guide

This guide covers deploying NodeTool on your own infrastructure.

## Overview

Self-hosted deployment supports two modes:

1.  **Docker (Default)**: Runs the application in a Docker container. Recommended for isolation and ease of management.
2.  **Shell**: Installs dependencies (Python, ffmpeg, etc.) using `micromamba` and `uv` directly on the host, and manages the service via `systemd`. Useful for bare-metal performance or environments where Docker is not available.

## Deployment Configuration

Deployments are configured via `deployment.yaml`.

### Docker Mode

```yaml
deployments:
  my-server:
    type: self-hosted
    mode: docker  # Optional (default)
    host: 192.168.1.10
    ssh:
      user: ubuntu
      key_path: ~/.ssh/id_rsa
    container:
      name: nodetool-server
      port: 8000
      gpu: "all"
    paths:
      workspace: /data/nodetool
      hf_cache: /data/hf-cache
    image:
      name: nodetool/nodetool
      tag: latest
```

### Shell Mode

```yaml
deployments:
  my-metal-server:
    type: self-hosted
    mode: shell
    host: 192.168.1.11
    ssh:
      user: ubuntu
      key_path: ~/.ssh/id_rsa
    container:
      name: server-01  # Used for systemd service name (nodetool-server-01)
      port: 8000
    paths:
      workspace: /home/ubuntu/nodetool
      hf_cache: /home/ubuntu/.cache/huggingface
    image:
      name: nodetool/nodetool # Ignored in shell mode, but required by schema
```

## Deployment Process

### Docker Process

1. **Directory Creation**: Ensures `workspace` and `hf_cache` directories exist.
2. **Image Transfer**: Pushes/pulls Docker image.
3. **Container Management**: Restarts container with new configuration.
4. **Health Check**: Verifies HTTP endpoint.

### Shell Process

1. **Directory Creation**: Creates workspace and environment directories.
2. **Micromamba Setup**: Downloads and installs `micromamba` locally in the workspace if missing.
3. **Environment Creation**: Creates a Conda environment with system dependencies (ffmpeg, etc.).
4. **Package Installation**: Installs `nodetool-core` and `nodetool-base` using `uv`.
5. **Service Management**: Creates and enables a user-level `systemd` service (`nodetool-<name>.service`).
6. **Health Check**: Verifies HTTP endpoint.

## Manual Troubleshooting

### Docker Logs

```bash
ssh user@host "docker logs nodetool-server"
```

### Shell Logs

```bash
ssh user@host "journalctl --user -u nodetool-server-01 -f"
```
