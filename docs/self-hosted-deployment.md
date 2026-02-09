# Self-Hosted Deployment Guide

This guide covers deploying NodeTool on your own infrastructure.

## Overview

Self-hosted deployment supports:

1. **Docker**: Runs NodeTool in a container.
2. **SSH**: Installs and runs NodeTool as a `systemd` user service on a remote host.
3. **Local**: Same as SSH mode but on the local machine.

## Deployment Configuration

Deployments are configured via `deployment.yaml`.

### Docker Deployment

```yaml
deployments:
  my-server:
    type: docker
    host: 192.168.1.10
    ssh:
      user: ubuntu
      key_path: ~/.ssh/id_rsa
    container:
      name: nodetool-server
      port: 8000
      gpu: "0"
    paths:
      workspace: /data/nodetool
      hf_cache: /data/hf-cache
    image:
      name: ghcr.io/nodetool-ai/nodetool
      tag: latest
```

### SSH Deployment

```yaml
deployments:
  my-ssh-server:
    type: ssh
    host: 192.168.1.11
    ssh:
      user: ubuntu
      key_path: ~/.ssh/id_rsa
    port: 8000
    service_name: nodetool-8000
    paths:
      workspace: /home/ubuntu/nodetool
      hf_cache: /home/ubuntu/.cache/huggingface
```

### Local Deployment

```yaml
deployments:
  my-local-server:
    type: local
    host: localhost
    port: 8000
    service_name: nodetool-8000
```

## Apply Flow

### Docker Process

1. **Directory Creation**: Ensures `workspace` and `hf_cache` directories exist.
2. **Image Check**: Verifies the configured image exists locally/remote. `deploy apply` does not auto-pull.
3. **Image Transfer**: For remote hosts, copies image to remote runtime if needed.
4. **Container Management**: Restarts container with new configuration.
5. **Health Check**: Verifies HTTP endpoint.

### SSH/Local Process

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
