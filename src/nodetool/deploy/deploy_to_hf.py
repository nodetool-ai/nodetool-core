#!/usr/bin/env python3
"""Deploy NodeTool workflows to Hugging Face Inference Endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, Optional

from huggingface_hub import HfApi


def run_command(command: str, capture_output: bool = False) -> str:
    """Run a shell command and optionally capture output."""
    print(f"Running: {command}")
    try:
        if capture_output:
            result = subprocess.run(
                command, shell=True, check=True, capture_output=True, text=True
            )
            if result.stdout:
                print(result.stdout.strip())
            if result.stderr:
                print(result.stderr.strip(), file=sys.stderr)
            return result.stdout.strip()
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if process.stdout:
            for line in process.stdout:
                print(line.rstrip())
        returncode = process.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)
        return ""
    except subprocess.CalledProcessError as e:
        print(f"Command failed with code {e.returncode}: {e.cmd}")
        sys.exit(1)


def format_image_name(
    base_name: str, docker_username: str, registry: str = "docker.io"
) -> str:
    if registry == "docker.io":
        return f"{docker_username}/{base_name}"
    return f"{registry}/{docker_username}/{base_name}"


def sanitize_name(name: str) -> str:
    import re

    sanitized = re.sub(r"[^a-zA-Z0-9\-]", "-", name.lower())
    sanitized = re.sub(r"-+", "-", sanitized).strip("-")
    return sanitized or "workflow"


def generate_image_tag() -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    random_data = f"{time.time()}{os.getpid()}{os.urandom(8).hex()}"
    short_hash = hashlib.md5(random_data.encode()).hexdigest()[:6]
    return f"{timestamp}-{short_hash}"


def extract_models(workflow_data: Dict) -> list[Dict]:
    models: list[Dict] = []
    if "graph" not in workflow_data or "nodes" not in workflow_data["graph"]:
        return models
    for node in workflow_data["graph"]["nodes"]:
        if "data" not in node:
            continue
        model = node["data"].get("model")
        if isinstance(model, dict) and model.get("type", "").startswith("hf."):
            if model.get("repo_id"):
                models.append(
                    {
                        "type": model.get("type"),
                        "repo_id": model["repo_id"],
                        "path": model.get("path"),
                        "variant": model.get("variant"),
                        "allow_patterns": model.get("allow_patterns"),
                        "ignore_patterns": model.get("ignore_patterns"),
                    }
                )
    return models


def create_ollama_pull_script(models: list[Dict], build_dir: str) -> None:
    from pathlib import Path

    ollama_models = [
        m
        for m in models
        if m.get("type") == "language_model" and m.get("provider") == "ollama"
    ]
    script_path = Path(build_dir) / "pull_ollama_models.sh"
    if not ollama_models:
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\necho 'No Ollama models to pull'\n")
        script_path.chmod(0o755)
        return

    lines = ["#!/bin/bash", "set -e", "ollama serve &", "PID=$!", "sleep 5"]
    for model in ollama_models:
        if model.get("id"):
            lines.append(f"ollama pull {model['id']}")
    lines.append("kill $PID")
    with open(script_path, "w") as f:
        f.write("\n".join(lines))
    script_path.chmod(0o755)


def build_docker_image(
    workflow_path: str, image_name: str, tag: str, platform: str = "linux/amd64"
) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dockerfile_path = os.path.join(script_dir, "Dockerfile.hf")
    inference_path = os.path.join(script_dir, "hf_inference.py")
    download_models_path = os.path.join(script_dir, "download_models.py")

    with open(workflow_path, "r") as f:
        workflow_data = json.load(f)
    models = extract_models(workflow_data)

    build_dir = tempfile.mkdtemp(prefix="nodetool_hf_build_")
    try:
        shutil.copy(workflow_path, os.path.join(build_dir, "workflow.json"))
        shutil.copy(inference_path, os.path.join(build_dir, "hf_inference.py"))
        shutil.copy(dockerfile_path, os.path.join(build_dir, "Dockerfile"))
        shutil.copy(download_models_path, os.path.join(build_dir, "download_models.py"))
        with open(os.path.join(build_dir, "models.json"), "w") as f:
            json.dump(models, f)
        create_ollama_pull_script(models, build_dir)
        cwd = os.getcwd()
        os.chdir(build_dir)
        run_command(f"docker build --platform {platform} -t {image_name}:{tag} .")
        os.chdir(cwd)
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)


def push_to_registry(image_name: str, tag: str, registry: str = "docker.io") -> None:
    run_command(f"docker push {image_name}:{tag}")


def fetch_workflow_from_db(workflow_id: str) -> tuple[str, str]:
    import tempfile
    from nodetool.models.workflow import Workflow

    workflow = Workflow.get(workflow_id)
    if not workflow:
        print(f"Workflow {workflow_id} not found")
        sys.exit(1)
    fd, path = tempfile.mkstemp(suffix=".json", prefix="workflow_")
    with os.fdopen(fd, "w") as f:
        f.write(workflow.model_dump_json())
    return path, workflow.name


def create_hf_endpoint(
    name: str,
    repository: str,
    image: str,
    accelerator: str,
    instance_size: str,
    instance_type: str,
    region: str,
    vendor: str,
    token: Optional[str] = None,
) -> None:
    api = HfApi(token=token)
    endpoint = api.create_inference_endpoint(
        name=name,
        repository=repository,
        framework="custom",
        accelerator=accelerator,
        instance_size=instance_size,
        instance_type=instance_type,
        region=region,
        vendor=vendor,
        custom_image={"url": image},
    )
    print(f"Endpoint {endpoint.name} created with status {endpoint.status}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy workflow to Hugging Face Inference Endpoints"
    )
    parser.add_argument("--workflow-id", required=True, help="Workflow ID to deploy")
    parser.add_argument("--endpoint-name", required=True, help="Name of the endpoint")
    parser.add_argument(
        "--repository", required=True, help="Hugging Face repository ID"
    )
    parser.add_argument(
        "--docker-username", required=True, help="Docker registry username"
    )
    parser.add_argument(
        "--docker-registry", default="docker.io", help="Docker registry"
    )
    parser.add_argument("--image-name", help="Docker image base name")
    parser.add_argument("--tag", help="Docker image tag")
    parser.add_argument(
        "--platform", default="linux/amd64", help="Docker build platform"
    )
    parser.add_argument("--accelerator", default="gpu", help="Compute accelerator")
    parser.add_argument("--instance-size", default="small", help="Instance size")
    parser.add_argument("--instance-type", default="nvidia-a10g", help="Instance type")
    parser.add_argument("--region", default="us-east-1", help="Cloud region")
    parser.add_argument("--vendor", default="aws", help="Cloud vendor")
    parser.add_argument("--token", help="Hugging Face access token")

    args = parser.parse_args()

    workflow_path, workflow_name = fetch_workflow_from_db(args.workflow_id)
    image_base = sanitize_name(args.image_name or workflow_name)
    full_image = format_image_name(
        image_base, args.docker_username, args.docker_registry
    )
    tag = args.tag or generate_image_tag()

    build_docker_image(workflow_path, full_image, tag, args.platform)
    push_to_registry(full_image, tag, args.docker_registry)

    image_with_tag = f"{full_image}:{tag}"
    create_hf_endpoint(
        name=args.endpoint_name,
        repository=args.repository,
        image=image_with_tag,
        accelerator=args.accelerator,
        instance_size=args.instance_size,
        instance_type=args.instance_type,
        region=args.region,
        vendor=args.vendor,
        token=args.token,
    )

    os.unlink(workflow_path)


if __name__ == "__main__":
    main()
