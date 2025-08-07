#!/usr/bin/env python3
"""
Docker utilities for NodeTool deployment.

This module contains all Docker-related functionality for building, pushing,
and managing Docker images for NodeTool workflows.
"""
import os
import subprocess
import sys
import hashlib
import time
import json
import tempfile
import shutil
from pathlib import Path
from typing import Optional


def run_command(command: str, capture_output: bool = False) -> str:
    """Run a shell command with streaming output and return output if requested."""
    print(f"Running: {command}")
    try:
        if capture_output:
            # For commands that need to return output, capture but still stream
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, check=True
            )
            output = result.stdout.strip()
            if output:
                print(output)
            if result.stderr:
                print(result.stderr.strip(), file=sys.stderr)
            return output
        else:
            # For commands that don't need return value, stream in real-time
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Stream output line by line
            if process.stdout:
                for line in process.stdout:
                    print(line.rstrip())
                    sys.stdout.flush()

            # Wait for process to complete and check return code
            return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)

            return ""
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}: {command}")
        sys.exit(1)


def check_docker_auth(registry: str = "docker.io") -> bool:
    """
    Check if user is authenticated with the Docker registry.

    Args:
        registry (str): The Docker registry URL

    Returns:
        bool: True if authenticated, False otherwise
    """
    try:
        # Try to get auth info for the registry
        result = subprocess.run(
            [
                "docker",
                "system",
                "info",
                "--format",
                "{{.RegistryConfig.IndexConfigs}}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        # If we can't determine auth status, try a simple auth check
        if result.returncode != 0:
            # Try to authenticate by making a simple request
            auth_check = subprocess.run(
                ["docker", "login", "--get-login"],
                capture_output=True,
                text=True,
                check=False,
            )
            return auth_check.returncode == 0

        return True
    except Exception:
        return False


def ensure_docker_auth(registry: str = "docker.io") -> None:
    """
    Ensure user is authenticated with Docker registry.

    Args:
        registry (str): The Docker registry URL

    Raises:
        SystemExit: If authentication fails or user chooses not to login
    """
    if not check_docker_auth(registry):
        print(f"\nYou are not authenticated with Docker registry: {registry}")
        print("You need to login to push images to the registry.")

        response = (
            input(f"Do you want to login to {registry} now? (y/n): ").lower().strip()
        )
        if response in ["y", "yes"]:
            try:
                if registry == "docker.io":
                    subprocess.run(["docker", "login"], check=True)
                else:
                    subprocess.run(["docker", "login", registry], check=True)
                print("Successfully authenticated with Docker registry.")
            except subprocess.CalledProcessError:
                print("Failed to authenticate with Docker registry.")
                sys.exit(1)
        else:
            print("Docker authentication is required to push images.")
            print("Please run 'docker login' manually and try again.")
            sys.exit(1)


def format_image_name(
    base_name: str, docker_username: str, registry: str = "docker.io"
) -> str:
    """
    Format the image name with proper registry and username prefix.

    Args:
        base_name (str): The base image name (e.g., "my-workflow")
        docker_username (str): Docker Hub username or organization
        registry (str): Docker registry URL

    Returns:
        str: Properly formatted image name

    Examples:
        format_image_name("my-workflow", "myuser") -> "myuser/my-workflow"
        format_image_name("my-workflow", "myuser", "ghcr.io") -> "ghcr.io/myuser/my-workflow"
    """
    if registry == "docker.io":
        # For Docker Hub, we don't need the registry prefix
        return f"{docker_username}/{base_name}"
    else:
        # For other registries, include the registry prefix
        return f"{registry}/{docker_username}/{base_name}"


def generate_image_tag() -> str:
    """
    Generate a unique image tag based on current timestamp and random hash.

    This ensures each deployment gets a unique Docker image tag, which is
    important for proper cache invalidation and deployment tracking.

    Returns:
        str: A unique tag in format 'YYYYMMDD-HHMMSS-abcdef'

    Example:
        '20231215-143052-a7b9c3'
    """
    # Get current timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Generate a short random hash
    random_data = f"{time.time()}{os.getpid()}{os.urandom(8).hex()}"
    hash_obj = hashlib.md5(random_data.encode())
    short_hash = hash_obj.hexdigest()[:6]

    return f"{timestamp}-{short_hash}"



def extract_models(workflow_data: dict) -> list[dict]:
    """
    Extract both Hugging Face and Ollama models from a workflow graph.

    Scans through all nodes in the workflow graph to find models that need to be
    pre-downloaded. This includes:
    - Hugging Face models (type starts with "hf.")
    - Ollama language models (type="language_model" and provider="ollama")

    Args:
        workflow_data (dict): The complete workflow data dictionary

    Returns:
        list[dict]: List of serialized model objects found in the workflow
    """
    models = []
    seen_models = set()  # Track unique models

    if "graph" not in workflow_data or "nodes" not in workflow_data["graph"]:
        return models

    for node in workflow_data["graph"]["nodes"]:
        if "data" not in node:
            continue

        node_data = node["data"]

        # Check for HuggingFace models (model field with type and repo_id)
        if "model" in node_data and isinstance(node_data["model"], dict):
            model = node_data["model"]

            # HuggingFace models
            if (
                "type" in model
                and model.get("type", "").startswith("hf.")
                and "repo_id" in model
                and model["repo_id"]
            ):
                # Create a unique key for this model
                model_key = (
                    "hf",
                    model.get("type"),
                    model.get("repo_id"),
                    model.get("path"),
                    model.get("variant"),
                )

                if model_key not in seen_models:
                    seen_models.add(model_key)
                    # Create a serialized HuggingFaceModel object
                    hf_model = {
                        "type": model.get("type", "hf.model"),
                        "repo_id": model["repo_id"],
                        "path": model.get("path"),
                        "variant": model.get("variant"),
                        "allow_patterns": model.get("allow_patterns"),
                        "ignore_patterns": model.get("ignore_patterns"),
                    }
                    models.append(hf_model)

            # Ollama language models
            elif (
                model.get("type") == "language_model"
                and model.get("provider") == "ollama"
                and model.get("id")
            ):
                model_key = ("ollama", model["id"])

                if model_key not in seen_models:
                    seen_models.add(model_key)
                    ollama_model = {
                        "type": "language_model",
                        "provider": "ollama",
                        "id": model["id"],
                    }
                    models.append(ollama_model)

        # Check for language models at the root level (some nodes might have them directly)
        if (
            node_data.get("type") == "language_model"
            and node_data.get("provider") == "ollama"
            and node_data.get("id")
        ):
            model_key = ("ollama", node_data["id"])

            if model_key not in seen_models:
                seen_models.add(model_key)
                ollama_model = {
                    "type": "language_model",
                    "provider": "ollama",
                    "id": node_data["id"],
                }
                models.append(ollama_model)

        # Check for nested model references (e.g., in arrays like loras)
        for key, value in node_data.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        # HuggingFace models in arrays
                        if (
                            "type" in item
                            and item.get("type", "").startswith("hf.")
                            and "repo_id" in item
                            and item["repo_id"]
                        ):
                            model_key = (
                                "hf",
                                item.get("type"),
                                item.get("repo_id"),
                                item.get("path"),
                                item.get("variant"),
                            )

                            if model_key not in seen_models:
                                seen_models.add(model_key)
                                hf_model = {
                                    "type": item.get("type", "hf.model"),
                                    "repo_id": item["repo_id"],
                                    "path": item.get("path"),
                                    "variant": item.get("variant"),
                                    "allow_patterns": item.get("allow_patterns"),
                                    "ignore_patterns": item.get("ignore_patterns"),
                                }
                                models.append(hf_model)

    return models


def build_docker_image(
    workflows_source: str,
    image_name: str,
    tag: str,
    platform: str = "linux/amd64",
    use_cache: bool = True,
    auto_push: bool = True,
) -> bool:
    """
    Build a Docker image for RunPod deployment with embedded workflow(s).

    Always expects workflows_source to be a directory containing workflow JSON files.

    This function creates a specialized Docker image by:
    1. Using the RunPod-specific Dockerfile from src/nodetool/deploy/
    2. Creating a temporary build directory with all necessary files
    3. Building the final image using Docker buildx with Docker Hub cache optimization

    The resulting image is self-contained and includes:
    - All NodeTool dependencies and runtime
    - Workflow(s) in /app/workflows/ directory
    - The fastapi_server.py configured as the entry point
    - start.sh script for proper RunPod initialization

    The image is built with --platform linux/amd64 to ensure compatibility with
    RunPod's Linux servers, regardless of the build machine's architecture.

    Args:
        workflows_source (str): Path to workflows directory
        image_name (str): Name of the Docker image (including registry/username)
        tag (str): Tag of the Docker image
        platform (str): Docker build platform (default: linux/amd64)
        use_cache (bool): Whether to use Docker Hub cache optimization (default: True)
        auto_push (bool): Whether to automatically push to registry during build (default: True)

    Returns:
        bool: True if the image was pushed to registry, False if only built locally

    Raises:
        SystemExit: If Docker build fails or required files are missing

    Note:
        Creates and cleans up temporary build directory during the process.
        When use_cache=True and auto_push=True, the image is automatically pushed.
    """
    print(f"Building Docker image with embedded workflow(s) from {workflows_source}")
    print(f"Platform: {platform}")

    # Get the deploy directory where Dockerfile, handlers, and scripts are located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    deploy_dockerfile_path = os.path.join(script_dir, "Dockerfile")
    start_script_path = os.path.join(script_dir, "start.sh")

    # Create a temporary build directory
    build_dir = tempfile.mkdtemp(prefix="nodetool_build_")
    print(f"Using build directory: {build_dir}")

    try:
        # Copy workflow files to build directory
        workflows_build_dir = os.path.join(build_dir, "workflows")
        shutil.copytree(workflows_source, workflows_build_dir)

        shutil.copy(deploy_dockerfile_path, os.path.join(build_dir, "Dockerfile"))
        shutil.copy(start_script_path, os.path.join(build_dir, "start.sh"))

        # Build with the Dockerfile from the build directory
        original_dir = os.getcwd()
        os.chdir(build_dir)
        image_pushed = False

        if use_cache:
            print("Building with Docker Hub cache optimization...")

            # Ensure docker buildx is available
            run_command(
                "docker buildx create --use --name nodetool-builder --driver docker-container || true"
            )

            # Try to build with cache from/to Docker Hub registry
            cache_from = f"--cache-from=type=registry,ref={image_name}:buildcache"
            cache_to = f"--cache-to=type=registry,ref={image_name}:buildcache,mode=max"

            push_flag = "--push" if auto_push else "--load"

            build_cmd_with_cache = (
                f"docker buildx build "
                f"--platform {platform} "
                f"-t {image_name}:{tag} "
                f"{cache_from} "
                f"{cache_to} "
                f"{push_flag} "
                f"."
            )

            print(f"Cache image: {image_name}:buildcache")

            try:
                # Try building with cache first
                run_command(build_cmd_with_cache)
                image_pushed = auto_push
            except subprocess.CalledProcessError:
                print(
                    "Cache build failed, falling back to build without cache import..."
                )
                # Fallback to build without cache import (but still export cache)
                build_cmd_fallback = (
                    f"docker buildx build "
                    f"--platform {platform} "
                    f"-t {image_name}:{tag} "
                    f"{cache_to} "
                    f"{push_flag} "
                    f"."
                )
                run_command(build_cmd_fallback)
                image_pushed = auto_push
        else:
            # Traditional docker build without cache optimization
            print("Building without cache optimization...")
            run_command(f"docker build --platform {platform} -t {image_name}:{tag} .")
            image_pushed = False

        os.chdir(original_dir)

        print("Docker image built successfully")
        return image_pushed
    finally:
        # Clean up temporary build directory
        shutil.rmtree(build_dir, ignore_errors=True)


def push_to_registry(image_name: str, tag: str, registry: str = "docker.io"):
    """
    Push Docker image to a registry with proper authentication checks.

    Args:
        image_name (str): Full image name including registry/username
        tag (str): Image tag
        registry (str): Docker registry URL

    Raises:
        SystemExit: If push fails or authentication is not set up
    """
    print(f"Pushing Docker image {image_name}:{tag} to registry {registry}...")

    # Ensure we're authenticated with the registry
    ensure_docker_auth(registry)

    # Push the image
    try:
        run_command(f"docker push {image_name}:{tag}")
        print(f"Docker image {image_name}:{tag} pushed successfully")

    except subprocess.CalledProcessError:
        print(f"Failed to push image {image_name}:{tag}")
        print("Common issues:")
        print("1. Check your Docker registry authentication: docker login")
        print("2. Verify the image name includes your username: username/image-name")
        print("3. Ensure you have push permissions to the repository")
        sys.exit(1)


def get_docker_username_from_config(registry: str = "docker.io") -> str | None:
    """
    Get Docker username from Docker's configuration file.

    Docker stores authentication information in ~/.docker/config.json.
    This function attempts to extract the username from the stored auth data.

    Args:
        registry (str): The Docker registry URL

    Returns:
        str | None: The Docker username if found, None otherwise
    """
    import json
    import base64

    try:
        # Docker config is typically stored in ~/.docker/config.json
        docker_config_path = Path.home() / ".docker" / "config.json"

        if not docker_config_path.exists():
            return None

        with open(docker_config_path, "r") as f:
            config = json.load(f)

        # Check for authentication data
        auths = config.get("auths", {})

        # Docker Hub can be referenced as docker.io, index.docker.io, or https://index.docker.io/v1/
        possible_registry_keys = [
            registry,
            f"https://{registry}/v1/",
            (
                "https://index.docker.io/v1/"
                if registry == "docker.io"
                else f"https://{registry}/v1/"
            ),
            "index.docker.io" if registry == "docker.io" else registry,
        ]

        for reg_key in possible_registry_keys:
            if reg_key in auths:
                auth_data = auths[reg_key]

                # Check if there's a direct username field
                if "username" in auth_data:
                    return auth_data["username"]

                # Check if there's base64 encoded auth data
                if "auth" in auth_data:
                    try:
                        # Decode base64 auth string (format: username:password)
                        auth_str = base64.b64decode(auth_data["auth"]).decode("utf-8")
                        username, _ = auth_str.split(":", 1)
                        return username
                    except Exception:
                        continue

        # Check credential helpers
        cred_helpers = config.get("credHelpers", {})
        if registry in cred_helpers:
            # For credential helpers, we can't easily get the username
            # but we can try to run the helper (this is more complex)
            pass

        return None

    except Exception as e:
        print(f"Warning: Could not read Docker config: {e}")
        return None
