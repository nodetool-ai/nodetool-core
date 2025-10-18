#!/usr/bin/env python3
"""
Docker utilities for NodeTool deployment.

This module contains all Docker-related functionality for building, pushing,
and managing Docker images for NodeTool deployments.
"""

import os
import shlex
import subprocess
import sys
import hashlib
import time
import tempfile
import shutil
from pathlib import Path
import json

# Note: no need for urllib.parse after removing editable support
from importlib import metadata as importlib_metadata


def run_command(command: str, capture_output: bool = False) -> str:
    """Run a shell command with streaming output and return output if requested."""
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


def build_docker_image(
    image_name: str,
    tag: str,
    platform: str = "linux/amd64",
    use_cache: bool = True,
    auto_push: bool = True,
) -> bool:
    """
    Build a Docker image for deployment.

    This function creates a Docker image by:
    1. Using the Dockerfile from src/nodetool/deploy/
    2. Creating a temporary build directory with necessary files
    3. Building the final image using Docker buildx with optional cache optimization

    The image is built with --platform linux/amd64 to ensure compatibility with
    RunPod's Linux servers, regardless of the build machine's architecture.

    Args:
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
    print("Building Docker image")
    print(f"Platform: {platform}")

    # Get the deploy directory where Dockerfile, handlers, and scripts are located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    deploy_dockerfile_path = os.path.join(script_dir, "Dockerfile")
    start_script_path = os.path.join(script_dir, "start.sh")

    # Create a temporary build directory
    build_dir = tempfile.mkdtemp(prefix="nodetool_build_")
    print(f"Using build directory: {build_dir}")

    try:
        shutil.copy(deploy_dockerfile_path, os.path.join(build_dir, "Dockerfile"))
        shutil.copy(start_script_path, os.path.join(build_dir, "start.sh"))

        # Discover installed nodetool-* packages from the current environment
        def _read_direct_url_info(dist: importlib_metadata.Distribution) -> dict | None:
            try:
                files = list(dist.files or [])
            except Exception:
                files = []
            direct_url_rel = None
            for file_path in files:
                # importlib.metadata returns PackagePath objects
                if str(file_path).endswith(".dist-info/direct_url.json"):
                    direct_url_rel = file_path
                    break
            if direct_url_rel is None:
                # Try a fallback: locate the .dist-info directory and probe for direct_url.json
                dist_info_dir = None
                for file_path in files:
                    path_str = str(file_path)
                    if path_str.endswith(".dist-info/METADATA"):
                        dist_info_dir = Path(path_str).parent
                        break
                if dist_info_dir is None:
                    return None
                direct_url_rel = dist_info_dir / "direct_url.json"
            try:
                direct_url_abs = dist.locate_file(direct_url_rel)
                if not os.path.exists(str(direct_url_abs)):
                    return None
                with open(str(direct_url_abs), "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None

        def _discover_nodetool_packages() -> list[dict]:
            discovered: list[dict] = []
            for dist in importlib_metadata.distributions():
                try:
                    raw_name = (
                        dist.metadata.get("Name") or dist.metadata.get("Summary") or ""
                    )
                except Exception:
                    continue
                if not raw_name:
                    continue
                name_normalized = raw_name.strip()
                if not name_normalized.lower().startswith("nodetool-"):
                    continue

                info: dict[str, str] = {"name": name_normalized}

                direct_info = _read_direct_url_info(dist)
                if direct_info:
                    url = direct_info.get("url")
                    # Installed from VCS or URL; if local file/editable, ignore and fall back
                    if isinstance(url, str) and url and not url.startswith("file:"):
                        # Prefix with git+ if it looks like a Git URL and not already prefixed
                        install_url = url
                        if install_url.startswith(
                            "https://github.com/"
                        ) and not install_url.startswith("git+"):
                            install_url = f"git+{install_url}"
                        info["install_url"] = install_url
                        discovered.append(info)
                        continue

                # Fallback to canonical GitHub repo under nodetool-ai org
                info["install_url"] = (
                    f"git+https://github.com/nodetool-ai/{name_normalized}"
                )
                discovered.append(info)

            # Stable order for reproducible Dockerfiles
            discovered.sort(key=lambda d: str(d.get("name")).lower())
            return discovered

        nodetool_packages = _discover_nodetool_packages()

        # Inject a cache-optimized install block for nodetool packages into Dockerfile (before CMD)
        if nodetool_packages:
            dockerfile_path = os.path.join(build_dir, "Dockerfile")
            with open(dockerfile_path, "r", encoding="utf-8") as f:
                dockerfile_contents = f.read()

            lines = dockerfile_contents.splitlines()
            try:
                cmd_index = max(
                    i for i, ln in enumerate(lines) if ln.strip().startswith("CMD ")
                )
            except ValueError:
                cmd_index = len(lines)

            # Build RUN command
            run_lines: list[str] = []
            run_lines.append("RUN --mount=type=cache,target=/root/.cache/uv \\\n")
            run_lines.append('    echo "Installing nodetool packages..." \\\n')

            for idx, pkg in enumerate(nodetool_packages):
                install_url = str(pkg.get("install_url") or "").strip()
                install_arg = install_url
                # Add continuation except for the last line
                is_last = idx == len(nodetool_packages) - 1
                suffix = "" if is_last else " \\\n"
                run_lines.append(f"    && uv pip install {install_arg}{suffix}")

            # Insert before CMD
            new_lines = lines[:cmd_index] + run_lines + lines[cmd_index:]
            with open(dockerfile_path, "w", encoding="utf-8") as f:
                f.write(
                    "\n".join(new_lines)
                    + ("\n" if not new_lines[-1].endswith("\n") else "")
                )

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


def run_docker_image(
    image_name: str,
    tag: str,
    host_port: int,
    container_port: int = 80,
    *,
    container_name: str | None = None,
    env: dict[str, str] | None = None,
    volumes: list[tuple[str, str]] | None = None,
    detach: bool = True,
    gpus: str | bool | None = None,
    remove: bool = True,
    extra_args: list[str] | None = None,
) -> None:
    """
    Run a Docker image and map it to a given host port.

    Args:
        image_name (str): The Docker image repository/name (e.g., "myuser/myapp").
        tag (str): The Docker image tag (e.g., "latest").
        host_port (int): The host port to bind.
        container_port (int): The container port to expose (default: 80).
        container_name (str | None): Optional container name.
        env (dict[str, str] | None): Environment variables to pass to the container.
        volumes (list[tuple[str, str]] | None): Host to container volume mounts [(host, container)].
        detach (bool): Run container in detached mode (default: True).
        gpus (str | bool | None): GPU option (e.g., "all" or count). True implies "all".
        remove (bool): Automatically remove the container when it exits (default: True).
        extra_args (list[str] | None): Additional raw args to append to `docker run`.

    Raises:
        SystemExit: If `docker run` fails.
    """
    args: list[str] = [
        "docker",
        "run",
    ]

    if remove:
        args.append("--rm")
    if detach:
        args.append("-d")
    if container_name:
        args.extend(["--name", container_name])

    # Port mapping
    args.extend(["-p", f"{host_port}:{container_port}"])

    # Environment variables
    if env:
        for key, value in env.items():
            # Use shlex.quote to avoid shell injection issues in values
            quoted_value = shlex.quote(str(value))
            args.extend(["-e", f"{key}={quoted_value}"])

    # Volumes
    if volumes:
        for host_path, container_path in volumes:
            args.extend(["-v", f"{host_path}:{container_path}"])

    # GPUs
    if gpus:
        gpu_arg = "all" if gpus is True else str(gpus)
        args.extend(["--gpus", gpu_arg])

    if extra_args:
        args.extend(extra_args)

    # Image reference
    args.append(f"{image_name}:{tag}")

    # Build final command string
    # We join with spaces; individual values are already safe as we avoided shell interpolation for env values.
    command = " ".join(args)
    run_command(command)


if __name__ == "__main__":
    print(get_docker_username_from_config())
