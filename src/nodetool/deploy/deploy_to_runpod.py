#!/usr/bin/env python3
"""
RunPod Deployment Script for NodeTool Workflows

This script automates the deployment of NodeTool workflows to RunPod serverless infrastructure.
It performs the following operations:

1. Fetches a specific workflow from the NodeTool database
2. Embeds the complete workflow data into a Docker image
3. Builds a specialized Docker container for RunPod execution
4. Optionally creates RunPod templates and endpoints using the RunPod SDK

The resulting Docker image contains:
- Complete NodeTool runtime environment
- Embedded workflow JSON with all metadata
- Configured runpod_handler for serverless execution
- Environment variables for workflow identification

Usage:
    python deploy_to_runpod.py --workflow-id WORKFLOW_ID [--docker-username USERNAME]

Requirements:
    - Docker installed and running
    - Access to NodeTool database
    - RunPod API key (for deployment operations)
    - runpod Python SDK installed
    - Docker registry authentication (docker login)
    
Important Notes:
    - Images are built with --platform linux/amd64 for RunPod compatibility
    - Cross-platform builds may take longer on ARM-based systems (Apple Silicon)

Docker Username Resolution:
    The script automatically detects your Docker username from:
    1. --docker-username command line argument (highest priority)
    2. DOCKER_USERNAME environment variable
    3. Docker config file (~/.docker/config.json) - set by 'docker login'

Environment Variables:
    RUNPOD_API_KEY: Required for RunPod API operations
    DOCKER_USERNAME: Docker Hub username (optional if docker login was used)
    DOCKER_REGISTRY: Docker registry URL (defaults to Docker Hub)

Examples:
    # Basic usage (auto-detects username from docker login)
    python deploy_to_runpod.py --workflow-id abc123
    
    # Check your Docker configuration
    python deploy_to_runpod.py --check-docker-config
    
    # Specify username explicitly
    python deploy_to_runpod.py --workflow-id abc123 --docker-username myusername
    
    # Deploy with specific GPU types (use exact GPU IDs)
    python deploy_to_runpod.py --workflow-id abc123 --gpu-types AMPERE_24 ADA_48_PRO
    
    # Use specific tag instead of auto-generated
    python deploy_to_runpod.py --workflow-id abc123 --tag v1.0.0
    
    # Use different platform (advanced users only) 
    python deploy_to_runpod.py --workflow-id abc123 --platform linux/arm64
"""
import os
import subprocess
import sys
import argparse
import hashlib
import time
import json
import traceback
import runpod
from runpod.api.graphql import run_graphql_query
from enum import Enum
from typing import List, Optional

import dotenv
dotenv.load_dotenv()

# Configuration
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

assert RUNPOD_API_KEY, "RUNPOD_API_KEY environment variable is not set"


# RunPod API Enums and Constants
class ComputeType(Enum):
    """Compute types for RunPod endpoints."""
    CPU = "CPU"
    GPU = "GPU"

    def __str__(self):
        return self.value

    @classmethod
    def list_values(cls):
        return [item.value for item in cls]


class ScalerType(Enum):
    """Scaler types for RunPod endpoints."""
    QUEUE_DELAY = "QUEUE_DELAY"
    REQUEST_COUNT = "REQUEST_COUNT"
    
    def __str__(self):
        return self.value

    @classmethod
    def list_values(cls):
        return [item.value for item in cls]


class GPUType(Enum):
    """GPU types available in RunPod using their official GPU ID codes."""
    # Ada Lovelace Architecture
    ADA_24 = "ADA_24"  # L4, RTX 4000 series consumer cards
    ADA_32_PRO = "ADA_32_PRO"  # Professional Ada cards with 32GB
    ADA_48_PRO = "ADA_48_PRO"  # L40, L40S, RTX 6000 Ada
    ADA_80_PRO = "ADA_80_PRO"  # High-end Ada professional cards
    
    # Ampere Architecture
    AMPERE_16 = "AMPERE_16"  # RTX 3060, A2000, A4000
    AMPERE_24 = "AMPERE_24"  # RTX 3070/3080/3090, A4500, A5000
    AMPERE_48 = "AMPERE_48"  # A40, RTX A6000
    AMPERE_80 = "AMPERE_80"  # A100 80GB
    
    # Hopper Architecture
    HOPPER_141 = "HOPPER_141"  # H200 with 141GB memory

    def __str__(self):
        return self.value

    @classmethod
    def list_values(cls):
        return [item.value for item in cls]


class CPUFlavor(Enum):
    """CPU flavor IDs for RunPod endpoints."""
    CPU_3C = "cpu3c"
    CPU_3G = "cpu3g"
    CPU_5C = "cpu5c"
    CPU_5G = "cpu5g"

    def __str__(self):
        return self.value

    @classmethod
    def list_values(cls):
        return [item.value for item in cls]


class DataCenter(Enum):
    """Data center locations for RunPod endpoints."""
    # United States
    US_CALIFORNIA_2 = "US-CA-2"
    US_DELAWARE_1 = "US-DE-1"
    US_GEORGIA_1 = "US-GA-1"
    US_GEORGIA_2 = "US-GA-2"
    US_ILLINOIS_1 = "US-IL-1"
    US_KANSAS_2 = "US-KS-2"
    US_KANSAS_3 = "US-KS-3"
    US_NORTH_CAROLINA_1 = "US-NC-1"
    US_TEXAS_1 = "US-TX-1"
    US_TEXAS_3 = "US-TX-3"
    US_TEXAS_4 = "US-TX-4"
    US_WASHINGTON_1 = "US-WA-1"

    # Canada
    CA_MONTREAL_1 = "CA-MTL-1"
    CA_MONTREAL_2 = "CA-MTL-2"
    CA_MONTREAL_3 = "CA-MTL-3"

    # Europe
    EU_CZECH_REPUBLIC_1 = "EU-CZ-1"
    EU_FRANCE_1 = "EU-FR-1"
    EU_NETHERLANDS_1 = "EU-NL-1"
    EU_ROMANIA_1 = "EU-RO-1"
    EU_SWEDEN_1 = "EU-SE-1"

    # Europe Extended (Nordic/Iceland/Norway)
    EUR_ICELAND_1 = "EUR-IS-1"
    EUR_ICELAND_2 = "EUR-IS-2"
    EUR_ICELAND_3 = "EUR-IS-3"
    EUR_NORWAY_1 = "EUR-NO-1"

    # Asia-Pacific
    AP_JAPAN_1 = "AP-JP-1"

    # Oceania
    OC_AUSTRALIA_1 = "OC-AU-1"

    def __str__(self):
        return self.value

    @classmethod
    def list_values(cls):
        return [item.value for item in cls]


class CUDAVersion(Enum):
    """CUDA versions available in RunPod."""
    CUDA_11_8 = "11.8"
    CUDA_12_0 = "12.0"
    CUDA_12_1 = "12.1"
    CUDA_12_2 = "12.2"
    CUDA_12_3 = "12.3"
    CUDA_12_4 = "12.4"
    CUDA_12_5 = "12.5"
    CUDA_12_6 = "12.6"
    CUDA_12_7 = "12.7"
    CUDA_12_8 = "12.8"

    def __str__(self):
        return self.value

    @classmethod
    def list_values(cls):
        return [item.value for item in cls]


def print_enum_options(enum_class, title: str):
    """Print available options for an enum class."""
    print(f"\n{title}:")
    for item in enum_class:
        print(f"  {item.value}")

def run_command(command: str, capture_output: bool = False) -> str:
    """Run a shell command with streaming output and return output if requested."""
    print(f"Running: {command}")
    try:
        if capture_output:
            # For commands that need to return output, capture but still stream
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
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
                universal_newlines=True
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
            ["docker", "system", "info", "--format", "{{.RegistryConfig.IndexConfigs}}"],
            capture_output=True,
            text=True,
            check=False
        )
        
        # If we can't determine auth status, try a simple auth check
        if result.returncode != 0:
            # Try to authenticate by making a simple request
            auth_check = subprocess.run(
                ["docker", "login", "--get-login"],
                capture_output=True,
                text=True,
                check=False
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
        
        response = input(f"Do you want to login to {registry} now? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
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

def format_image_name(base_name: str, docker_username: str, registry: str = "docker.io") -> str:
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

def sanitize_name(name: str) -> str:
    """
    Sanitize a name for use as Docker image name or RunPod template name.
    
    Converts to lowercase, replaces spaces and special characters with hyphens,
    removes consecutive hyphens, and removes leading/trailing hyphens.
    
    Args:
        name (str): The name to sanitize
        
    Returns:
        str: The sanitized name
    """
    import re
    
    # Convert to lowercase and replace spaces/special chars with hyphens
    sanitized = re.sub(r'[^a-zA-Z0-9\-]', '-', name.lower())
    
    # Remove consecutive hyphens
    sanitized = re.sub(r'-+', '-', sanitized)
    
    # Remove leading/trailing hyphens
    sanitized = sanitized.strip('-')
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = "workflow"
    
    return sanitized

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


def create_ollama_pull_script(models: list[dict], build_dir: str) -> None:
    """
    Create a script to pull Ollama models during Docker build.
    
    Args:
        models (list[dict]): List of model dictionaries
        build_dir (str): Path to the Docker build directory
    """
    from pathlib import Path
    
    ollama_models = [m for m in models if m.get("type") == "language_model" and m.get("provider") == "ollama"]
    
    if not ollama_models:
        print("No Ollama models found to pull")
        # Create empty script so Docker build doesn't fail
        script_path = Path(build_dir) / "pull_ollama_models.sh"
        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\necho 'No Ollama models to pull'\n")
        script_path.chmod(0o755)
        return
    
    print(f"Creating script to pull {len(ollama_models)} Ollama models")
    
    script_content = ["#!/bin/bash", "set -e", ""]
    script_content.append("echo 'Starting Ollama service...'")
    script_content.append("ollama serve &")
    script_content.append("OLLAMA_PID=$!")
    script_content.append("")
    script_content.append("echo 'Waiting for Ollama to start...'")
    script_content.append("sleep 5")
    script_content.append("")
    
    for model in ollama_models:
        model_id = model.get("id")
        if model_id:
            script_content.append(f"echo 'Pulling Ollama model: {model_id}'")
            script_content.append(f"ollama pull {model_id}")
            script_content.append("")
    
    script_content.append("echo 'Stopping Ollama service...'")
    script_content.append("kill $OLLAMA_PID")
    script_content.append("wait $OLLAMA_PID 2>/dev/null || true")
    script_content.append("")
    script_content.append("echo 'All Ollama models pulled successfully'")
    
    script_path = Path(build_dir) / "pull_ollama_models.sh"
    with open(script_path, 'w') as f:
        f.write('\n'.join(script_content))
    
    # Make script executable
    script_path.chmod(0o755)
    
    print(f"Created Ollama pull script: {script_path}")
    for model in ollama_models:
        model_id = model.get("id")
        if model_id:
            print(f"  - {model_id}")

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
            if "type" in model and model.get("type", "").startswith("hf.") and "repo_id" in model and model["repo_id"]:
                # Create a unique key for this model
                model_key = (
                    "hf",
                    model.get("type"),
                    model.get("repo_id"),
                    model.get("path"),
                    model.get("variant")
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
                        "ignore_patterns": model.get("ignore_patterns")
                    }
                    models.append(hf_model)
            
            # Ollama language models
            elif model.get("type") == "language_model" and model.get("provider") == "ollama" and model.get("id"):
                model_key = ("ollama", model["id"])
                
                if model_key not in seen_models:
                    seen_models.add(model_key)
                    ollama_model = {
                        "type": "language_model",
                        "provider": "ollama",
                        "id": model["id"]
                    }
                    models.append(ollama_model)
        
        # Check for language models at the root level (some nodes might have them directly)
        if node_data.get("type") == "language_model" and node_data.get("provider") == "ollama" and node_data.get("id"):
            model_key = ("ollama", node_data["id"])
            
            if model_key not in seen_models:
                seen_models.add(model_key)
                ollama_model = {
                    "type": "language_model",
                    "provider": "ollama",
                    "id": node_data["id"]
                }
                models.append(ollama_model)
        
        # Check for nested model references (e.g., in arrays like loras)
        for key, value in node_data.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        # HuggingFace models in arrays
                        if "type" in item and item.get("type", "").startswith("hf.") and "repo_id" in item and item["repo_id"]:
                            model_key = (
                                "hf",
                                item.get("type"),
                                item.get("repo_id"),
                                item.get("path"),
                                item.get("variant")
                            )
                            
                            if model_key not in seen_models:
                                seen_models.add(model_key)
                                hf_model = {
                                    "type": item.get("type", "hf.model"),
                                    "repo_id": item["repo_id"],
                                    "path": item.get("path"),
                                    "variant": item.get("variant"),
                                    "allow_patterns": item.get("allow_patterns"),
                                    "ignore_patterns": item.get("ignore_patterns")
                                }
                                models.append(hf_model)
    
    return models

def fetch_workflow_from_db(workflow_id: str):
    """
    Fetch a workflow from the NodeTool database and save to a temporary file.
    
    This function connects to the NodeTool database, retrieves the specified workflow
    (respecting user permissions), and saves all workflow data to a temporary JSON file
    that will be embedded in the Docker image.
    
    Args:
        workflow_id (str): The unique identifier of the workflow to fetch
    
    Returns:
        tuple: (workflow_path, workflow_name) - Path to the temporary file and workflow name
        
    Raises:
        SystemExit: If workflow is not found, not accessible, or database connection fails
        
    Note:
        The returned file path should be cleaned up after use.
        All workflow fields from the database model are included in the JSON.
    """
    import tempfile
    from nodetool.models.workflow import Workflow
    
    # Fetch workflow
    workflow = Workflow.get(workflow_id)
    if not workflow:
        print(f"Error: Workflow {workflow_id} not found or not accessible")
        sys.exit(1)
    
    # Create temporary workflow file
    workflow_fd, workflow_path = tempfile.mkstemp(suffix='.json', prefix='workflow_')
    with os.fdopen(workflow_fd, 'w') as f:
        f.write(workflow.model_dump_json())
    
    print(f"Workflow '{workflow.name}' saved to {workflow_path}")
    return workflow_path, workflow.name

def build_docker_image(workflow_path: str, image_name: str, tag: str, platform: str = "linux/amd64", embed_models: bool = True):
    """
    Build a Docker image for RunPod deployment with an embedded workflow.
    
    This function creates a specialized Docker image by:
    1. Using the RunPod-specific Dockerfile from src/nodetool/deploy/
    2. Extracting Hugging Face models from the workflow
    3. Adding model copy instructions to pre-cache models from local HF cache
    4. Creating a temporary build directory with all necessary files
    5. Building the final image using Docker from the build directory
    
    The resulting image is self-contained and includes:
    - All NodeTool dependencies and runtime
    - Pre-downloaded Hugging Face models in /huggingface/hub cache
    - The specific workflow JSON embedded at /app/workflow.json
    - The runpod_handler.py configured as the entry point
    - start.sh script for proper RunPod initialization
    
    The image is built with --platform linux/amd64 to ensure compatibility with
    RunPod's Linux servers, regardless of the build machine's architecture.
    
    Args:
        workflow_path (str): Path to the temporary workflow JSON file
        image_name (str): Name of the Docker image (including registry/username)
        tag (str): Tag of the Docker image
        platform (str): Docker build platform (default: linux/amd64)
        
    Raises:
        SystemExit: If Docker build fails or required files are missing
        
    Note:
        Creates and cleans up temporary build directory during the process.
    """
    import tempfile
    import shutil
    
    print(f"Building Docker image with embedded workflow from {workflow_path}")
    print(f"Platform: {platform}")
    
    # Load workflow data to extract all models
    with open(workflow_path, 'r') as f:
        workflow_data = json.load(f)
    
    models = extract_models(workflow_data)
    
    if models:
        print(f"Found {len(models)} models to download during Docker build:")
        for model in models:
            if model.get("type", "").startswith("hf."):
                print(f"  - HuggingFace {model['type']}: {model['repo_id']}")
                if model.get('path'):
                    print(f"    Path: {model['path']}")
                if model.get('variant'):
                    print(f"    Variant: {model['variant']}")
            elif model.get("type") == "language_model" and model.get("provider") == "ollama":
                print(f"  - Ollama: {model['id']}")
    
    # Get the deploy directory where Dockerfile, runpod_handler.py, download_models.py are located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    deploy_dockerfile_path = os.path.join(script_dir, "Dockerfile")
    runpod_handler_path = os.path.join(script_dir, "runpod_handler.py")
    download_models_path = os.path.join(script_dir, "download_models.py")
    
    # Create a temporary build directory
    build_dir = tempfile.mkdtemp(prefix='nodetool_build_')
    print(f"Using build directory: {build_dir}")
    
    try:
        # Copy all necessary files to the build directory
        shutil.copy(workflow_path, os.path.join(build_dir, "workflow.json"))
        shutil.copy(runpod_handler_path, os.path.join(build_dir, "runpod_handler.py"))
        shutil.copy(deploy_dockerfile_path, os.path.join(build_dir, "Dockerfile"))
        shutil.copy(download_models_path, os.path.join(build_dir, "download_models.py"))
        
        # Create models.json file with list of all models
        models_file_path = os.path.join(build_dir, "models.json")
        with open(models_file_path, 'w') as f:
            json.dump(models, f, indent=2)
        print(f"Created models.json with {len(models)} models")
        
        
        # Create script to pull Ollama models during Docker build
        create_ollama_pull_script(models, build_dir)
        
        # Build with the Dockerfile from the build directory
        original_dir = os.getcwd()
        os.chdir(build_dir)
        run_command(f"docker build --platform {platform} -t {image_name}:{tag} .")
        os.chdir(original_dir)
        
        print("Docker image built successfully")
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

def delete_runpod_template_by_name(template_name: str) -> bool:
    """
    Delete a RunPod template by name using GraphQL API.
    
    According to RunPod documentation, templates must have unique names,
    and deletion is done by template name via GraphQL mutation.
    
    Args:
        template_name (str): The template name to delete
        
    Returns:
        bool: True if deletion was successful or template doesn't exist
    """
    if not RUNPOD_API_KEY:
        print("Error: RUNPOD_API_KEY environment variable not set")
        sys.exit(1)
    
    # GraphQL mutation to delete template by name
    query = f"""
    mutation {{
        deleteTemplate(templateName: "{template_name}")
    }}
    """
    
    try:
        result = run_graphql_query(query)
        
        if 'errors' in result:
            # Template might not exist, which is fine for our use case
            error_msg = result['errors'][0].get('message', 'Unknown error')
            print(f"Template deletion result: {error_msg}")
            return True  # Treat as success if template doesn't exist
        print(f"Template '{template_name}' deleted successfully")
        return True
            
    except Exception as e:
        print(f"Error deleting template '{template_name}': {e}")
        return False

def delete_runpod_endpoint_by_name(endpoint_name: str) -> bool:
    """
    Delete a RunPod endpoint by name using GraphQL API.
    
    Endpoints must have unique names, so we need to delete existing ones
    before creating new ones with the same name.
    
    Args:
        endpoint_name (str): The endpoint name to delete
        
    Returns:
        bool: True if deletion was successful or endpoint doesn't exist
    """
    if not RUNPOD_API_KEY:
        print("Error: RUNPOD_API_KEY environment variable not set")
        sys.exit(1)
    
    try:
        # First, get all endpoints to find the one with matching name
        query = """
        query getEndpoints {
            myself {
                serverlessDiscount {
                    discountFactor
                    type
                }
                endpoints {
                    id
                    name
                }
            }
        }
        """
        
        result = run_graphql_query(query)
        
        if 'errors' in result:
            print(f"Error fetching endpoints: {result['errors']}")
            return False
            
        endpoints = result.get('data', {}).get('myself', {}).get('endpoints', [])
        endpoint_id = None
        
        # Find endpoint with matching name
        for endpoint in endpoints:
            if endpoint.get('name') == endpoint_name:
                endpoint_id = endpoint.get('id')
                break
        
        if not endpoint_id:
            print(f"Endpoint '{endpoint_name}' not found (may not exist)")
            return True  # Treat as success if endpoint doesn't exist
        
        # Delete the endpoint
        delete_query = f"""
        mutation {{
            deleteEndpoint(endpointId: "{endpoint_id}")
        }}
        """
        
        delete_result = run_graphql_query(delete_query)
        
        if 'errors' in delete_result:
            error_msg = delete_result['errors'][0].get('message', 'Unknown error')
            print(f"Error deleting endpoint: {error_msg}")
            return False
            
        print(f"Endpoint '{endpoint_name}' (ID: {endpoint_id}) deleted successfully")
        return True
        
    except Exception as e:
        print(f"Error deleting endpoint '{endpoint_name}': {e}")
        return False

def get_runpod_template_by_name(template_name: str) -> dict | None:
    """
    Get a RunPod template by name using GraphQL API.
    
    Args:
        template_name (str): The template name to find
        
    Returns:
        dict | None: Template data if found, None otherwise
    """
    if not RUNPOD_API_KEY:
        print("Error: RUNPOD_API_KEY environment variable not set")
        return None
    
    try:
        # Query all templates to find the one with matching name
        query = """
        query getTemplates {
            myself {
                podTemplates {
                    id
                    name
                    imageName
                    dockerArgs
                    containerDiskInGb
                    volumeInGb
                    volumeMountPath
                    ports
                    env {
                        key
                        value
                    }
                    isServerless
                }
            }
        }
        """
        
        result = run_graphql_query(query)
        
        if 'errors' in result:
            print(f"Error fetching templates: {result['errors']}")
            return None
            
        templates = result.get('data', {}).get('myself', {}).get('podTemplates', [])
        
        # Find template with matching name
        for template in templates:
            if template.get('name') == template_name:
                return template
        
        return None
        
    except Exception as e:
        print(f"Error fetching template '{template_name}': {e}")
        return None

def update_runpod_template(template_data: dict, image_name: str, tag: str) -> bool:
    """
    Update an existing RunPod template with a new Docker image.
    
    Args:
        template_data (dict): The existing template data with all current settings
        image_name (str): Name of the Docker image
        tag (str): Tag of the Docker image
        
    Returns:
        bool: True if update was successful
    """
    if not RUNPOD_API_KEY:
        print("Error: RUNPOD_API_KEY environment variable not set")
        return False
    
    try:
        template_id = template_data['id']
        
        # Prepare environment variables for GraphQL (always required, even if empty)
        env_parts = []
        if template_data.get('env'):
            for env_item in template_data['env']:
                key = env_item.get('key', '')
                value = env_item.get('value', '')
                env_parts.append(f'{{"key": "{key}", "value": "{value}"}}')
        
        env_vars_str = f'[{", ".join(env_parts)}]'
        
        # GraphQL mutation to update template with all required fields
        query = f"""
        mutation {{
            saveTemplate(input: {{
                id: "{template_id}"
                name: "{template_data.get('name', '')}"
                imageName: "{image_name}:{tag}"
                dockerArgs: "{template_data.get('dockerArgs', '')}"
                containerDiskInGb: {template_data.get('containerDiskInGb', 20)}
                volumeInGb: {template_data.get('volumeInGb', 0)}
                volumeMountPath: "{template_data.get('volumeMountPath', '/workspace')}"
                ports: "{template_data.get('ports', '8000/http')}"
                isServerless: {str(template_data.get('isServerless', True)).lower()}
                env: {env_vars_str}
            }}) {{
                id
                name
                imageName
            }}
        }}
        """
        
        result = run_graphql_query(query)
        
        if 'errors' in result:
            error_msg = result['errors'][0].get('message', 'Unknown error')
            print(f"Error updating template: {error_msg}")
            return False
            
        updated_template = result.get('data', {}).get('saveTemplate', {})
        print(f"Template '{updated_template.get('name')}' updated with image: {updated_template.get('imageName')}")
        return True
        
    except Exception as e:
        print(f"Error updating template: {e}")
        return False

def create_or_update_runpod_template(template_name: str, image_name: str, tag: str) -> str:
    """
    Create or update a RunPod template with the latest Docker image.
    
    This function first checks if a template with the given name exists:
    - If it exists, updates it with the new Docker image
    - If it doesn't exist, creates a new template
    
    This approach is more efficient than deleting and recreating templates,
    as it preserves template settings while updating the image.
    
    Template Configuration:
    - Container disk: 20GB for dependencies and temporary files
    - Environment variables: PYTHONPATH set to /app
    - Ports: 8000/http for potential web interfaces
    - No Jupyter or SSH access (serverless execution only)
    
    Args:
        template_name (str): Name of the template
        image_name (str): Name of the Docker image
        tag (str): Tag of the Docker image
    
    Returns:
        str: The template ID for use in endpoint creation
        
    Raises:
        SystemExit: If template creation/update fails or API key is invalid
        
    Note:
        GPU types must use RunPod's official GPU ID codes:
        - AMPERE_16, AMPERE_24, AMPERE_48, AMPERE_80
        - ADA_24, ADA_32_PRO, ADA_48_PRO, ADA_80_PRO
        - HOPPER_141
        
    Note:
        Requires RUNPOD_API_KEY environment variable to be set.
    """
    if not RUNPOD_API_KEY:
        print("Error: RUNPOD_API_KEY environment variable not set")
        sys.exit(1)
    
    runpod.api_key = RUNPOD_API_KEY
    
    # Check if template already exists
    print(f"Checking for existing template: {template_name}")
    existing_template = get_runpod_template_by_name(template_name)
    
    if existing_template:
        # Update existing template with new image
        template_id = existing_template['id']
        print(f"Found existing template (ID: {template_id})")
        print(f"Current image: {existing_template.get('imageName', 'unknown')}")
        print(f"Updating with new image: {image_name}:{tag}")
        
        if update_runpod_template(existing_template, image_name, tag):
            print(f"Template '{template_name}' updated successfully")
            return template_id
        else:
            print(f"Failed to update template '{template_name}'")
            sys.exit(1)
    else:
        # Create new template
        print(f"Template not found, creating new template: {template_name}")
        try:
            template = runpod.create_template(
                name=template_name,
                image_name=f"{image_name}:{tag}",
                container_disk_in_gb=20,
                volume_in_gb=0,
                volume_mount_path="/workspace",
                ports="8000/http",
                is_serverless=True,
            )
            
            template_id = template["id"]
            print(f"Template created successfully: {template_id}")
            return template_id
            
        except Exception as e:
            error_msg = str(e)
            if "already exists" in error_msg.lower() or "unique" in error_msg.lower():
                print(f"❌ Template '{template_name}' already exists!")
                print("This might be a race condition. Please try again.")
            else:
                traceback.print_exc()
                print(f"Failed to create template: {e}")
            sys.exit(1)

def create_runpod_endpoint(
    template_id: str,
    name: str,
    compute_type: str = ComputeType.GPU.value,
    gpu_type_ids: Optional[List[str]] = None,
    gpu_count: Optional[int] = None,
    cpu_flavor_ids: Optional[List[str]] = None,
    vcpu_count: Optional[int] = None,
    data_center_ids: Optional[List[str]] = None,
    workers_min: int = 0,
    workers_max: int = 3,
    idle_timeout: int = 5,
    scaler_type: str = ScalerType.QUEUE_DELAY.value,
    scaler_value: int = 4,
    execution_timeout_ms: Optional[int] = None,
    flashboot: bool = False,
    network_volume_id: Optional[str] = None,
    allowed_cuda_versions: Optional[List[str]] = None,
):
    """
    Create a RunPod serverless endpoint using raw GraphQL queries.
    
    Creates a serverless endpoint that can execute the NodeTool workflow
    with auto-scaling capabilities and optional GPU acceleration.
    
    If an endpoint with the same name already exists, it will be deleted first.
    
    Args:
        template_id (str): The RunPod template ID from create_or_update_runpod_template()
        name (str): Name of the endpoint
        compute_type (str): Type of compute (CPU or GPU)
        gpu_type_ids (List[str], optional): List of GPU type IDs (e.g. AMPERE_24, ADA_48_PRO)
        gpu_count (int, optional): Number of GPUs per worker
        cpu_flavor_ids (List[str], optional): List of CPU flavors for CPU compute
        vcpu_count (int, optional): Number of vCPUs for CPU compute
        data_center_ids (List[str], optional): Preferred data center locations
        workers_min (int): Minimum number of workers (0 for auto-scaling)
        workers_max (int): Maximum number of workers
        idle_timeout (int): Seconds before scaling down idle workers
        scaler_type (str): Type of auto-scaler (QUEUE_DELAY or REQUEST_COUNT)
        scaler_value (int): Threshold value for the scaler
        execution_timeout_ms (int, optional): Maximum execution time in milliseconds
        flashboot (bool): Enable flashboot for faster worker startup
        network_volume_id (str, optional): Network volume to attach
        allowed_cuda_versions (List[str], optional): Allowed CUDA versions
        
    Returns:
        str: The endpoint ID for workflow execution
        
    Raises:
        SystemExit: If endpoint creation fails or template is invalid
        
    Note:
        The endpoint will automatically scale workers based on incoming requests.
        Workers are terminated after idle_timeout seconds of inactivity to minimize costs.
    """
    if not RUNPOD_API_KEY:
        print("Error: RUNPOD_API_KEY environment variable not set")
        sys.exit(1)
    
    # Set sensible defaults
    if gpu_type_ids is None and compute_type == ComputeType.GPU.value:
        gpu_type_ids = [GPUType.AMPERE_24.value]
    
    if cpu_flavor_ids is None and compute_type == ComputeType.CPU.value:
        cpu_flavor_ids = [CPUFlavor.CPU_3C.value]
    
    if data_center_ids is None:
        # Leave empty for RunPod to choose optimal data centers automatically
        data_center_ids = []
    
    # Delete existing endpoint if it exists
    print(f"Checking for existing endpoint: {name}")
    if delete_runpod_endpoint_by_name(name):
        print(f"Existing endpoint '{name}' deleted successfully")
    else:
        print(f"Note: Endpoint '{name}' may not have existed")
    
    try:
        print(f"Creating new endpoint: {name}")
        print(f"  Compute type: {compute_type}")
        if compute_type == ComputeType.GPU.value:
            print(f"  GPU types: {gpu_type_ids}")
            if gpu_count:
                print(f"  GPU count: {gpu_count}")
        else:
            print(f"  CPU flavors: {cpu_flavor_ids}")
            if vcpu_count:
                print(f"  vCPU count: {vcpu_count}")
        print(f"  Data centers: {data_center_ids if data_center_ids else 'Auto-selected'}")
        print(f"  Workers: {workers_min}-{workers_max}")
        print(f"  Scaler: {scaler_type} (threshold: {scaler_value})")
        print(f"  Idle timeout: {idle_timeout}s")
        
        # Build the GraphQL mutation for saveEndpoint
        # GPU IDs should be passed as a comma-separated string
        if compute_type == ComputeType.GPU.value:
            # Simply join the GPU IDs with commas
            gpu_ids_field = ",".join(gpu_type_ids) if gpu_type_ids else "AMPERE_24"
        
        # Build the mutation
        mutation_parts = []
        mutation_parts.append(f'name: "{name}"')
        mutation_parts.append(f'templateId: "{template_id}"')
        mutation_parts.append('type: "QB"')  # Serverless type
        mutation_parts.append(f'workersMin: {workers_min}')
        mutation_parts.append(f'workersMax: {workers_max}')
        mutation_parts.append(f'idleTimeout: {idle_timeout}')
        mutation_parts.append(f'scalerType: "{scaler_type}"')
        mutation_parts.append(f'scalerValue: {scaler_value}')
        
        # Add compute-specific fields
        if compute_type == ComputeType.GPU.value:
            mutation_parts.append(f'gpuIds: "{gpu_ids_field}"')
            if gpu_count:
                mutation_parts.append(f'gpuCount: {gpu_count}')
        
        # Add optional fields
        if data_center_ids:
            # Format locations as JSON array string
            locations_str = json.dumps(data_center_ids)
            mutation_parts.append(f'locations: {locations_str}')
        else:
            mutation_parts.append('locations: null')
        
        if network_volume_id:
            mutation_parts.append(f'networkVolumeId: "{network_volume_id}"')
        else:
            mutation_parts.append('networkVolumeId: null')
        
        # Build the complete mutation
        mutation = f"""
        mutation {{
            saveEndpoint(input: {{
                {', '.join(mutation_parts)}
            }}) {{
                id
                name
                gpuIds
                idleTimeout
                locations
                type
                networkVolumeId
                scalerType
                scalerValue
                templateId
                userId
                workersMax
                workersMin
                gpuCount
                __typename
            }}
        }}
        """
        
        print(f"\nExecuting GraphQL mutation:")
        print(mutation)
        
        result = run_graphql_query(mutation)
        
        if 'errors' in result:
            error_msg = result['errors'][0].get('message', 'Unknown error')
            print(f"Error creating endpoint: {error_msg}")
            print(f"Full error response: {json.dumps(result, indent=2)}")
            sys.exit(1)
        
        endpoint_data = result.get('data', {}).get('saveEndpoint', {})
        endpoint_id = endpoint_data.get('id')
        
        if not endpoint_id:
            print(f"Error: No endpoint ID returned")
            print(f"Response data: {json.dumps(result, indent=2)}")
            sys.exit(1)
        
        print(f"\nEndpoint created successfully!")
        print(f"  ID: {endpoint_id}")
        print(f"  Name: {endpoint_data.get('name')}")
        print(f"  GPU Configuration: {endpoint_data.get('gpuIds')}")
        print(f"  Workers: {endpoint_data.get('workersMin')}-{endpoint_data.get('workersMax')}")
        print(f"  Scaler: {endpoint_data.get('scalerType')} (threshold: {endpoint_data.get('scalerValue')})")
        
        return endpoint_id
        
    except Exception as e:
        traceback.print_exc()
        print(f"Failed to create endpoint: {e}")
        print("\nTroubleshooting:")
        print("1. Check that your RunPod API key is valid")
        print("2. Verify the template ID exists and is accessible")
        print("3. Ensure the specified GPU types are available in your chosen regions")
        print("4. Check your RunPod account limits and quotas")
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
    from pathlib import Path
    
    try:
        # Docker config is typically stored in ~/.docker/config.json
        docker_config_path = Path.home() / ".docker" / "config.json"
        
        if not docker_config_path.exists():
            return None
            
        with open(docker_config_path, 'r') as f:
            config = json.load(f)
        
        # Check for authentication data
        auths = config.get('auths', {})
        
        # Docker Hub can be referenced as docker.io, index.docker.io, or https://index.docker.io/v1/
        possible_registry_keys = [
            registry,
            f"https://{registry}/v1/",
            "https://index.docker.io/v1/" if registry == "docker.io" else f"https://{registry}/v1/",
            "index.docker.io" if registry == "docker.io" else registry
        ]
        
        for reg_key in possible_registry_keys:
            if reg_key in auths:
                auth_data = auths[reg_key]
                
                # Check if there's a direct username field
                if 'username' in auth_data:
                    return auth_data['username']
                
                # Check if there's base64 encoded auth data
                if 'auth' in auth_data:
                    try:
                        # Decode base64 auth string (format: username:password)
                        auth_str = base64.b64decode(auth_data['auth']).decode('utf-8')
                        username, _ = auth_str.split(':', 1)
                        return username
                    except Exception:
                        continue
        
        # Check credential helpers
        cred_helpers = config.get('credHelpers', {})
        if registry in cred_helpers:
            # For credential helpers, we can't easily get the username
            # but we can try to run the helper (this is more complex)
            pass
            
        return None
        
    except Exception as e:
        print(f"Warning: Could not read Docker config: {e}")
        return None