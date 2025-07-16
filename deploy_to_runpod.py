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
    python deploy_to_runpod.py --workflow-id WORKFLOW_ID --user-id USER_ID

Requirements:
    - Docker installed and running
    - Access to NodeTool database
    - RunPod API key (for deployment operations)
    - runpod Python SDK installed

Environment Variables:
    RUNPOD_API_KEY: Required for RunPod API operations
"""
import os
import subprocess
import sys
import json
import argparse
import runpod

# Configuration
DOCKER_IMAGE_NAME = "nodetool-runpod"
DOCKER_TAG = "latest"
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

def run_command(command: str, capture_output: bool = False) -> str:
    """Run a shell command and return output if requested."""
    print(f"Running: {command}")
    try:
        if capture_output:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        else:
            subprocess.run(command, shell=True, check=True)
            return ""
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        sys.exit(1)

def fetch_workflow_from_db(workflow_id: str, user_id: str):
    """
    Fetch a workflow from the NodeTool database and save to a temporary file.
    
    This function connects to the NodeTool database, retrieves the specified workflow
    (respecting user permissions), and saves all workflow data to a temporary JSON file
    that will be embedded in the Docker image.
    
    Args:
        workflow_id (str): The unique identifier of the workflow to fetch
        user_id (str): The user ID for permission checking (must own workflow or workflow must be public)
    
    Returns:
        str: Path to the temporary file containing the workflow JSON data
        
    Raises:
        SystemExit: If workflow is not found, not accessible, or database connection fails
        
    Note:
        The returned file path should be cleaned up after use.
        All workflow fields from the database model are included in the JSON.
    """
    import tempfile
    from nodetool.models.workflow import Workflow
    from nodetool.common.environment import Environment
    
    # Initialize environment for database access
    Environment.setup()
    
    # Fetch workflow
    workflow = Workflow.find(user_id, workflow_id)
    if not workflow:
        print(f"Error: Workflow {workflow_id} not found or not accessible")
        sys.exit(1)
    
    # Save all workflow fields to JSON file
    workflow_data = {
        "id": workflow.id,
        "user_id": workflow.user_id,
        "access": workflow.access,
        "created_at": workflow.created_at,
        "updated_at": workflow.updated_at,
        "name": workflow.name,
        "tags": workflow.tags,
        "description": workflow.description,
        "package_name": workflow.package_name,
        "thumbnail": workflow.thumbnail,
        "thumbnail_url": workflow.thumbnail_url,
        "graph": workflow.graph,
        "settings": workflow.settings,
        "receive_clipboard": workflow.receive_clipboard,
        "run_mode": workflow.run_mode
    }
    
    # Create temporary workflow file
    workflow_fd, workflow_path = tempfile.mkstemp(suffix='.json', prefix='workflow_')
    with os.fdopen(workflow_fd, 'w') as f:
        json.dump(workflow_data, f, indent=2, default=str)
    
    print(f"Workflow '{workflow.name}' saved to {workflow_path}")
    return workflow_path

def build_docker_image(workflow_path: str):
    """
    Build a Docker image for RunPod deployment with an embedded workflow.
    
    This function creates a specialized Docker image by:
    1. Reading the base NodeTool Dockerfile
    2. Appending additional layers to embed the workflow and runpod handler
    3. Setting environment variables for workflow discovery
    4. Building the final image using Docker
    
    The resulting image is self-contained and includes:
    - All NodeTool dependencies and runtime
    - The specific workflow JSON embedded at /app/embedded_workflow.json
    - The runpod_handler.py configured as the entry point
    - EMBEDDED_WORKFLOW_PATH environment variable set
    
    Args:
        workflow_path (str): Path to the temporary workflow JSON file
        
    Raises:
        SystemExit: If Docker build fails or required files are missing
        
    Note:
        Creates and cleans up temporary files during the build process.
        The workflow file is copied to the build context temporarily.
    """
    import tempfile
    
    print(f"Building Docker image with embedded workflow from {workflow_path}")
    
    # Create temporary Dockerfile by appending to original
    dockerfile_fd, dockerfile_path = tempfile.mkstemp(suffix='.dockerfile', prefix='Dockerfile_')
    
    # Read original Dockerfile and append workflow commands
    with open("Dockerfile", "r") as f:
        original_dockerfile = f.read()
    
    workflow_filename = os.path.basename(workflow_path)
    
    additional_lines = f"""
# Copy embedded workflow
COPY {workflow_filename} /app/embedded_workflow.json
ENV EMBEDDED_WORKFLOW_PATH=/app/embedded_workflow.json

# Copy runpod handler
COPY src/nodetool/api/runpod_handler.py /app/runpod_handler.py

# Set the entrypoint
CMD ["python", "/app/runpod_handler.py"]
"""
    
    # Write combined Dockerfile
    with os.fdopen(dockerfile_fd, 'w') as f:
        f.write(original_dockerfile)
        f.write(additional_lines)
    
    # Copy workflow file to build context
    workflow_build_path = os.path.basename(workflow_path)
    run_command(f"cp {workflow_path} {workflow_build_path}")
    
    try:
        # Build with the modified Dockerfile
        run_command(f"docker build -f {dockerfile_path} -t {DOCKER_IMAGE_NAME}:{DOCKER_TAG} .")
        print("Docker image built successfully")
    finally:
        # Clean up temporary files
        os.unlink(dockerfile_path)
        os.unlink(workflow_build_path)

def push_to_registry():
    """Push Docker image to a registry."""
    print("Pushing Docker image to registry...")
    run_command(f"docker push {DOCKER_IMAGE_NAME}:{DOCKER_TAG}")
    print("Docker image pushed successfully")

def create_runpod_template():
    """
    Create a RunPod template using the RunPod SDK.
    
    Creates a template configuration that defines the Docker image and runtime
    settings for the NodeTool workflow execution environment on RunPod.
    
    Template Configuration:
    - Container disk: 20GB for dependencies and temporary files
    - Environment variables: PYTHONPATH set to /app
    - Ports: 8000/http for potential web interfaces
    - No Jupyter or SSH access (serverless execution only)
    
    Returns:
        str: The template ID for use in endpoint creation
        
    Raises:
        SystemExit: If template creation fails or API key is invalid
        
    Note:
        Requires RUNPOD_API_KEY environment variable to be set.
        The template references the Docker image built by build_docker_image().
    """
    if not RUNPOD_API_KEY:
        print("Error: RUNPOD_API_KEY environment variable not set")
        sys.exit(1)
    
    runpod.api_key = RUNPOD_API_KEY
    
    try:
        template = runpod.create_template(
            name="nodetool-workflow",
            image_name=f"{DOCKER_IMAGE_NAME}:{DOCKER_TAG}",
            docker_args="",
            container_disk_in_gb=20,
            volume_in_gb=0,
            volume_mount_path="/workspace",
            env_vars={
                "PYTHONPATH": "/app"
            },
            ports="8000/http",
            start_jupyter=False,
            start_ssh=False,
            readme="NodeTool workflow runner for RunPod serverless"
        )
        
        template_id = template["id"]
        print(f"Template created successfully: {template_id}")
        return template_id
        
    except Exception as e:
        print(f"Failed to create template: {e}")
        sys.exit(1)

def create_runpod_endpoint(template_id: str):
    """
    Create a RunPod serverless endpoint using the RunPod SDK.
    
    Creates a serverless endpoint that can execute the NodeTool workflow
    with auto-scaling capabilities and GPU acceleration.
    
    Endpoint Configuration:
    - GPU: AMPERE_16 (RTX A4000/A5000 class)
    - Auto-scaling: 0-3 workers based on queue delay
    - Idle timeout: 5 seconds for cost efficiency
    - Locations: US regions for optimal performance
    - Scaler: QUEUE_DELAY with 4-second threshold
    
    Args:
        template_id (str): The RunPod template ID from create_runpod_template()
        
    Returns:
        str: The endpoint ID for workflow execution
        
    Raises:
        SystemExit: If endpoint creation fails or template is invalid
        
    Note:
        The endpoint will automatically scale workers based on incoming requests.
        Workers are terminated after 5 seconds of inactivity to minimize costs.
    """
    try:
        endpoint = runpod.create_endpoint(
            name="nodetool-workflow-endpoint",
            template_id=template_id,
            gpu_ids="AMPERE_16",
            workers_min=0,
            workers_max=3,
            idle_timeout=5,
            scaler_type="QUEUE_DELAY",
            scaler_value=4,
            locations="US"
        )
        
        endpoint_id = endpoint["id"]
        print(f"Endpoint created successfully: {endpoint_id}")
        return endpoint_id
        
    except Exception as e:
        print(f"Failed to create endpoint: {e}")
        sys.exit(1)

def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy workflow to RunPod")
    parser.add_argument("--workflow-id", required=True, help="Workflow ID to deploy")
    parser.add_argument("--user-id", required=True, help="User ID (for workflow access)")
    parser.add_argument("--skip-build", action="store_true", help="Skip Docker build")
    parser.add_argument("--skip-deploy", action="store_true", help="Skip RunPod deployment")
    
    args = parser.parse_args()
    
    print("Starting RunPod deployment...")
    
    # Check if Docker is running
    if not args.skip_build:
        try:
            run_command("docker --version", capture_output=True)
        except:
            print("Error: Docker is not installed or not running")
            sys.exit(1)
    
    # Fetch workflow from database
    workflow_path = fetch_workflow_from_db(args.workflow_id, args.user_id)
    
    try:
        # Build Docker image with embedded workflow
        if not args.skip_build:
            build_docker_image(workflow_path)
        
        # Deploy to RunPod
        if not args.skip_deploy:
            print("\nNote: You need to push the Docker image to a public registry first")
            print("Example: docker tag nodetool-runpod:latest yourusername/nodetool-runpod:latest")
            print("         docker push yourusername/nodetool-runpod:latest")
            print("\nThen uncomment the deployment code below:")
            # push_to_registry()
            # template_id = create_runpod_template()
            # endpoint_id = create_runpod_endpoint(template_id)
        
        print(f"\nDeployment script completed for workflow {args.workflow_id}")
        
    finally:
        # Clean up workflow file
        os.unlink(workflow_path)

if __name__ == "__main__":
    main()