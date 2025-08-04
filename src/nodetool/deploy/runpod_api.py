#!/usr/bin/env python3
"""
RunPod API Module

This module provides a clean interface to RunPod's REST API for managing
templates and endpoints. It handles authentication, request/response processing,
and error handling for all RunPod operations.

Key Features:
- Template management (create, update, delete, get)
- Endpoint management (create, delete)
- Proper error handling and logging
- Type safety with enums for RunPod constants

Usage:
    from nodetool.deploy.runpod_api import (
        make_runpod_api_call,
        create_or_update_runpod_template,
        create_runpod_endpoint
    )
"""
import os
import sys
import json
import traceback
import requests
from enum import Enum
from typing import List, Optional

# Configuration
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_REST_BASE_URL = "https://rest.runpod.io/v1"

assert RUNPOD_API_KEY, "RUNPOD_API_KEY environment variable is not set"


def make_runpod_api_call(
    endpoint: str, method: str = "GET", data: dict | None = None
) -> dict:
    """
    Make a REST API call to RunPod.

    Args:
        endpoint (str): The API endpoint (e.g., "endpoints", "templates")
        method (str): HTTP method (GET, POST, PUT, PATCH, DELETE)
        data (dict): Request data for POST/PUT/PATCH requests

    Returns:
        dict: API response data, or empty dict for DELETE requests

    Raises:
        SystemExit: If API call fails
    """
    url = f"{RUNPOD_REST_BASE_URL}/{endpoint}"
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        if method == "POST":
            response = requests.post(url, headers=headers, json=data)
        elif method == "PUT":
            response = requests.put(url, headers=headers, json=data)
        elif method == "PATCH":
            response = requests.patch(url, headers=headers, json=data)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        else:
            response = requests.get(url, headers=headers)

        response.raise_for_status()

        # DELETE requests might not return JSON content
        if method == "DELETE" and response.status_code == 204:
            return {}

        # Try to parse JSON, but handle empty responses
        try:
            return response.json()
        except ValueError:
            return {}

    except requests.exceptions.RequestException as e:
        print(f"RunPod API call failed: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        sys.exit(1)


# Network Volume Management
def create_network_volume(name: str, size: int, data_center_id: str) -> dict:
    """
    Create a new network volume.
    
    Args:
        name (str): Name of the network volume
        size (int): Size in GB
        data_center_id (str): Data center ID where volume should be created
    
    Returns:
        dict: Created network volume data
    """
    data = {
        "dataCenterId": data_center_id,
        "name": name,
        "size": size
    }
    
    return make_runpod_api_call("networkvolumes", "POST", data)


def list_network_volumes() -> dict:
    """
    List all network volumes.
    
    Returns:
        dict: List of network volumes
    """
    return make_runpod_api_call("networkvolumes", "GET")


def get_network_volume(volume_id: str) -> dict:
    """
    Get details of a specific network volume.
    
    Args:
        volume_id (str): ID of the network volume
    
    Returns:
        dict: Network volume details
    """
    return make_runpod_api_call(f"networkvolumes/{volume_id}", "GET")


def update_network_volume(volume_id: str, name: str | None = None, size: int | None = None) -> dict:
    """
    Update a network volume.
    
    Args:
        volume_id (str): ID of the network volume
        name (str, optional): New name for the volume
        size (int, optional): New size in GB
    
    Returns:
        dict: Updated network volume data
    """
    data = {}
    if name is not None:
        data["name"] = name
    if size is not None:
        data["size"] = size
    
    return make_runpod_api_call(f"networkvolumes/{volume_id}", "PATCH", data)


# RunPod API Enums and Constants
class ComputeType(str, Enum):
    """Compute types for RunPod endpoints."""

    CPU = "CPU"
    GPU = "GPU"

    def __str__(self):
        return self.value

    @classmethod
    def list_values(cls):
        return [item.value for item in cls]


class ScalerType(str, Enum):
    """Scaler types for RunPod endpoints."""

    QUEUE_DELAY = "QUEUE_DELAY"
    REQUEST_COUNT = "REQUEST_COUNT"

    def __str__(self):
        return self.value

    @classmethod
    def list_values(cls):
        return [item.value for item in cls]


class CPUFlavor(str, Enum):
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


class DataCenter(str, Enum):
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


class CUDAVersion(str, Enum):
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


def delete_runpod_template_by_name(template_name: str) -> bool:
    """
    Delete a RunPod template by name using REST API.

    According to RunPod documentation, templates must have unique names,
    and deletion is done by template ID via REST API.

    Args:
        template_name (str): The template name to delete

    Returns:
        bool: True if deletion was successful or template doesn't exist
    """
    try:
        # First, find the template by name to get its ID
        template = get_runpod_template_by_name(template_name)

        if not template:
            print(f"Template '{template_name}' not found (may not exist)")
            return True  # Treat as success if template doesn't exist

        template_id = template.get("id")

        # Delete the template using REST API
        result = make_runpod_api_call(f"templates/{template_id}", "DELETE")

        print(f"Template '{template_name}' (ID: {template_id}) deleted successfully")
        return True

    except Exception as e:
        print(f"Error deleting template '{template_name}': {e}")
        return False


def delete_runpod_endpoint_by_name(endpoint_name: str) -> bool:
    """
    Delete a RunPod endpoint by name using REST API.

    Endpoints must have unique names, so we need to delete existing ones
    before creating new ones with the same name.

    Args:
        endpoint_name (str): The endpoint name to delete

    Returns:
        bool: True if deletion was successful or endpoint doesn't exist
    """
    try:
        # First, get all endpoints to find the one with matching name
        result = make_runpod_api_call("endpoints", "GET")

        endpoints = result.get("endpoints", [])
        endpoint_id = None

        # Find endpoint with matching name
        for endpoint in endpoints:
            if endpoint.get("name") == endpoint_name:
                endpoint_id = endpoint.get("id")
                break

        if not endpoint_id:
            print(f"Endpoint '{endpoint_name}' not found (may not exist)")
            return True  # Treat as success if endpoint doesn't exist

        # Delete the endpoint using REST API
        delete_result = make_runpod_api_call(f"endpoints/{endpoint_id}", "DELETE")

        print(f"Endpoint '{endpoint_name}' (ID: {endpoint_id}) deleted successfully")
        return True

    except Exception as e:
        print(f"Error deleting endpoint '{endpoint_name}': {e}")
        return False


def get_runpod_template_by_name(template_name: str) -> dict | None:
    """
    Get a RunPod template by name using REST API.

    Args:
        template_name (str): The template name to find

    Returns:
        dict | None: Template data if found, None otherwise
    """
    try:
        # Get all templates using REST API
        result = make_runpod_api_call("templates", "GET")

        templates = result if isinstance(result, list) else result.get("templates", [])

        # Find template with matching name
        for template in templates:
            if template.get("name") == template_name:
                return template

        return None

    except Exception as e:
        print(f"Error fetching template '{template_name}': {e}")
        return None


def update_runpod_template(template_data: dict, image_name: str, tag: str) -> bool:
    """
    Update an existing RunPod template with a new Docker image using REST API.

    Args:
        template_data (dict): The existing template data with all current settings
        image_name (str): Name of the Docker image
        tag (str): Tag of the Docker image

    Returns:
        bool: True if update was successful
    """
    try:
        template_id = template_data["id"]

        # Prepare the update data - preserve existing settings but update image
        update_data = {
            "containerDiskInGb": template_data.get("containerDiskInGb", 20),
            "imageName": f"{image_name}:{tag}",
            "name": template_data.get("name"),
            "ports": template_data.get("ports", ["8000/http"]),
            "volumeInGb": template_data.get("volumeInGb", 0),
            "volumeMountPath": template_data.get("volumeMountPath", "/workspace"),
            "isPublic": template_data.get("isPublic", False),
        }

        # Handle environment variables - convert from GraphQL format to REST format if needed
        if template_data.get("env"):
            # Check if it's in GraphQL format (list of {key, value} objects)
            if (
                isinstance(template_data["env"], list)
                and template_data["env"]
                and "key" in template_data["env"][0]
            ):
                # Convert from GraphQL format to REST format (object with key-value pairs)
                env_dict = {}
                for env_item in template_data["env"]:
                    env_dict[env_item["key"]] = env_item["value"]
                update_data["env"] = env_dict
            else:
                # Already in REST format
                update_data["env"] = template_data["env"]
        else:
            update_data["env"] = {"PYTHONPATH": "/app"}

        # Add optional fields if they exist
        if template_data.get("dockerEntrypoint"):
            update_data["dockerEntrypoint"] = template_data["dockerEntrypoint"]
        if template_data.get("dockerStartCmd"):
            update_data["dockerStartCmd"] = template_data["dockerStartCmd"]
        if template_data.get("readme"):
            update_data["readme"] = template_data["readme"]

        print(f"Updating template with data:")
        print(json.dumps(update_data, indent=2))

        result = make_runpod_api_call(f"templates/{template_id}", "PATCH", update_data)

        print(
            f"Template '{template_data.get('name')}' updated with image: {image_name}:{tag}"
        )
        return True

    except Exception as e:
        print(f"Error updating template: {e}")
        return False


def create_or_update_runpod_template(
    template_name: str, image_name: str, tag: str
) -> str:
    """
    Create or update a RunPod template with the latest Docker image using REST API.

    This function first checks if a template with the given name exists:
    - If it exists, updates it with the new Docker image
    - If it doesn't exist, creates a new template

    This approach is more efficient than deleting and recreating templates,
    as it preserves template settings while updating the image.

    Template Configuration:
    - Container disk: 20GB for dependencies and temporary files
    - Environment variables: PYTHONPATH set to /app
    - Ports: ["8000/http"] for potential web interfaces
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
        Requires RUNPOD_API_KEY environment variable to be set.
    """
    # Check if template already exists
    print(f"Checking for existing template: {template_name}")
    existing_template = get_runpod_template_by_name(template_name)

    if existing_template:
        # Update existing template with new image
        template_id = existing_template["id"]
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
        # Create new template using REST API
        print(f"Template not found, creating new template: {template_name}")
        try:
            template_data = {
                "containerDiskInGb": 20,
                "imageName": f"{image_name}:{tag}",
                "name": template_name,
                "ports": ["8000/http"],
                "volumeInGb": 0,
                "volumeMountPath": "/workspace",
                "isPublic": False,
                "env": {"PYTHONPATH": "/app"},
            }

            print(f"Creating template with data:")
            print(json.dumps(template_data, indent=2))

            result = make_runpod_api_call("templates", "POST", template_data)
            template_id = result.get("id")

            if not template_id:
                print(f"Error: No template ID returned")
                print(f"Response data: {json.dumps(result, indent=2)}")
                sys.exit(1)

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
    Create a RunPod serverless endpoint using REST API.

    Creates a serverless endpoint that can execute the NodeTool workflow
    with auto-scaling capabilities and optional GPU acceleration.

    If an endpoint with the same name already exists, it will be deleted first.

    Args:
        template_id (str): The RunPod template ID from create_or_update_runpod_template()
        name (str): Name of the endpoint
        compute_type (str): Type of compute (CPU or GPU)
        gpu_type_ids (List[str], optional): List of GPU type IDs (e.g. ["NVIDIA GeForce RTX 4090"])
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
    # Handle network volume and data center coordination
    if network_volume_id:
        print(f"Network volume specified: {network_volume_id}")
        volume_info = get_network_volume(network_volume_id)
        volume_data_center = volume_info.get("dataCenterId")
        if volume_data_center:
            print(f"Network volume is in data center: {volume_data_center}")
            # Override data_center_ids to match the network volume location
            data_center_ids = [volume_data_center]
            print(f"Endpoint will be deployed to same region: {volume_data_center}")

    # Set sensible defaults - convert from old enum format to actual GPU names
    if gpu_type_ids is None and compute_type == ComputeType.GPU.value:
        gpu_type_ids = ["NVIDIA GeForce RTX 4090"]  # Default GPU type

    if data_center_ids is None:
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
        print(
            f"  Data centers: {data_center_ids if data_center_ids else 'Auto-selected'}"
        )
        print(f"  Workers: {workers_min}-{workers_max}")
        print(f"  Scaler: {scaler_type} (threshold: {scaler_value})")

        # Prepare the request data according to RunPod REST API format
        endpoint_data = {
            "computeType": compute_type,
            "templateId": template_id,
            "name": name,
            "workersMin": workers_min,
            "workersMax": workers_max,
            "scalerType": scaler_type,
            "scalerValue": scaler_value,
            "idleTimeout": idle_timeout,
            "flashboot": flashboot,
        }

        # Add compute-specific fields
        if compute_type == ComputeType.GPU.value:
            endpoint_data["gpuTypeIds"] = gpu_type_ids or ["NVIDIA GeForce RTX 4090"]
            endpoint_data["gpuCount"] = gpu_count or 1
        else:
            endpoint_data["cpuFlavorIds"] = cpu_flavor_ids or ["cpu3c"]
            if vcpu_count:
                endpoint_data["vcpuCount"] = vcpu_count

        # Add optional fields
        if allowed_cuda_versions:
            endpoint_data["allowedCudaVersions"] = allowed_cuda_versions

        if data_center_ids:
            endpoint_data["dataCenterIds"] = data_center_ids

        if execution_timeout_ms:
            endpoint_data["executionTimeoutMs"] = execution_timeout_ms

        if network_volume_id:
            endpoint_data["networkVolumeId"] = network_volume_id

        print(f"\nCreating endpoint with data:")
        print(json.dumps(endpoint_data, indent=2))

        result = make_runpod_api_call("endpoints", "POST", endpoint_data)

        endpoint_id = result.get("id")
        if not endpoint_id:
            print(f"Error: No endpoint ID returned")
            print(f"Response data: {json.dumps(result, indent=2)}")
            sys.exit(1)

        print(f"\nEndpoint created successfully!")
        print(f"  ID: {endpoint_id}")
        print(f"  Name: {result.get('name', name)}")
        print(f"  GPU Configuration: {result.get('gpuTypeIds')}")
        print(f"  Workers: {result.get('workersMin')}-{result.get('workersMax')}")

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
