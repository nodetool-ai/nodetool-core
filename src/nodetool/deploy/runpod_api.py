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
        create_runpod_endpoint,
        invoke_streaming_endpoint
    )
"""

import json
import logging
import os
import sys
import traceback
from enum import Enum
from typing import Any, Optional

import requests
from runpod import error
from runpod.user_agent import USER_AGENT

logger = logging.getLogger("nodetool.runpod")

RUNPOD_REST_BASE_URL = "https://rest.runpod.io/v1"

HTTP_STATUS_UNAUTHORIZED = 401


def run_graphql_query(query: str) -> dict[str, Any]:
    """
    Run a GraphQL query
    """
    api_url_base = os.environ.get("RUNPOD_API_BASE_URL", "https://api.runpod.io")
    url = f"{api_url_base}/graphql"
    api_key = os.getenv("RUNPOD_API_KEY")
    assert api_key, "RUNPOD_API_KEY environment variable is not set"

    headers = {
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
        "Authorization": f"Bearer {api_key}",
    }

    data = json.dumps({"query": query})
    response = requests.post(url, headers=headers, data=data, timeout=30)

    if response.status_code == HTTP_STATUS_UNAUTHORIZED:
        raise error.AuthenticationError("Unauthorized request, please check your API key.")

    if "errors" in response.json():
        raise error.QueryError(response.json()["errors"][0]["message"], query)

    return response.json()


class GPUType(str, Enum):
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


def make_runpod_api_call(endpoint: str, method: str = "GET", data: dict | None = None) -> dict:
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
    api_key = os.getenv("RUNPOD_API_KEY")
    assert api_key, "RUNPOD_API_KEY environment variable is not set"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT
        or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        if method == "POST":
            response = requests.post(url, headers=headers, json=data, timeout=60)
        elif method == "PUT":
            response = requests.put(url, headers=headers, json=data, timeout=60)
        elif method == "PATCH":
            response = requests.patch(url, headers=headers, json=data, timeout=60)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, timeout=60)
        else:
            response = requests.get(url, headers=headers, timeout=60)

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
        logger.error("RunPod API call failed: %s", e)
        if hasattr(e, "response") and e.response is not None:
            logger.error("Response status: %s", e.response.status_code)
            logger.error("Response body: %s", e.response.text)
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
    data = {"dataCenterId": data_center_id, "name": name, "size": size}

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
        make_runpod_api_call(f"templates/{template_id}", "DELETE")

        print(f"Template '{template_name}' (ID: {template_id}) deleted successfully")
        return True

    except Exception as e:
        print(f"Error deleting template '{template_name}': {e}")
        return False


def get_runpod_endpoint_by_name(endpoint_name: str) -> dict | None:
    """
    Get a RunPod endpoint by name using REST API.

    Args:
        endpoint_name (str): The endpoint name to find

    Returns:
        dict | None: Endpoint data if found, None otherwise
    """
    # Get all endpoints to find the one with matching name
    endpoints = make_runpod_api_call("endpoints", "GET")

    print(f"🔍 Looking for endpoint: '{endpoint_name}'")
    print(f"📝 Found {len(endpoints)} total endpoints")

    # Debug: List all endpoint names
    if endpoints:
        print("📋 Available endpoints:")
        for i, endpoint in enumerate(endpoints):
            name = endpoint.get("name", "<no name>")
            endpoint_id = endpoint.get("id", "<no id>")
            print(f"  [{i + 1}] Name: '{name}' (ID: {endpoint_id})")

    # Find endpoint with matching name (exact match first)
    for endpoint in endpoints:
        if endpoint.get("name").startswith(endpoint_name):
            print(f"✅ Found exact match for endpoint: '{endpoint_name}'")
            return endpoint

    print(f"❌ No endpoint found with name: '{endpoint_name}'")
    return None


def update_runpod_endpoint(endpoint_id: str, template_id: str, **kwargs) -> bool:
    """
    Update an existing RunPod endpoint with a new template.

    Args:
        endpoint_id (str): The endpoint ID to update
        template_id (str): The new template ID
        **kwargs: Additional endpoint configuration options

    Returns:
        bool: True if update was successful
    """
    try:
        # Prepare update data with the new template
        update_data = {
            "templateId": template_id,
        }

        # Add any additional configuration options
        for key, value in kwargs.items():
            if value is not None:
                update_data[key] = value

        print(f"Updating endpoint {endpoint_id} with template {template_id}")
        make_runpod_api_call(f"endpoints/{endpoint_id}", "PATCH", update_data)

        print(f"Endpoint '{endpoint_id}' updated successfully with template: {template_id}")
        return True

    except Exception as e:
        print(f"Error updating endpoint '{endpoint_id}': {e}")
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

        print(f"🗑️ Looking for endpoint to delete: '{endpoint_name}'")
        print(f"📝 Found {len(endpoints)} total endpoints")

        # Find endpoint with matching name (exact match first)
        for endpoint in endpoints:
            if endpoint.get("name") == endpoint_name:
                endpoint_id = endpoint.get("id")
                print(f"✅ Found exact match for deletion: '{endpoint_name}' (ID: {endpoint_id})")
                break

        # Try case-insensitive match as fallback
        if not endpoint_id:
            for endpoint in endpoints:
                if endpoint.get("name", "").lower() == endpoint_name.lower():
                    endpoint_id = endpoint.get("id")
                    print(
                        f"✅ Found case-insensitive match for deletion: '{endpoint_name}' -> '{endpoint.get('name')}' (ID: {endpoint_id})"
                    )
                    break

        if not endpoint_id:
            print(f"❌ Endpoint '{endpoint_name}' not found (may not exist)")
            return True  # Treat as success if endpoint doesn't exist

        # Delete the endpoint using REST API
        make_runpod_api_call(f"endpoints/{endpoint_id}", "DELETE")

        print(f"🗑️ Endpoint '{endpoint_name}' (ID: {endpoint_id}) deleted successfully")
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
        env = template_data.get("env", {})

        env["PORT"] = "8000"
        env["PORT_HEALTH"] = "8000"

        # Prepare the update data - preserve existing settings but update image
        update_data = {
            "containerDiskInGb": template_data.get("containerDiskInGb", 20),
            "imageName": f"{image_name}:{tag}",
            "name": template_data.get("name"),
            "ports": template_data.get("ports", ["8000/http"]),
            "volumeInGb": template_data.get("volumeInGb", 0),
            "volumeMountPath": template_data.get("volumeMountPath", "/workspace"),
            "isPublic": template_data.get("isPublic", False),
            "env": env,
        }

        # Add optional fields if they exist
        if template_data.get("dockerEntrypoint"):
            update_data["dockerEntrypoint"] = template_data["dockerEntrypoint"]
        if template_data.get("dockerStartCmd"):
            update_data["dockerStartCmd"] = template_data["dockerStartCmd"]
        if template_data.get("readme"):
            update_data["readme"] = template_data["readme"]

        print("Updating template with data:")
        print(json.dumps(update_data, indent=2))

        make_runpod_api_call(f"templates/{template_id}", "PATCH", update_data)

        print(f"Template '{template_data.get('name')}' updated with image: {image_name}:{tag}")
        return True

    except Exception as e:
        print(f"Error updating template: {e}")
        return False


def create_or_update_runpod_template(
    template_name: str,
    image_name: str,
    tag: str,
    env: dict[str, str] | None = None,
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
        env (dict[str, str]): Environment variables to set in the template

    Returns:
        str: The template ID for use in endpoint creation

    Raises:
        SystemExit: If template creation/update fails or API key is invalid

    Note:
        Requires RUNPOD_API_KEY environment variable to be set.
    """
    env = env or {}
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
                "ports": ["8000/http", "22/ssh"],
                "volumeInGb": 0,
                "volumeMountPath": "/workspace",
                "isPublic": False,
                "env": env,
            }

            print("Creating template with data:")
            print(json.dumps(template_data, indent=2))

            result = make_runpod_api_call("templates", "POST", template_data)
            template_id = result.get("id")

            if not template_id:
                print("Error: No template ID returned")
                print(f"Response data: {json.dumps(result, indent=2)}")
                sys.exit(1)

            # template_id is str after the above check (sys.exit doesn't return)
            assert template_id is not None
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


def create_or_update_runpod_endpoint(
    template_id: str,
    name: str,
    compute_type: str = ComputeType.GPU.value,
    gpu_type_ids: Optional[list[str]] = None,
    gpu_count: Optional[int] = None,
    cpu_flavor_ids: Optional[list[str]] = None,
    vcpu_count: Optional[int] = None,
    data_center_ids: Optional[list[str]] = None,
    workers_min: int = 0,
    workers_max: int = 3,
    idle_timeout: int = 5,
    execution_timeout_ms: Optional[int] = None,
    flashboot: bool = False,
    network_volume_id: Optional[str] = None,
    allowed_cuda_versions: Optional[list[str]] = None,
):
    """
    Create or update a RunPod serverless endpoint using REST API.

    Creates a serverless endpoint that can execute the NodeTool workflow
    with auto-scaling capabilities and optional GPU acceleration.

    If an endpoint with the same name already exists, it will be updated with the new template.

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

    # Check if endpoint already exists
    print(f"Checking for existing endpoint: {name}")
    existing_endpoint = get_runpod_endpoint_by_name(name)

    if existing_endpoint:
        endpoint_id = existing_endpoint["id"]
        print(f"Found existing endpoint (ID: {endpoint_id})")
        print(f"Current template: {existing_endpoint.get('templateId', 'unknown')}")
        print(f"Updating with new template: {template_id}")

        if update_runpod_endpoint(endpoint_id, template_id):
            print(f"Endpoint '{name}' updated successfully")
            return endpoint_id
        else:
            print(f"Failed to update endpoint '{name}', creating new one...")
            # Fall through to create new endpoint after deleting the problematic one
            delete_runpod_endpoint_by_name(name)

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

        # Prepare the request data according to RunPod REST API format
        endpoint_data = {
            "computeType": compute_type,
            "templateId": template_id,
            "name": name,
            "workersMin": workers_min,
            "workersMax": workers_max,
            "scalerType": "REQUEST_COUNT",
            "scalerValue": 4,
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

        print("\nCreating endpoint with data:")
        print(json.dumps(endpoint_data, indent=2))

        result = make_runpod_api_call("endpoints", "POST", endpoint_data)

        endpoint_id = result.get("id")
        if not endpoint_id:
            print("Error: No endpoint ID returned")
            print(f"Response data: {json.dumps(result, indent=2)}")
            sys.exit(1)

        print("\nEndpoint created successfully!")
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


def create_runpod_endpoint_graphql(
    template_id: str,
    name: str,
    compute_type: str = ComputeType.GPU.value,
    gpu_type_ids: Optional[list[str]] = None,
    gpu_count: Optional[int] = None,
    cpu_flavor_ids: Optional[list[str]] = None,
    vcpu_count: Optional[int] = None,
    data_center_ids: Optional[list[str]] = None,
    workers_min: int = 0,
    workers_max: int = 3,
    idle_timeout: int = 5,
    execution_timeout_ms: Optional[int] = None,
    flashboot: bool = False,
    network_volume_id: Optional[str] = None,
    allowed_cuda_versions: Optional[list[str]] = None,
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
    api_key = os.getenv("RUNPOD_API_KEY")
    assert api_key, "RUNPOD_API_KEY environment variable is not set"

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
        print(f"  GPU types: {gpu_type_ids}")
        print(f"  GPU count: {gpu_count}")
        print(f"  CPU flavors: {cpu_flavor_ids}")
        print(f"  vCPU count: {vcpu_count}")
        print(f"  Data centers: {data_center_ids if data_center_ids else 'Auto-selected'}")
        print(f"  Workers: {workers_min}-{workers_max}")
        print(f"  Idle timeout: {idle_timeout}s")
        print(f"  Execution timeout: {execution_timeout_ms}ms")
        print(f"  Flashboot: {flashboot}")
        print(f"  Network volume: {network_volume_id}")
        print(f"  Allowed CUDA versions: {allowed_cuda_versions}")
        gpu_ids_str = ",".join(list(gpu_type_ids)) if gpu_type_ids else "AMPERE_24"
        # Build the GraphQL mutation for saveEndpoint
        # Build the mutation
        mutation_parts = []
        mutation_parts.append(f'name: "{name if not flashboot else f"{name}-fb"}"')
        mutation_parts.append(f'templateId: "{template_id}"')
        mutation_parts.append('type: "LB"')  # Load Balancer type
        mutation_parts.append(f"workersMin: {workers_min}")
        mutation_parts.append(f"workersMax: {workers_max}")
        mutation_parts.append(f"idleTimeout: {idle_timeout}")
        mutation_parts.append('scalerType: "REQUEST_COUNT"')
        mutation_parts.append("scalerValue: 4")
        mutation_parts.append(f"executionTimeoutMs: {execution_timeout_ms or 300000}")
        mutation_parts.append(f'gpuIds: "{gpu_ids_str}"')
        mutation_parts.append(f"gpuCount: {gpu_count or 1}")
        # mutation_parts.append(f"cpuFlavorIds: {json.dumps(cpu_flavor_ids)}")
        if vcpu_count:
            mutation_parts.append(f"vcpuCount: {vcpu_count}")

        # Add optional fields
        if data_center_ids:
            # Format locations as JSON array string
            locations_str = json.dumps(data_center_ids)
            mutation_parts.append(f"locations: {locations_str}")
        else:
            mutation_parts.append("locations: null")

        if network_volume_id:
            mutation_parts.append(f'networkVolumeId: "{network_volume_id}"')
        else:
            mutation_parts.append("networkVolumeId: null")

        # Build the complete mutation
        mutation = f"""
        mutation {{
            saveEndpoint(input: {{
                {", ".join(mutation_parts)}
            }}) {{
                id
                name
                gpuIds
                idleTimeout
                locations
                type
                networkVolumeId
                executionTimeoutMs
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

        print("\nExecuting GraphQL mutation:")
        print(mutation)

        result = run_graphql_query(mutation)

        if "errors" in result:
            error_msg = result["errors"][0].get("message", "Unknown error")
            print(f"Error creating endpoint: {error_msg}")
            print(f"Full error response: {json.dumps(result, indent=2)}")
            sys.exit(1)

        endpoint_data = result.get("data", {}).get("saveEndpoint", {})
        endpoint_id = endpoint_data.get("id")

        if not endpoint_id:
            print("Error: No endpoint ID returned")
            print(f"Response data: {json.dumps(result, indent=2)}")
            sys.exit(1)

        print("\nEndpoint created successfully!")
        print(f"  ID: {endpoint_id}")
        print(f"  Name: {endpoint_data.get('name')}")
        print(f"  GPU Configuration: {endpoint_data.get('gpuIds')}")
        print(f"  Workers: {endpoint_data.get('workersMin')}-{endpoint_data.get('workersMax')}")
        print(f"  Scaler: {endpoint_data.get('scalerType')}")

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
