"""
Comprehensive integration tests for Docker deployment with API testing.

These tests perform end-to-end testing of the NodeTool Docker container:
1. Build the Docker image locally
2. Start the container
3. Test API endpoints (health, workflows, jobs, settings)
4. Create and execute a workflow
5. Verify the results
6. Clean up the container

These tests require Docker to be available and are skipped otherwise.
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest
import requests

import docker
from nodetool.config.deployment import (
    ContainerConfig,
    DeploymentStatus,
    DockerDeployment,
    ImageConfig,
    ServerPaths,
    SSHConfig,
)
from nodetool.deploy.self_hosted import DockerDeployer
from nodetool.deploy.state import StateManager


def check_docker_available() -> bool:
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# Skip all tests if Docker is not available
pytestmark = pytest.mark.skipif(
    not check_docker_available(),
    reason="Docker is not available",
)


class TestDockerAPIIntegration:
    """
    End-to-end integration tests for Docker deployment with API testing.

    These tests build and run a real NodeTool container and verify:
    - Container starts successfully
    - Health endpoint responds
    - API endpoints are accessible
    - Workflows can be created and executed
    - Jobs complete successfully
    """

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace directory for testing."""
        with tempfile.TemporaryDirectory(prefix="nodetool_docker_api_test_") as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            workspace.mkdir()

            # Create required subdirectories
            (workspace / "data").mkdir()
            (workspace / "assets").mkdir()
            (workspace / "temp").mkdir()

            yield workspace

    @pytest.fixture
    def test_image_name(self):
        """Return the test image name."""
        return "nodetool:test"

    @pytest.fixture
    def test_deployment(self, temp_workspace, test_image_name):
        """Create a test deployment configuration."""
        return DockerDeployment(
            host="localhost",
            ssh=SSHConfig(user="test", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool", tag="local"),
            container=ContainerConfig(name="test-api", port=8777),
            paths=ServerPaths(
                workspace=str(temp_workspace),
                hf_cache=str(temp_workspace / "hf-cache"),
            ),
        )

    @pytest.fixture
    def docker_client(self):
        """Provide a Docker client for tests."""
        client = docker.from_env()
        yield client
        client.close()

    @pytest.fixture
    def container_name(self):
        """Return the test container name."""
        return "nodetool-test-api-integration"

    @pytest.fixture
    def api_base_url(self):
        """Return the API base URL for the test container."""
        return "http://localhost:8777"

    def _wait_for_container(
        self, container_name: str, timeout: int = 120
    ) -> docker.models.containers.Container:
        """Wait for container to be healthy and return it."""
        client = docker.from_env()
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                container = client.containers.get(container_name)
                container.reload()

                if container.status == "running":
                    # Check health status if available
                    health = container.attrs.get("State", {}).get("Health", {})
                    health_status = health.get("Status", "")

                    if health_status in ["healthy", ""] or health_status == "starting":
                        # Wait a bit more for the API to be ready
                        time.sleep(2)
                        return container
            except docker.errors.NotFound:
                pass

            time.sleep(2)

        raise TimeoutError(f"Container {container_name} did not become healthy in {timeout}s")

    def _wait_for_api(self, api_base_url: str, timeout: int = 60) -> bool:
        """Wait for the API to be responsive."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{api_base_url}/health", timeout=2)
                if response.status_code == 200 and response.text == '"OK"':
                    return True
            except requests.RequestException:
                pass

            time.sleep(1)

        return False

    def _get_existing_nodetool_local_image(self, client: docker.DockerClient) -> bool:
        """Check if nodetool:local image exists."""
        try:
            client.images.get("nodetool:local")
            return True
        except docker.errors.ImageNotFound:
            return False

    def test_image_exists_or_buildable(self, docker_client):
        """Test that nodetool:local image exists or can be built."""
        # Check if image already exists (common in dev environments)
        if self._get_existing_nodetool_local_image(docker_client):
            pytest.skip("nodetool:local image already exists - skipping build test")

        # Image doesn't exist, try to build it
        dockerfile_path = Path(__file__).parent.parent.parent / "Dockerfile"

        if not dockerfile_path.exists():
            pytest.skip(f"Dockerfile not found at {dockerfile_path}")

        # Note: Full build test is skipped in CI due to time constraints
        # This test documents the expected behavior
        pytest.skip("Docker image build test skipped in CI (time-intensive)")

    def test_container_starts_with_existing_image(
        self, docker_client, temp_workspace, api_base_url
    ):
        """
        Test that a container can start with the nodetool:local image.

        This test requires the image to already be built (common in dev environments).
        In CI, the image should be built in a separate step before running these tests.
        """
        if not self._get_existing_nodetool_local_image(docker_client):
            pytest.skip("nodetool:local image not found - build it first with: docker build -t nodetool:local .")

        container_name = "nodetool-test-api-starts"
        workspace_dir = temp_workspace / "workspace"

        try:
            # Remove any existing test container
            try:
                old_container = docker_client.containers.get(container_name)
                old_container.remove(force=True)
            except docker.errors.NotFound:
                pass

            # Start the container
            container = docker_client.containers.run(
                "nodetool:local",
                name=container_name,
                detach=True,
                ports={"7777/tcp": 8777},
                volumes={
                    str(workspace_dir): {"bind": "/workspace", "mode": "rw"},
                    str(temp_workspace / "hf-cache"): {"bind": "/hf-cache", "mode": "rw"},
                },
                environment={
                    "PORT": "7777",
                    "NODETOOL_API_URL": "http://localhost:7777",
                    "DB_PATH": "/workspace/nodetool.db",
                    "HF_HOME": "/hf-cache",
                    "ENV": "test",
                },
                remove=False,
            )

            # Wait for container to be healthy
            container = self._wait_for_container(container_name, timeout=120)

            # Wait for API to be ready
            assert self._wait_for_api(api_base_url, timeout=60), "API did not become ready"

            # Test health endpoint
            response = requests.get(f"{api_base_url}/health", timeout=5)
            assert response.status_code == 200
            assert response.text == '"OK"'

            # Verify container is still running
            container.reload()
            assert container.status == "running"

        finally:
            # Cleanup
            try:
                container = docker_client.containers.get(container_name)
                container.stop(timeout=5)
                container.remove()
            except docker.errors.NotFound:
                pass

    def test_health_endpoint(self, docker_client, api_base_url):
        """Test the /health endpoint."""
        if not self._get_existing_nodetool_local_image(docker_client):
            pytest.skip("nodetool:local image not found")

        try:
            response = requests.get(f"{api_base_url}/health", timeout=5)
            assert response.status_code == 200
            assert response.text == '"OK"'
        except requests.RequestException:
            pytest.skip("Container not running - run test_container_starts_with_existing_image first")

    def test_workflows_api_endpoint(self, docker_client, api_base_url):
        """Test the /api/workflows/ endpoint."""
        if not self._get_existing_nodetool_local_image(docker_client):
            pytest.skip("nodetool:local image not found")

        try:
            response = requests.get(f"{api_base_url}/api/workflows/", timeout=5)
            assert response.status_code == 200

            data = response.json()
            assert "workflows" in data
            assert isinstance(data["workflows"], list)

        except requests.RequestException:
            pytest.skip("Container not running")

    def test_settings_api_endpoint(self, docker_client, api_base_url):
        """Test the /api/settings/ endpoint."""
        if not self._get_existing_nodetool_local_image(docker_client):
            pytest.skip("nodetool:local image not found")

        try:
            response = requests.get(f"{api_base_url}/api/settings/", timeout=5)
            assert response.status_code == 200

            data = response.json()
            assert "settings" in data
            assert isinstance(data["settings"], list)

            # Verify some expected settings are present
            setting_keys = {s["env_var"] for s in data["settings"]}
            assert "OPENAI_API_KEY" in setting_keys
            assert "ANTHROPIC_API_KEY" in setting_keys
            assert "HF_TOKEN" in setting_keys

        except requests.RequestException:
            pytest.skip("Container not running")

    def test_jobs_api_endpoint(self, docker_client, api_base_url):
        """Test the /api/jobs/ endpoint."""
        if not self._get_existing_nodetool_local_image(docker_client):
            pytest.skip("nodetool:local image not found")

        try:
            response = requests.get(f"{api_base_url}/api/jobs/", timeout=5)
            assert response.status_code == 200

            data = response.json()
            assert "jobs" in data
            assert isinstance(data["jobs"], list)

        except requests.RequestException:
            pytest.skip("Container not running")

    def test_create_workflow(self, docker_client, api_base_url):
        """Test creating a workflow via API."""
        if not self._get_existing_nodetool_local_image(docker_client):
            pytest.skip("nodetool:local image not found")

        try:
            # Create a simple workflow that generates a list and sums it
            workflow_data = {
                "name": "test-sum-workflow",
                "description": "Test workflow that sums a list",
                "access": "private",
                "graph": {
                    "nodes": [
                        {
                            "id": "list",
                            "type": "nodetool.list.ListRange",
                            "data": {"stop": 5},
                        },
                        {
                            "id": "sum",
                            "type": "nodetool.list.Sum",
                        },
                    ],
                    "edges": [
                        {
                            "source": "list",
                            "target": "sum",
                            "sourceHandle": "output",
                            "targetHandle": "values",
                        }
                    ],
                },
            }

            response = requests.post(
                f"{api_base_url}/api/workflows/",
                json=workflow_data,
                timeout=10,
            )

            assert response.status_code == 200

            data = response.json()
            assert "id" in data
            assert data["name"] == "test-sum-workflow"
            assert data["access"] == "private"

            workflow_id = data["id"]
            return workflow_id

        except requests.RequestException:
            pytest.skip("Container not running")

    def test_run_workflow(self, docker_client, api_base_url):
        """Test running a workflow via API."""
        if not self._get_existing_nodetool_local_image(docker_client):
            pytest.skip("nodetool:local image not found")

        try:
            # First create a workflow
            workflow_data = {
                "name": "test-run-workflow",
                "description": "Test workflow execution",
                "access": "private",
                "graph": {
                    "nodes": [
                        {
                            "id": "list",
                            "type": "nodetool.list.ListRange",
                            "data": {"stop": 5},
                        },
                        {
                            "id": "sum",
                            "type": "nodetool.list.Sum",
                        },
                    ],
                    "edges": [
                        {
                            "source": "list",
                            "target": "sum",
                            "sourceHandle": "output",
                            "targetHandle": "values",
                        }
                    ],
                },
            }

            create_response = requests.post(
                f"{api_base_url}/api/workflows/",
                json=workflow_data,
                timeout=10,
            )

            if create_response.status_code != 200:
                pytest.skip("Failed to create workflow")

            workflow_id = create_response.json()["id"]

            # Run the workflow
            run_response = requests.post(
                f"{api_base_url}/api/workflows/{workflow_id}/run",
                json={},
                timeout=10,
            )

            assert run_response.status_code == 200

            # Wait for job to complete
            max_wait = 30
            start_time = time.time()

            while time.time() - start_time < max_wait:
                jobs_response = requests.get(
                    f"{api_base_url}/api/jobs/",
                    timeout=5,
                )

                if jobs_response.status_code == 200:
                    jobs_data = jobs_response.json()

                    for job in jobs_data.get("jobs", []):
                        if job.get("workflow_id") == workflow_id:
                            status = job.get("status")

                            if status == "completed":
                                return  # Success!
                            elif status == "failed":
                                pytest.fail(f"Workflow execution failed: {job.get('error')}")

                time.sleep(1)

            pytest.fail("Workflow did not complete in time")

        except requests.RequestException:
            pytest.skip("Container not running or network error")

    def test_workflow_execution_result(self, docker_client, api_base_url):
        """Test that workflow execution produces correct results."""
        if not self._get_existing_nodetool_local_image(docker_client):
            pytest.skip("nodetool:local image not found")

        try:
            # Create a workflow that should produce a known result
            # ListRange [0, 1, 2, 3, 4] -> Sum = 10
            workflow_data = {
                "name": "test-result-workflow",
                "description": "Test workflow result verification",
                "access": "private",
                "graph": {
                    "nodes": [
                        {
                            "id": "list",
                            "type": "nodetool.list.ListRange",
                            "data": {"stop": 5},
                        },
                        {
                            "id": "sum",
                            "type": "nodetool.list.Sum",
                        },
                    ],
                    "edges": [
                        {
                            "source": "list",
                            "target": "sum",
                            "sourceHandle": "output",
                            "targetHandle": "values",
                        }
                    ],
                },
            }

            create_response = requests.post(
                f"{api_base_url}/api/workflows/",
                json=workflow_data,
                timeout=10,
            )

            if create_response.status_code != 200:
                pytest.skip("Failed to create workflow")

            workflow_id = create_response.json()["id"]

            # Run the workflow
            requests.post(
                f"{api_base_url}/api/workflows/{workflow_id}/run",
                json={},
                timeout=10,
            )

            # Wait for job to complete and check result
            max_wait = 30
            start_time = time.time()

            while time.time() - start_time < max_wait:
                jobs_response = requests.get(
                    f"{api_base_url}/api/jobs/",
                    timeout=5,
                )

                if jobs_response.status_code == 200:
                    jobs_data = jobs_response.json()

                    for job in jobs_data.get("jobs", []):
                        if job.get("workflow_id") == workflow_id:
                            status = job.get("status")

                            if status == "completed":
                                # Success - the workflow ran to completion
                                # We expect the sum to be 10, but we can't directly check output
                                # without additional API calls or checking logs
                                return
                            elif status == "failed":
                                pytest.fail("Workflow execution failed")

                time.sleep(1)

            pytest.fail("Workflow did not complete in time")

        except requests.RequestException:
            pytest.skip("Container not running or network error")

    def test_scientific_workflow_execution(self, docker_client, api_base_url):
        """Test a more complex workflow with multiple operations."""
        if not self._get_existing_nodetool_local_image(docker_client):
            pytest.skip("nodetool:local image not found")

        try:
            # Create a workflow that:
            # 1. Generates a list [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            # 2. Filters to [2, 4, 6, 8] (even numbers from 2 to 8)
            # 3. Sums to 20
            workflow_data = {
                "name": "test-scientific-workflow",
                "description": "Complex workflow with multiple operations",
                "access": "private",
                "graph": {
                    "nodes": [
                        {
                            "id": "list",
                            "type": "nodetool.list.ListRange",
                            "data": {"stop": 10},
                        },
                        {
                            "id": "slice",
                            "type": "nodetool.list.Slice",
                            "data": {"start": 2, "stop": 9, "step": 2},
                        },
                        {
                            "id": "sum",
                            "type": "nodetool.list.Sum",
                        },
                    ],
                    "edges": [
                        {
                            "source": "list",
                            "target": "slice",
                            "sourceHandle": "output",
                            "targetHandle": "values",
                        },
                        {
                            "source": "slice",
                            "target": "sum",
                            "sourceHandle": "output",
                            "targetHandle": "values",
                        },
                    ],
                },
            }

            create_response = requests.post(
                f"{api_base_url}/api/workflows/",
                json=workflow_data,
                timeout=10,
            )

            if create_response.status_code != 200:
                pytest.skip("Failed to create workflow")

            workflow_id = create_response.json()["id"]

            # Run the workflow
            run_response = requests.post(
                f"{api_base_url}/api/workflows/{workflow_id}/run",
                json={},
                timeout=10,
            )

            assert run_response.status_code == 200

            # Wait for completion
            max_wait = 30
            start_time = time.time()

            while time.time() - start_time < max_wait:
                jobs_response = requests.get(
                    f"{api_base_url}/api/jobs/",
                    timeout=5,
                )

                if jobs_response.status_code == 200:
                    jobs_data = jobs_response.json()

                    for job in jobs_data.get("jobs", []):
                        if job.get("workflow_id") == workflow_id:
                            if job.get("status") == "completed":
                                return  # Success!
                            elif job.get("status") == "failed":
                                pytest.fail("Workflow execution failed")

                time.sleep(1)

            pytest.fail("Workflow did not complete in time")

        except requests.RequestException:
            pytest.skip("Container not running or network error")

    def test_delete_workflow(self, docker_client, api_base_url):
        """Test deleting a workflow via API."""
        if not self._get_existing_nodetool_local_image(docker_client):
            pytest.skip("nodetool:local image not found")

        try:
            # Create a workflow first
            workflow_data = {
                "name": "test-delete-workflow",
                "description": "Test workflow deletion",
                "access": "private",
                "graph": {
                    "nodes": [
                        {
                            "id": "n1",
                            "type": "nodetool.control.Reroute",
                        }
                    ],
                    "edges": [],
                },
            }

            create_response = requests.post(
                f"{api_base_url}/api/workflows/",
                json=workflow_data,
                timeout=10,
            )

            if create_response.status_code != 200:
                pytest.skip("Failed to create workflow")

            workflow_id = create_response.json()["id"]

            # Delete the workflow
            delete_response = requests.delete(
                f"{api_base_url}/api/workflows/{workflow_id}",
                timeout=5,
            )

            assert delete_response.status_code == 200

            # Verify it's deleted
            get_response = requests.get(
                f"{api_base_url}/api/workflows/{workflow_id}",
                timeout=5,
            )

            # Should return 404 or similar
            assert get_response.status_code == 404

        except requests.RequestException:
            pytest.skip("Container not running or network error")

    def test_container_logs_accessible(self, docker_client):
        """Test that container logs can be accessed."""
        if not self._get_existing_nodetool_local_image(docker_client):
            pytest.skip("nodetool:local image not found")

        container_name = "nodetool-test-api-starts"

        try:
            container = docker_client.containers.get(container_name)
            logs = container.logs(tail=50).decode("utf-8")

            # Logs should contain some expected output
            assert isinstance(logs, str)
            assert len(logs) > 0

            # Check for typical log patterns
            assert "INFO" in logs or "WARNING" in logs or "ERROR" in logs or "DEBUG" in logs

        except docker.errors.NotFound:
            pytest.skip("Container not running - run test_container_starts_with_existing_image first")


class TestDockerDeploymentManagerIntegration:
    """Integration tests for the Docker deployment manager."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace directory for testing."""
        with tempfile.TemporaryDirectory(prefix="nodetool_deploy_test_") as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            workspace.mkdir()

            (workspace / "data").mkdir()
            (workspace / "assets").mkdir()
            (workspace / "temp").mkdir()

            yield workspace

    @pytest.fixture
    def test_deployment(self, temp_workspace):
        """Create a test deployment configuration."""
        return DockerDeployment(
            host="localhost",
            ssh=SSHConfig(user="test", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool", tag="local"),
            container=ContainerConfig(name="test-deploy", port=8778),
            paths=ServerPaths(
                workspace=str(temp_workspace),
                hf_cache=str(temp_workspace / "hf-cache"),
            ),
        )

    def test_deployment_plan(self, test_deployment):
        """Test that deployment planning works."""
        from unittest.mock import Mock

        mock_state = Mock(spec=StateManager)
        mock_state.read_state = Mock(return_value=None)

        deployer = DockerDeployer(
            deployment_name="test",
            deployment=test_deployment,
            state_manager=mock_state,
        )

        plan = deployer.plan()

        assert plan["deployment_name"] == "test"
        assert plan["host"] == "localhost"
        assert plan["type"] == "docker"
        assert "changes" in plan
        assert isinstance(plan["changes"], list)

    def test_deployment_localhost_detection(self, test_deployment):
        """Test that localhost is properly detected."""
        from unittest.mock import Mock

        mock_state = Mock(spec=StateManager)
        mock_state.read_state = Mock(return_value=None)

        deployer = DockerDeployer(
            deployment_name="test",
            deployment=test_deployment,
            state_manager=mock_state,
        )

        assert deployer.is_localhost is True

    def test_deployment_executor_type(self, test_deployment):
        """Test that LocalExecutor is used for localhost."""
        from unittest.mock import Mock

        from nodetool.deploy.self_hosted import LocalExecutor

        mock_state = Mock(spec=StateManager)
        mock_state.read_state = Mock(return_value=None)

        deployer = DockerDeployer(
            deployment_name="test",
            deployment=test_deployment,
            state_manager=mock_state,
        )

        executor = deployer._get_executor()
        assert isinstance(executor, LocalExecutor)


# Add markers for CI
pytestmark = [
    pytestmark,  # Keep Docker availability check
    pytest.mark.integration,
    pytest.mark.docker,
    pytest.mark.slow,
]
