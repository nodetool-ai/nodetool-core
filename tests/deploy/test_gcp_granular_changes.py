import unittest
from unittest.mock import MagicMock, patch

from nodetool.config.deployment import (
    GCPDeployment,
    GCPIAMConfig,
    GCPImageConfig,
    GCPResourceConfig,
    GCPStorageConfig,
)
from nodetool.deploy.gcp import GCPDeployer
from nodetool.deploy.state import StateManager


class TestGCPGranularChanges(unittest.TestCase):
    def setUp(self):
        self.deployment = GCPDeployment(
            project_id="test-project",
            service_name="test-service",
            image=GCPImageConfig(
                registry="us-docker.pkg.dev",
                repository="nodetool/test-service",
                tag="latest",
            ),
            resources=GCPResourceConfig(
                cpu="4",
                memory="16Gi",
                min_instances=1,
                max_instances=5,
                concurrency=80,
                timeout=3600,
                gpu_type="nvidia-l4",  # Implies gpu_count=1
            ),
            storage=GCPStorageConfig(
                gcs_bucket="test-bucket",
            ),
            iam=GCPIAMConfig(
                service_account="test-sa@test-project.iam.gserviceaccount.com",
            ),
        )

        self.state_manager_mock = MagicMock(spec=StateManager)
        self.state_manager_mock.read_state.return_value = {
            "last_deployed": "2023-01-01T00:00:00",
            "status": "serving",
        }

        self.deployer = GCPDeployer(
            deployment_name="test-deployment",
            deployment=self.deployment,
            state_manager=self.state_manager_mock,
        )

        self.expected_env_list = [
            {"name": "NODETOOL_SERVER_MODE", "value": "private"},
            {"name": "HF_HOME", "value": "/mnt/gcs/.cache/huggingface"},
            {"name": "HF_HUB_CACHE", "value": "/mnt/gcs/.cache/huggingface/hub"},
            {"name": "TRANSFORMERS_CACHE", "value": "/mnt/gcs/.cache/transformers"},
            {"name": "OLLAMA_MODELS", "value": "/mnt/gcs/.ollama/models"},
            {"name": "AUTH_PROVIDER", "value": "static"},
        ]

    def _get_base_service_dict(self):
        return {
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "autoscaling.knative.dev/minScale": "1",
                            "autoscaling.knative.dev/maxScale": "5",
                            "run.googleapis.com/gpu": "1",
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "image": "us-docker.pkg.dev/test-project/nodetool/test-service:latest",
                                "resources": {"limits": {"cpu": "4000m", "memory": "16Gi"}},
                                "env": self.expected_env_list,
                            }
                        ],
                        "serviceAccountName": "test-sa@test-project.iam.gserviceaccount.com",
                        "containerConcurrency": 80,
                        "timeoutSeconds": 3600,
                    },
                }
            }
        }

    @patch("nodetool.deploy.gcp.get_cloud_run_service")
    def test_no_changes_detected(self, mock_get_service):
        mock_get_service.return_value = self._get_base_service_dict()

        plan = self.deployer.plan()

        self.assertIn("No configuration changes detected", plan["changes"])
        self.assertEqual(len(plan["will_update"]), 0)

    @patch("nodetool.deploy.gcp.get_cloud_run_service")
    def test_cpu_change_detected(self, mock_get_service):
        service_dict = self._get_base_service_dict()
        # Change remote CPU to 2 (2000m)
        service_dict["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["cpu"] = "2000m"
        mock_get_service.return_value = service_dict

        plan = self.deployer.plan()

        self.assertTrue(any("CPU: 2 -> 4" in s for s in plan["changes"]))
        self.assertIn(f"Cloud Run service: {self.deployment.service_name}", plan["will_update"])

    @patch("nodetool.deploy.gcp.get_cloud_run_service")
    def test_memory_change_detected(self, mock_get_service):
        service_dict = self._get_base_service_dict()
        # Change remote memory to 8Gi
        service_dict["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["memory"] = "8Gi"
        mock_get_service.return_value = service_dict

        plan = self.deployer.plan()

        self.assertTrue(any("Memory: 8Gi -> 16Gi" in s for s in plan["changes"]))
        self.assertIn(f"Cloud Run service: {self.deployment.service_name}", plan["will_update"])

    @patch("nodetool.deploy.gcp.get_cloud_run_service")
    def test_min_instances_change_detected(self, mock_get_service):
        service_dict = self._get_base_service_dict()
        # Change remote min scale to 0
        service_dict["spec"]["template"]["metadata"]["annotations"]["autoscaling.knative.dev/minScale"] = "0"
        mock_get_service.return_value = service_dict

        plan = self.deployer.plan()

        self.assertTrue(any("Min Instances: 0 -> 1" in s for s in plan["changes"]))

    @patch("nodetool.deploy.gcp.get_cloud_run_service")
    def test_env_var_added_locally(self, mock_get_service):
        service_dict = self._get_base_service_dict()
        # Remove AUTH_PROVIDER from remote
        env_list = [
            e for e in service_dict["spec"]["template"]["spec"]["containers"][0]["env"] if e["name"] != "AUTH_PROVIDER"
        ]
        service_dict["spec"]["template"]["spec"]["containers"][0]["env"] = env_list
        mock_get_service.return_value = service_dict

        plan = self.deployer.plan()

        self.assertTrue(any("added AUTH_PROVIDER" in s for s in plan["changes"]))

    @patch("nodetool.deploy.gcp.get_cloud_run_service")
    def test_env_var_value_changed(self, mock_get_service):
        service_dict = self._get_base_service_dict()
        # Change remote AUTH_PROVIDER to 'supabase'
        for e in service_dict["spec"]["template"]["spec"]["containers"][0]["env"]:
            if e["name"] == "AUTH_PROVIDER":
                e["value"] = "supabase"
        mock_get_service.return_value = service_dict

        plan = self.deployer.plan()

        self.assertTrue(any("changed AUTH_PROVIDER" in s for s in plan["changes"]))

    @patch("nodetool.deploy.gcp.get_cloud_run_service")
    def test_gpu_count_change(self, mock_get_service):
        service_dict = self._get_base_service_dict()
        # Change remote GPU to 0
        service_dict["spec"]["template"]["metadata"]["annotations"]["run.googleapis.com/gpu"] = "0"
        mock_get_service.return_value = service_dict

        plan = self.deployer.plan()

        self.assertTrue(any("GPU Count: 0 -> 1" in s for s in plan["changes"]))
