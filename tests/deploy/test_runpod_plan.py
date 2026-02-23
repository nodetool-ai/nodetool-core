import unittest
from unittest.mock import MagicMock, patch
from nodetool.deploy.runpod import RunPodDeployer
from nodetool.config.deployment import RunPodDeployment, RunPodImageConfig

class TestRunPodPlan(unittest.TestCase):
    def setUp(self):
        self.deployment = RunPodDeployment(
            image=RunPodImageConfig(name="my-repo/my-image", tag="v1"),
            gpu_types=["AMPERE_16"],
            workers_min=0,
            workers_max=1,
            idle_timeout=30,
            flashboot=False,
            environment={"KEY": "VALUE"},
        )
        self.deployment_name = "test-deployment"
        self.state_manager = MagicMock()
        self.deployer = RunPodDeployer(
            deployment_name=self.deployment_name,
            deployment=self.deployment,
            state_manager=self.state_manager,
        )

    @patch("nodetool.deploy.runpod.get_runpod_endpoint_by_name")
    @patch("nodetool.deploy.runpod.get_runpod_template_by_name")
    def test_plan_no_changes(self, mock_get_template, mock_get_endpoint):
        # Mock state to simulate existing deployment
        self.state_manager.read_state.return_value = {"last_deployed": "some_date"}

        # Mock API responses
        mock_get_endpoint.return_value = {
            "workersMin": 0,
            "workersMax": 1,
            "idleTimeout": 30,
            "gpuIds": "AMPERE_16",
            "flashboot": False,
        }
        mock_get_template.return_value = {
            "imageName": "my-repo/my-image:v1",
            "env": [{"key": "KEY", "value": "VALUE"}, {"key": "PORT", "value": "8000"}, {"key": "PORT_HEALTH", "value": "8000"}],
        }

        plan = self.deployer.plan()

        self.assertIn("No configuration changes detected", plan["changes"])
        self.assertEqual(len(plan["will_update"]), 0)

    @patch("nodetool.deploy.runpod.get_runpod_endpoint_by_name")
    @patch("nodetool.deploy.runpod.get_runpod_template_by_name")
    def test_plan_worker_change(self, mock_get_template, mock_get_endpoint):
        self.state_manager.read_state.return_value = {"last_deployed": "some_date"}

        # Endpoint has different max workers
        mock_get_endpoint.return_value = {
            "workersMin": 0,
            "workersMax": 5, # changed
            "idleTimeout": 30,
            "gpuIds": "AMPERE_16",
            "flashboot": False,
        }
        mock_get_template.return_value = {
            "imageName": "my-repo/my-image:v1",
            "env": [{"key": "KEY", "value": "VALUE"}, {"key": "PORT", "value": "8000"}, {"key": "PORT_HEALTH", "value": "8000"}],
        }

        plan = self.deployer.plan()

        self.assertTrue(any("Max workers changed" in c for c in plan["changes"]))
        self.assertIn("Max workers", plan["will_update"])

    @patch("nodetool.deploy.runpod.get_runpod_endpoint_by_name")
    @patch("nodetool.deploy.runpod.get_runpod_template_by_name")
    def test_plan_image_change(self, mock_get_template, mock_get_endpoint):
        self.state_manager.read_state.return_value = {"last_deployed": "some_date"}

        mock_get_endpoint.return_value = {
            "workersMin": 0,
            "workersMax": 1,
            "idleTimeout": 30,
            "gpuIds": "AMPERE_16",
            "flashboot": False,
        }
        # Template has different image
        mock_get_template.return_value = {
            "imageName": "my-repo/my-image:old", # changed
            "env": [{"key": "KEY", "value": "VALUE"}, {"key": "PORT", "value": "8000"}, {"key": "PORT_HEALTH", "value": "8000"}],
        }

        plan = self.deployer.plan()

        self.assertTrue(any("Docker image changed" in c for c in plan["changes"]))
        self.assertIn("Update template image", plan["will_update"])
        self.assertIn("Redeploy endpoint", plan["will_update"])

    @patch("nodetool.deploy.runpod.get_runpod_endpoint_by_name")
    @patch("nodetool.deploy.runpod.get_runpod_template_by_name")
    def test_plan_missing_endpoint(self, mock_get_template, mock_get_endpoint):
        self.state_manager.read_state.return_value = {"last_deployed": "some_date"}

        mock_get_endpoint.return_value = None # Endpoint missing

        plan = self.deployer.plan()

        self.assertTrue(any("Endpoint 'test-deployment' not found" in c for c in plan["changes"]))
        self.assertIn("Create endpoint", plan["will_update"])

    @patch("nodetool.deploy.runpod.get_runpod_endpoint_by_name")
    @patch("nodetool.deploy.runpod.get_runpod_template_by_name")
    def test_plan_env_change(self, mock_get_template, mock_get_endpoint):
        self.state_manager.read_state.return_value = {"last_deployed": "some_date"}

        mock_get_endpoint.return_value = {
            "workersMin": 0,
            "workersMax": 1,
            "idleTimeout": 30,
            "gpuIds": "AMPERE_16",
            "flashboot": False,
        }
        # Template has different env
        mock_get_template.return_value = {
            "imageName": "my-repo/my-image:v1",
            "env": [{"key": "KEY", "value": "OLD_VALUE"}, {"key": "PORT", "value": "8000"}, {"key": "PORT_HEALTH", "value": "8000"}], # changed
        }

        plan = self.deployer.plan()

        self.assertTrue(any("Environment variables changed" in c for c in plan["changes"]))
        self.assertIn("Update template environment", plan["will_update"])
