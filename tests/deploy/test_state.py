"""
Unit tests for deployment state management.
"""

import threading
import time
from datetime import datetime

import pytest

from nodetool.config.deployment import (
    ContainerConfig,
    DeploymentConfig,
    ImageConfig,
    SelfHostedDeployment,
    SSHConfig,
)
from nodetool.deploy.state import (
    StateManager,
    create_state_snapshot,
    restore_state_from_snapshot,
)


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create a temporary config directory."""
    config_dir = tmp_path / ".config" / "nodetool"
    config_dir.mkdir(parents=True)
    return config_dir


@pytest.fixture
def sample_config(temp_config_dir):
    """Create a sample deployment configuration."""
    config = DeploymentConfig(
        deployments={
            "test-server": SelfHostedDeployment(
                host="192.168.1.100",
                ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
                image=ImageConfig(name="nodetool/nodetool", tag="latest"),
                container=ContainerConfig(name="wf1", port=8001, workflows=["abc123"]),
            ),
            "test-server-2": SelfHostedDeployment(
                host="192.168.1.101",
                ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
                image=ImageConfig(name="nodetool/nodetool", tag="latest"),
                container=ContainerConfig(name="wf2", port=8002, workflows=[]),
            ),
        }
    )

    config_path = temp_config_dir / "deployment.yaml"
    # Save config using YAML directly
    import yaml

    data = config.model_dump(mode="json", exclude_none=True)
    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    yield config, config_path


class TestStateManager:
    """Tests for StateManager class."""

    def test_state_manager_init(self, temp_config_dir):
        """Test StateManager initialization."""
        config_path = temp_config_dir / "deployment.yaml"
        manager = StateManager(config_path=config_path)

        assert manager.config_path == config_path
        assert manager.lock_path == config_path.with_suffix(".lock")

    def test_read_state(self, sample_config):
        """Test reading deployment state."""
        _config, config_path = sample_config
        manager = StateManager(config_path=config_path)

        state = manager.read_state("test-server")

        assert state is not None
        assert state["status"] == "unknown"
        assert "last_deployed" in state

    def test_read_state_nonexistent(self, sample_config):
        """Test reading state for nonexistent deployment."""
        _config, config_path = sample_config
        manager = StateManager(config_path=config_path)

        state = manager.read_state("nonexistent")

        assert state is None

    def test_write_state(self, sample_config):
        """Test writing deployment state."""
        _config, config_path = sample_config
        manager = StateManager(config_path=config_path)

        # Update state
        manager.write_state(
            "test-server",
            {"status": "running", "container_id": "abc123"},
            update_timestamp=False,
        )

        # Read back
        state = manager.read_state("test-server")

        assert state["status"] == "running"
        assert state["container_id"] == "abc123"

    def test_write_state_with_timestamp(self, sample_config):
        """Test writing state with automatic timestamp."""
        _config, config_path = sample_config
        manager = StateManager(config_path=config_path)

        before = datetime.utcnow()

        # Update state with timestamp
        manager.write_state("test-server", {"status": "running"}, update_timestamp=True)

        after = datetime.utcnow()

        # Read back
        state = manager.read_state("test-server")

        assert state["status"] == "running"
        assert state["last_deployed"] is not None

        # Check timestamp is recent
        last_deployed = datetime.fromisoformat(
            state["last_deployed"].replace("Z", "+00:00")
        )
        assert before <= last_deployed <= after

    def test_write_state_nonexistent_deployment(self, sample_config):
        """Test writing state for nonexistent deployment raises error."""
        _config, config_path = sample_config
        manager = StateManager(config_path=config_path)

        with pytest.raises(KeyError, match="Deployment 'nonexistent' not found"):
            manager.write_state("nonexistent", {"status": "running"})

    def test_update_deployment_status(self, sample_config):
        """Test updating deployment status."""
        _config, config_path = sample_config
        manager = StateManager(config_path=config_path)

        manager.update_deployment_status("test-server", "running")

        state = manager.read_state("test-server")
        assert state["status"] == "running"

    def test_get_all_states(self, sample_config):
        """Test getting all deployment states."""
        _config, config_path = sample_config
        manager = StateManager(config_path=config_path)

        # Update some states
        manager.write_state(
            "test-server", {"status": "running"}, update_timestamp=False
        )
        manager.write_state(
            "test-server-2", {"status": "stopped"}, update_timestamp=False
        )

        all_states = manager.get_all_states()

        assert len(all_states) == 2
        assert "test-server" in all_states
        assert "test-server-2" in all_states
        assert all_states["test-server"]["status"] == "running"
        assert all_states["test-server-2"]["status"] == "stopped"

    def test_clear_state(self, sample_config):
        """Test clearing deployment state."""
        _config, config_path = sample_config
        manager = StateManager(config_path=config_path)

        # Set some state
        manager.write_state("test-server", {"status": "running", "container_id": "abc"})

        # Clear it
        manager.clear_state("test-server")

        # State should be reset to defaults
        state = manager.read_state("test-server")
        assert state["status"] == "unknown"
        assert state["container_id"] is None

    def test_get_last_deployed(self, sample_config):
        """Test getting last deployed timestamp."""
        _config, config_path = sample_config
        manager = StateManager(config_path=config_path)

        # Initially None
        last = manager.get_last_deployed("test-server")
        assert last is None

        # Update with timestamp
        manager.write_state("test-server", {"status": "running"}, update_timestamp=True)

        # Should have timestamp now
        last = manager.get_last_deployed("test-server")
        assert last is not None
        assert isinstance(last, datetime)

    def test_has_been_deployed(self, sample_config):
        """Test checking if deployment has been deployed."""
        _config, config_path = sample_config
        manager = StateManager(config_path=config_path)

        # Initially false
        assert not manager.has_been_deployed("test-server")

        # Deploy
        manager.write_state("test-server", {"status": "running"}, update_timestamp=True)

        # Should be true now
        assert manager.has_been_deployed("test-server")


class TestStateLocking:
    """Tests for state file locking."""

    def test_lock_basic(self, sample_config):
        """Test basic lock acquisition and release."""
        _config, config_path = sample_config
        manager = StateManager(config_path=config_path)

        with manager.lock():
            # Lock should be held
            assert manager.lock_path.exists()

        # Lock should be released
        # Give it a moment for cleanup
        time.sleep(0.1)
        assert not manager.lock_path.exists()

    def test_lock_timeout(self, sample_config):
        """Test lock timeout when lock is held by another process."""
        _config, config_path = sample_config
        manager1 = StateManager(config_path=config_path)
        manager2 = StateManager(config_path=config_path)

        with manager1.lock(), pytest.raises(
            TimeoutError, match="Could not acquire lock"
        ), manager2.lock(timeout=1):
            pass

    def test_concurrent_read_safety(self, sample_config):
        """Test that concurrent reads are safe with locking."""
        _config, config_path = sample_config
        manager = StateManager(config_path=config_path)

        # Write initial state
        manager.write_state(
            "test-server", {"status": "running"}, update_timestamp=False
        )

        results = []
        errors = []

        def read_state():
            try:
                state = manager.read_state("test-server")
                results.append(state)
            except Exception as e:
                errors.append(e)

        # Create multiple threads reading concurrently
        threads = [threading.Thread(target=read_state) for _ in range(5)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # All reads should succeed
        assert len(errors) == 0
        assert len(results) == 5
        assert all(r["status"] == "running" for r in results)

    @pytest.mark.xdist_group(name="state_locking")
    def test_concurrent_write_safety(self, sample_config):
        """Test that concurrent writes are safe with locking."""
        _config, config_path = sample_config
        manager = StateManager(config_path=config_path)

        errors = []

        # Use valid status values from the enum
        valid_statuses = ["running", "stopped", "error", "pending", "deploying"]

        def update_state(value):
            try:
                manager.write_state(
                    "test-server", {"status": value}, update_timestamp=False
                )
            except Exception as e:
                errors.append(e)

        # Create multiple threads writing concurrently
        threads = [
            threading.Thread(target=update_state, args=(valid_statuses[i],))
            for i in range(5)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # All writes should succeed (no errors)
        assert len(errors) == 0

        # Final state should be one of the written values
        state = manager.read_state("test-server")
        assert state["status"] in valid_statuses


class TestStateSnapshot:
    """Tests for state snapshot and restore."""

    def test_create_snapshot(self, sample_config):
        """Test creating a state snapshot."""
        _config, config_path = sample_config
        manager = StateManager(config_path=config_path)

        # Update some states
        manager.write_state(
            "test-server", {"status": "running"}, update_timestamp=False
        )
        manager.write_state(
            "test-server-2", {"status": "stopped"}, update_timestamp=False
        )

        # Create snapshot

        # Mock the load function
        import nodetool.deploy.state as state_module

        original_load = state_module.load_deployment_config

        def mock_load():
            import yaml

            from nodetool.config.deployment import DeploymentConfig

            with open(config_path) as f:
                data = yaml.safe_load(f)
            return DeploymentConfig.model_validate(data)

        state_module.load_deployment_config = mock_load

        try:
            current_config = mock_load()
            snapshot = create_state_snapshot(current_config, config_path=config_path)

            assert "timestamp" in snapshot
            assert "version" in snapshot
            assert "deployments" in snapshot
            assert len(snapshot["deployments"]) == 2
            assert (
                snapshot["deployments"]["test-server"]["state"]["status"] == "running"
            )
            assert (
                snapshot["deployments"]["test-server-2"]["state"]["status"] == "stopped"
            )
        finally:
            state_module.load_deployment_config = original_load

    def test_restore_snapshot(self, sample_config):
        """Test restoring state from a snapshot."""
        _config, config_path = sample_config
        manager = StateManager(config_path=config_path)

        # Set initial state
        manager.write_state(
            "test-server", {"status": "running"}, update_timestamp=False
        )

        # Create snapshot

        import nodetool.deploy.state as state_module

        original_load = state_module.load_deployment_config
        original_save = state_module.save_deployment_config

        def mock_load():
            import yaml

            from nodetool.config.deployment import DeploymentConfig

            with open(config_path) as f:
                data = yaml.safe_load(f)
            return DeploymentConfig.model_validate(data)

        def mock_save(cfg):
            import yaml

            data = cfg.model_dump(mode="json", exclude_none=True)
            with open(config_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)

        state_module.load_deployment_config = mock_load
        state_module.save_deployment_config = mock_save

        try:
            current_config = mock_load()
            snapshot = create_state_snapshot(current_config, config_path=config_path)

            # Change state
            manager.write_state(
                "test-server", {"status": "error"}, update_timestamp=False
            )

            # Verify changed
            state = manager.read_state("test-server")
            assert state["status"] == "error"

            # Restore from snapshot
            restore_state_from_snapshot(snapshot, config_path=config_path)

            # Should be back to running
            state = manager.read_state("test-server")
            assert state["status"] == "running"

        finally:
            state_module.load_deployment_config = original_load
            state_module.save_deployment_config = original_save

    def test_restore_specific_deployment(self, sample_config):
        """Test restoring a specific deployment from snapshot."""
        _config, config_path = sample_config
        manager = StateManager(config_path=config_path)

        # Set states
        manager.write_state(
            "test-server", {"status": "running"}, update_timestamp=False
        )
        manager.write_state(
            "test-server-2", {"status": "stopped"}, update_timestamp=False
        )

        import nodetool.deploy.state as state_module

        original_load = state_module.load_deployment_config
        original_save = state_module.save_deployment_config

        def mock_load():
            import yaml

            from nodetool.config.deployment import DeploymentConfig

            with open(config_path) as f:
                data = yaml.safe_load(f)
            return DeploymentConfig.model_validate(data)

        def mock_save(cfg):
            import yaml

            data = cfg.model_dump(mode="json", exclude_none=True)
            with open(config_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)

        state_module.load_deployment_config = mock_load
        state_module.save_deployment_config = mock_save

        try:
            current_config = mock_load()
            snapshot = create_state_snapshot(current_config, config_path=config_path)

            # Change both states
            manager.write_state(
                "test-server", {"status": "error"}, update_timestamp=False
            )
            manager.write_state(
                "test-server-2", {"status": "error"}, update_timestamp=False
            )

            # Restore only test-server
            restore_state_from_snapshot(
                snapshot, deployment_name="test-server", config_path=config_path
            )

            # test-server should be restored, test-server-2 should still be error
            state1 = manager.read_state("test-server")
            state2 = manager.read_state("test-server-2")

            assert state1["status"] == "running"
            assert state2["status"] == "error"

        finally:
            state_module.load_deployment_config = original_load
            state_module.save_deployment_config = original_save
