"""
Deployment state management for NodeTool.

This module provides utilities for managing deployment state with atomic operations,
locking, and timestamp tracking to ensure safe concurrent access.
"""

import fcntl
import secrets
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Generator

from nodetool.config.deployment import (
    DeploymentConfig,
    load_deployment_config,
    save_deployment_config,
)


class StateManager:
    """
    Manages deployment state with atomic operations and file locking.

    This class provides methods for safely reading and writing deployment state,
    with support for:
    - Atomic file operations
    - File-based locking to prevent concurrent modifications
    - Automatic timestamp tracking
    - State validation
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the StateManager.

        Args:
            config_path: Path to deployment config file (optional, uses default if not provided)
        """
        from nodetool.config.deployment import get_deployment_config_path

        self.config_path = config_path or get_deployment_config_path()
        self.lock_path = self.config_path.with_suffix(".lock")

    @contextmanager
    def lock(self, timeout: int = 30) -> Generator[None, None, None]:
        """
        Acquire an exclusive lock on the deployment configuration file.

        This prevents concurrent modifications by other processes.

        Args:
            timeout: Maximum time to wait for lock acquisition (seconds)

        Raises:
            TimeoutError: If lock cannot be acquired within timeout

        Yields:
            None: Lock is held while in context
        """
        # Ensure lock file exists
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = open(self.lock_path, "w")

        start_time = time.time()
        acquired = False

        try:
            # Try to acquire lock with timeout
            while time.time() - start_time < timeout:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except (IOError, OSError):
                    time.sleep(0.1)

            if not acquired:
                raise TimeoutError(
                    f"Could not acquire lock on {self.lock_path} within {timeout} seconds"
                )

            yield

        finally:
            if acquired:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except (IOError, OSError):
                    pass  # Best effort unlock
            lock_file.close()

            # Clean up lock file
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass

    def read_state(self, deployment_name: str) -> Optional[Dict[str, Any]]:
        """
        Read the state for a specific deployment.

        Args:
            deployment_name: Name of the deployment

        Returns:
            Dictionary containing deployment state, or None if deployment not found

        Raises:
            FileNotFoundError: If deployment config doesn't exist
        """
        with self.lock():
            # Load config from our specific path
            import yaml
            from nodetool.config.deployment import DeploymentConfig

            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f)
            config = DeploymentConfig.model_validate(data)

            deployment = config.deployments.get(deployment_name)
            if not deployment:
                return None

            # Return state as dict
            return deployment.state.model_dump(mode="json")

    def write_state(
        self,
        deployment_name: str,
        state_updates: Dict[str, Any],
        update_timestamp: bool = True,
    ) -> None:
        """
        Update the state for a specific deployment.

        Args:
            deployment_name: Name of the deployment
            state_updates: Dictionary of state fields to update
            update_timestamp: Whether to update last_deployed timestamp (default: True)

        Raises:
            FileNotFoundError: If deployment config doesn't exist
            KeyError: If deployment not found in config
        """
        with self.lock():
            # Load config from our specific path
            import yaml
            from nodetool.config.deployment import DeploymentConfig

            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f)
            config = DeploymentConfig.model_validate(data)

            deployment = config.deployments.get(deployment_name)
            if not deployment:
                raise KeyError(f"Deployment '{deployment_name}' not found")

            # Update timestamp if requested
            if update_timestamp:
                state_updates["last_deployed"] = datetime.utcnow()

            # Update state fields
            current_state = deployment.state.model_dump()
            current_state.update(state_updates)

            # Re-validate and update
            deployment.state = deployment.state.__class__.model_validate(current_state)

            # Save config to our specific path
            data = config.model_dump(mode="json", exclude_none=True)
            temp_path = self.config_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                yaml.dump(
                    data,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
            temp_path.replace(self.config_path)

    def update_deployment_status(
        self, deployment_name: str, status: str, update_timestamp: bool = True
    ) -> None:
        """
        Update the status of a deployment.

        Args:
            deployment_name: Name of the deployment
            status: New status value
            update_timestamp: Whether to update last_deployed timestamp (default: True)
        """
        self.write_state(
            deployment_name, {"status": status}, update_timestamp=update_timestamp
        )

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get state for all deployments.

        Returns:
            Dictionary mapping deployment names to their states
        """
        with self.lock():
            # Load config from our specific path
            import yaml
            from nodetool.config.deployment import DeploymentConfig

            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f)
            config = DeploymentConfig.model_validate(data)

            states = {}
            for name, deployment in config.deployments.items():
                states[name] = deployment.state.model_dump(mode="json")

            return states

    def get_or_create_secret(
        self,
        deployment_name: str,
        field_name: str,
        byte_length: int = 32,
    ) -> str:
        """
        Retrieve a secret from deployment state, generating and persisting it if missing.
        """
        with self.lock():
            import yaml
            from nodetool.config.deployment import DeploymentConfig

            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f)
            config = DeploymentConfig.model_validate(data)

            deployment = config.deployments.get(deployment_name)
            if deployment is None:
                raise KeyError(f"Deployment '{deployment_name}' not found")

            state_dict = deployment.state.model_dump()
            existing = state_dict.get(field_name)
            if existing:
                return existing

            secret_value = secrets.token_urlsafe(byte_length)
            state_dict[field_name] = secret_value

            deployment.state = deployment.state.__class__.model_validate(state_dict)

            data = config.model_dump(mode="json", exclude_none=True)
            temp_path = self.config_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                yaml.dump(
                    data,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
            temp_path.replace(self.config_path)

            return secret_value

    def clear_state(self, deployment_name: str) -> None:
        """
        Clear/reset the state for a deployment.

        This sets the state back to defaults while keeping the deployment configuration.

        Args:
            deployment_name: Name of the deployment

        Raises:
            KeyError: If deployment not found in config
        """
        with self.lock():
            # Load config from our specific path
            import yaml
            from nodetool.config.deployment import DeploymentConfig

            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f)
            config = DeploymentConfig.model_validate(data)

            deployment = config.deployments.get(deployment_name)
            if not deployment:
                raise KeyError(f"Deployment '{deployment_name}' not found")

            # Reset to default state for this deployment type
            deployment.state = deployment.state.__class__()

            # Save config to our specific path
            data = config.model_dump(mode="json", exclude_none=True)
            temp_path = self.config_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                yaml.dump(
                    data,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
            temp_path.replace(self.config_path)

    def get_last_deployed(self, deployment_name: str) -> Optional[datetime]:
        """
        Get the last deployment timestamp for a deployment.

        Args:
            deployment_name: Name of the deployment

        Returns:
            Last deployed datetime, or None if never deployed or not found
        """
        state = self.read_state(deployment_name)
        if state and state.get("last_deployed"):
            # Handle both string and datetime formats
            last_deployed = state["last_deployed"]
            if isinstance(last_deployed, str):
                return datetime.fromisoformat(last_deployed.replace("Z", "+00:00"))
            return last_deployed
        return None

    def has_been_deployed(self, deployment_name: str) -> bool:
        """
        Check if a deployment has ever been deployed.

        Args:
            deployment_name: Name of the deployment

        Returns:
            True if deployment has been deployed at least once
        """
        return self.get_last_deployed(deployment_name) is not None


def create_state_snapshot(config: DeploymentConfig) -> Dict[str, Any]:
    """
    Create a snapshot of the current state of all deployments.

    This can be used for backup, auditing, or rollback purposes.

    Args:
        config: Deployment configuration

    Returns:
        Dictionary containing snapshot of all deployment states
    """
    snapshot = {
        "timestamp": datetime.utcnow().isoformat(),
        "version": config.version,
        "deployments": {},
    }

    for name, deployment in config.deployments.items():
        snapshot["deployments"][name] = {
            "type": deployment.type,
            "enabled": deployment.enabled,
            "state": deployment.state.model_dump(mode="json"),
        }

    return snapshot


def restore_state_from_snapshot(
    snapshot: Dict[str, Any], deployment_name: Optional[str] = None
) -> None:
    """
    Restore deployment state from a snapshot.

    Args:
        snapshot: Snapshot dictionary created by create_state_snapshot()
        deployment_name: If provided, only restore this deployment (otherwise restore all)

    Raises:
        KeyError: If deployment_name specified but not found in snapshot
    """
    state_manager = StateManager()

    with state_manager.lock():
        config = load_deployment_config()

        deployments_to_restore = (
            [deployment_name] if deployment_name else snapshot["deployments"].keys()
        )

        for name in deployments_to_restore:
            if name not in snapshot["deployments"]:
                raise KeyError(f"Deployment '{name}' not found in snapshot")

            if name not in config.deployments:
                # Skip deployments that no longer exist in config
                continue

            snapshot_state = snapshot["deployments"][name]["state"]
            deployment = config.deployments[name]

            # Restore state
            deployment.state = deployment.state.__class__.model_validate(snapshot_state)

        save_deployment_config(config)
