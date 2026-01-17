"""
Deployment state management for NodeTool.

This module provides utilities for managing deployment state with atomic operations,
locking, and timestamp tracking to ensure safe concurrent access.
"""

import secrets
import threading
import time
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Generator, Optional

# Cross-platform file locking:
# - Unix: fcntl.flock
# - Windows: msvcrt.locking
try:  # pragma: no cover - platform-specific
    import fcntl  # type: ignore

    _HAS_FCNTL = True
except ModuleNotFoundError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore
    _HAS_FCNTL = False
    import msvcrt  # type: ignore

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
    - Thread-based locking for in-process thread safety
    - Automatic timestamp tracking
    - State validation
    """

    _thread_lock = threading.Lock()

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

        This prevents concurrent modifications by other processes and threads.

        Uses both a threading lock (for in-process thread safety) and
        file locking (for cross-process safety).

        Args:
            timeout: Maximum time to wait for lock acquisition (seconds)

        Raises:
            TimeoutError: If lock cannot be acquired within timeout

        Yields:
            None: Lock is held while in context
        """
        # First acquire the thread lock to prevent concurrent access from multiple threads
        acquired_thread_lock = self._thread_lock.acquire(timeout=timeout)
        if not acquired_thread_lock:
            raise TimeoutError(f"Could not acquire thread lock within {timeout} seconds")

        try:
            # Then acquire the file lock for cross-process safety
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            start_time = time.time()
            acquired = False

            with open(self.lock_path, "w") as lock_file:
                try:
                    # Try to acquire lock with timeout
                    while time.time() - start_time < timeout:
                        try:
                            if _HAS_FCNTL:
                                # Unix-style advisory lock
                                fcntl.flock(  # type: ignore[union-attr]
                                    lock_file.fileno(),
                                    fcntl.LOCK_EX | fcntl.LOCK_NB,  # type: ignore[union-attr]
                                )
                            else:
                                # Windows: lock 1 byte in the lock file (non-blocking)
                                lock_file.seek(0, 2)
                                if lock_file.tell() == 0:
                                    lock_file.write("0")
                                    lock_file.flush()
                                lock_file.seek(0)
                                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[name-defined]
                            acquired = True
                            break
                        except OSError:
                            time.sleep(0.1)

                    if not acquired:
                        raise TimeoutError(f"Could not acquire lock on {self.lock_path} within {timeout} seconds")

                    yield

                finally:
                    if acquired:
                        with suppress(OSError):
                            if _HAS_FCNTL:
                                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)  # type: ignore[union-attr]
                            else:
                                lock_file.seek(0)
                                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[name-defined]
                lock_file.close()

                # Clean up lock file
                with suppress(FileNotFoundError):
                    self.lock_path.unlink()
        finally:
            self._thread_lock.release()

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

            with open(self.config_path) as f:
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

            with open(self.config_path) as f:
                data = yaml.safe_load(f)
            config = DeploymentConfig.model_validate(data)

            deployment = config.deployments.get(deployment_name)
            if not deployment:
                raise KeyError(f"Deployment '{deployment_name}' not found")

            # Update timestamp if requested
            if update_timestamp:
                state_updates["last_deployed"] = datetime.now(UTC)

            # Update state fields
            current_state = deployment.state.model_dump()
            current_state.update(state_updates)

            # Re-validate and update
            deployment.state = deployment.state.__class__.model_validate(current_state)  # type: ignore[assignment]

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

    def update_deployment_status(self, deployment_name: str, status: str, update_timestamp: bool = True) -> None:
        """
        Update the status of a deployment.

        Args:
            deployment_name: Name of the deployment
            status: New status value
            update_timestamp: Whether to update last_deployed timestamp (default: True)
        """
        self.write_state(deployment_name, {"status": status}, update_timestamp=update_timestamp)

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

            with open(self.config_path) as f:
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

            with open(self.config_path) as f:
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

            deployment.state = deployment.state.__class__.model_validate(state_dict)  # type: ignore[assignment]

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

            with open(self.config_path) as f:
                data = yaml.safe_load(f)
            config = DeploymentConfig.model_validate(data)

            deployment = config.deployments.get(deployment_name)
            if not deployment:
                raise KeyError(f"Deployment '{deployment_name}' not found")

            # Reset to default state for this deployment type
            deployment.state = deployment.state.__class__()  # type: ignore[assignment]

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


def create_state_snapshot(config: DeploymentConfig, config_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """
    Create a snapshot of the current state of all deployments.

    This can be used for backup, auditing, or rollback purposes.

    Args:
        config: Deployment configuration

    Returns:
        Dictionary containing snapshot of all deployment states
    """
    snapshot: Dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "version": config.version,
        "deployments": {},
    }

    if config_path:
        snapshot["config_path"] = str(config_path)

    for name, deployment in config.deployments.items():
        snapshot["deployments"][name] = {
            "type": deployment.type,
            "enabled": deployment.enabled,
            "state": deployment.state.model_dump(mode="json"),
        }

    return snapshot


def restore_state_from_snapshot(
    snapshot: Dict[str, Any],
    deployment_name: Optional[str] = None,
    config_path: Optional[Path | str] = None,
) -> None:
    """
    Restore deployment state from a snapshot.

    Args:
        snapshot: Snapshot dictionary created by create_state_snapshot()
        deployment_name: If provided, only restore this deployment (otherwise restore all)

    Raises:
        KeyError: If deployment_name specified but not found in snapshot
    """
    resolved_config_path = Path(config_path) if config_path else None
    if not resolved_config_path and snapshot.get("config_path"):
        resolved_config_path = Path(snapshot["config_path"])

    state_manager = StateManager(config_path=resolved_config_path)

    with state_manager.lock():
        if resolved_config_path:
            import yaml

            with open(resolved_config_path) as f:
                data = yaml.safe_load(f)
            config = DeploymentConfig.model_validate(data) if data else DeploymentConfig()
        else:
            config = load_deployment_config()

        deployments_to_restore = [deployment_name] if deployment_name else snapshot["deployments"].keys()

        for name in deployments_to_restore:
            if name not in snapshot["deployments"]:
                raise KeyError(f"Deployment '{name}' not found in snapshot")

            if name not in config.deployments:
                # Skip deployments that no longer exist in config
                continue

            snapshot_state = snapshot["deployments"][name]["state"]
            deployment = config.deployments[name]

            # Restore state
            deployment.state = deployment.state.__class__.model_validate(snapshot_state)  # type: ignore[assignment]

        if resolved_config_path:
            import yaml

            data = config.model_dump(mode="json", exclude_none=True)
            with open(resolved_config_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            save_deployment_config(config)
