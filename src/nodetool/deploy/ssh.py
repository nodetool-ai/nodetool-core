"""
SSH connection management for remote deployment operations.

This module provides a high-level interface for SSH operations including:
- Connection management with automatic retry
- Remote command execution
- File transfer (SFTP)
- Connection pooling for efficiency
"""

import os
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    import paramiko
    from paramiko import AutoAddPolicy, SSHClient
    from paramiko.sftp_client import SFTPClient

try:
    import paramiko
    from paramiko import AutoAddPolicy, SSHClient
    from paramiko.sftp_client import SFTPClient

    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False
    # These will only be accessed at runtime when PARAMIKO_AVAILABLE is True
    paramiko = None  # type: ignore[assignment]
    SSHClient = None  # type: ignore[assignment]
    AutoAddPolicy = None  # type: ignore[assignment]
    SFTPClient = None  # type: ignore[assignment]


class SSHConnectionError(Exception):
    """Raised when SSH connection fails."""

    pass


class SSHCommandError(Exception):
    """Raised when remote command execution fails."""

    def __init__(self, message: str, exit_code: int, stdout: str, stderr: str):
        super().__init__(message)
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class SSHConnection:
    """
    Manages SSH connections to remote hosts with automatic retry and connection pooling.

    This class provides a high-level interface for SSH operations including
    command execution and file transfer.
    """

    def __init__(
        self,
        host: str,
        user: str,
        key_path: Optional[str] = None,
        password: Optional[str] = None,
        port: int = 22,
        timeout: int = 30,
        retry_attempts: int = 3,
        retry_delay: float = 2.0,
    ):
        """
        Initialize SSH connection manager.

        Args:
            host: Remote host address (IP or hostname)
            user: SSH username
            key_path: Path to SSH private key file (optional)
            password: SSH password (optional, not recommended)
            port: SSH port (default: 22)
            timeout: Connection timeout in seconds (default: 30)
            retry_attempts: Number of connection retry attempts (default: 3)
            retry_delay: Delay between retries in seconds (default: 2.0)

        Raises:
            ImportError: If paramiko is not installed
        """
        if not PARAMIKO_AVAILABLE:
            raise ImportError("paramiko is required for SSH operations. Install it with: pip install paramiko")

        self.host = host
        self.user = user
        self.key_path = key_path
        self.password = password
        self.port = port
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        self._client: Optional[SSHClient] = None
        self._sftp: Optional[SFTPClient] = None

    def connect(self) -> None:
        """
        Establish SSH connection to the remote host.

        This method will retry connection attempts based on retry_attempts setting.

        Raises:
            SSHConnectionError: If connection fails after all retry attempts
        """
        if not PARAMIKO_AVAILABLE:
            raise SSHConnectionError("Paramiko library is not available")

        for attempt in range(self.retry_attempts):
            try:
                self._client = SSHClient()  # type: ignore[misc]
                self._client.set_missing_host_key_policy(AutoAddPolicy())  # type: ignore[misc]

                connect_kwargs: dict[str, Any] = {
                    "hostname": self.host,
                    "port": self.port,
                    "username": self.user,
                    "timeout": self.timeout,
                    "banner_timeout": self.timeout,
                }

                # Use key-based authentication if key path provided
                if self.key_path:
                    key_path = Path(self.key_path).expanduser()
                    if not key_path.exists():
                        raise SSHConnectionError(f"SSH key not found at {key_path}")
                    connect_kwargs["key_filename"] = str(key_path)
                elif self.password:
                    connect_kwargs["password"] = self.password
                else:
                    # Try to use SSH agent
                    connect_kwargs["look_for_keys"] = True

                self._client.connect(**connect_kwargs)
                return  # Connection successful

            except Exception as e:
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    # Final attempt failed
                    raise SSHConnectionError(
                        f"Failed to connect to {self.user}@{self.host}:{self.port} "
                        f"after {self.retry_attempts} attempts: {e}"
                    ) from e

    def disconnect(self) -> None:
        """Close SSH and SFTP connections."""
        if self._sftp:
            with suppress(Exception):
                self._sftp.close()
            self._sftp = None

        if self._client:
            with suppress(Exception):
                self._client.close()
            self._client = None

    def is_connected(self) -> bool:
        """Check if SSH connection is active."""
        if not self._client:
            return False
        transport = self._client.get_transport()
        return transport is not None and transport.is_active()

    def ensure_connected(self) -> None:
        """Ensure connection is active, reconnect if necessary."""
        if not self.is_connected():
            self.connect()

    def execute(
        self,
        command: str,
        check: bool = True,
        timeout: Optional[int] = None,
    ) -> tuple[int, str, str]:
        """
        Execute a command on the remote host.

        Args:
            command: Command to execute
            check: If True, raise exception on non-zero exit code (default: True)
            timeout: Command timeout in seconds (optional)

        Returns:
            Tuple of (exit_code, stdout, stderr)

        Raises:
            SSHCommandError: If check=True and command returns non-zero exit code
            SSHConnectionError: If not connected
        """
        self.ensure_connected()

        if not self._client:
            raise SSHConnectionError("Not connected to remote host")

        _stdin, stdout, stderr = self._client.exec_command(command, timeout=timeout or self.timeout)

        # Wait for command to complete and read output
        exit_code = stdout.channel.recv_exit_status()
        stdout_data = stdout.read().decode("utf-8", errors="replace")
        stderr_data = stderr.read().decode("utf-8", errors="replace")

        if check and exit_code != 0:
            error_msg = f"Command failed with exit code {exit_code}: {command}\nSTDERR:\n{stderr_data}"
            raise SSHCommandError(
                error_msg,
                exit_code=exit_code,
                stdout=stdout_data,
                stderr=stderr_data,
            )

        return exit_code, stdout_data, stderr_data

    def execute_script(
        self,
        script: str,
        check: bool = True,
        timeout: Optional[int] = None,
    ) -> tuple[int, str, str]:
        """
        Execute a multi-line shell script on the remote host.

        Args:
            script: Shell script to execute (can contain multiple lines)
            check: If True, raise exception on non-zero exit code
            timeout: Script timeout in seconds (optional)

        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        # Wrap script in bash to handle multi-line scripts properly
        command = f"bash -c {repr(script)}"
        return self.execute(command, check=check, timeout=timeout)

    def _get_sftp(self) -> SFTPClient:
        """Get or create SFTP client."""
        self.ensure_connected()

        if not self._sftp:
            if not self._client:
                raise SSHConnectionError("Not connected to remote host")
            self._sftp = self._client.open_sftp()

        return self._sftp

    def upload_file(self, local_path: str, remote_path: str, mode: Optional[int] = None) -> None:
        """
        Upload a file to the remote host.

        Args:
            local_path: Path to local file
            remote_path: Destination path on remote host
            mode: Optional file mode (e.g., 0o755)

        Raises:
            FileNotFoundError: If local file doesn't exist
            SSHConnectionError: If not connected
        """
        local_file = Path(local_path)
        if not local_file.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        sftp = self._get_sftp()
        sftp.put(str(local_file), remote_path)

        if mode is not None:
            sftp.chmod(remote_path, mode)

    def upload_string(self, content: str, remote_path: str, mode: Optional[int] = None) -> None:
        """
        Upload string content as a file to the remote host.

        Args:
            content: String content to upload
            remote_path: Destination path on remote host
            mode: Optional file mode (e.g., 0o755)
        """
        sftp = self._get_sftp()

        # Write content to remote file
        with sftp.open(remote_path, "w") as remote_file:
            remote_file.write(content)

        if mode is not None:
            sftp.chmod(remote_path, mode)

    def download_file(self, remote_path: str, local_path: str) -> None:
        """
        Download a file from the remote host.

        Args:
            remote_path: Path to remote file
            local_path: Destination path on local host
        """
        sftp = self._get_sftp()

        # Ensure local directory exists
        local_file = Path(local_path)
        local_file.parent.mkdir(parents=True, exist_ok=True)

        sftp.get(remote_path, str(local_file))

    def file_exists(self, remote_path: str) -> bool:
        """
        Check if a file exists on the remote host.

        Args:
            remote_path: Path to check

        Returns:
            True if file exists, False otherwise
        """
        try:
            sftp = self._get_sftp()
            sftp.stat(remote_path)
            return True
        except FileNotFoundError:
            return False

    def mkdir(self, remote_path: str, mode: int = 0o755, parents: bool = True) -> None:
        """
        Create a directory on the remote host.

        Args:
            remote_path: Path to create
            mode: Directory mode (default: 0o755)
            parents: Create parent directories if needed (default: True)
        """
        if parents:
            # Create parent directories recursively
            parts = Path(remote_path).parts
            current = ""
            for part in parts:
                current = os.path.join(current, part)
                if not current or current == "/":
                    continue
                try:
                    sftp = self._get_sftp()
                    sftp.mkdir(current, mode=mode)
                except OSError:
                    # Directory might already exist
                    pass
        else:
            sftp = self._get_sftp()
            sftp.mkdir(remote_path, mode=mode)

    def rmdir(self, remote_path: str, recursive: bool = False) -> None:
        """
        Remove a directory on the remote host.

        Args:
            remote_path: Path to remove
            recursive: Remove directory recursively (default: False)
        """
        if recursive:
            # Use rm -rf for recursive removal
            self.execute(f"rm -rf {repr(remote_path)}")
        else:
            sftp = self._get_sftp()
            sftp.rmdir(remote_path)

    def __enter__(self):
        """Context manager entry - connect to remote host."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - disconnect from remote host."""
        self.disconnect()

    def __del__(self):
        """Ensure connection is closed on deletion."""
        self.disconnect()


@contextmanager
def ssh_connection(
    host: str,
    user: str,
    key_path: Optional[str] = None,
    password: Optional[str] = None,
    port: int = 22,
    **kwargs,
):
    """
    Context manager for SSH connections.

    Example:
        with ssh_connection("192.168.1.100", "ubuntu", key_path="~/.ssh/id_rsa") as ssh:
            exit_code, stdout, stderr = ssh.execute("ls -la")
            print(stdout)

    Args:
        host: Remote host address
        user: SSH username
        key_path: Path to SSH private key (optional)
        password: SSH password (optional)
        port: SSH port (default: 22)
        **kwargs: Additional arguments passed to SSHConnection

    Yields:
        SSHConnection instance
    """
    conn = SSHConnection(
        host=host,
        user=user,
        key_path=key_path,
        password=password,
        port=port,
        **kwargs,
    )
    try:
        conn.connect()
        yield conn
    finally:
        conn.disconnect()
