"""
Unit tests for SSH connection utilities.
"""

from unittest.mock import MagicMock, patch

import pytest

from nodetool.deploy.ssh import SSHCommandError, SSHConnection, SSHConnectionError

# Mark all tests to not use any fixtures from conftest
pytest_plugins = ()


class TestSSHConnection:
    """Tests for SSHConnection class."""

    @pytest.fixture
    def mock_paramiko(self):
        """Mock paramiko module."""
        with patch("nodetool.deploy.ssh.paramiko") as mock, patch(
            "nodetool.deploy.ssh.PARAMIKO_AVAILABLE", True
        ), patch("nodetool.deploy.ssh.SSHClient") as mock_client_cls, patch(
            "nodetool.deploy.ssh.AutoAddPolicy"
        ) as mock_policy:
            yield {
                "paramiko": mock,
                "SSHClient": mock_client_cls,
                "AutoAddPolicy": mock_policy,
            }

    def test_init_with_key_path(self):
        """Test initialization with SSH key path."""
        with patch("nodetool.deploy.ssh.PARAMIKO_AVAILABLE", True):
            conn = SSHConnection(
                host="example.com",
                user="user",
                key_path="~/.ssh/id_rsa",
            )

            assert conn.host == "example.com"
            assert conn.user == "user"
            assert conn.key_path == "~/.ssh/id_rsa"
            assert conn.password is None
            assert conn.port == 22

    def test_init_with_password(self):
        """Test initialization with password."""
        with patch("nodetool.deploy.ssh.PARAMIKO_AVAILABLE", True):
            conn = SSHConnection(
                host="example.com",
                user="user",
                password="secret",
            )

            assert conn.host == "example.com"
            assert conn.user == "user"
            assert conn.password == "secret"
            assert conn.key_path is None

    def test_init_with_custom_port(self):
        """Test initialization with custom port."""
        with patch("nodetool.deploy.ssh.PARAMIKO_AVAILABLE", True):
            conn = SSHConnection(
                host="example.com",
                user="user",
                key_path="~/.ssh/id_rsa",
                port=2222,
            )

            assert conn.port == 2222

    def test_init_without_paramiko(self):
        """Test initialization without paramiko raises error."""
        with patch("nodetool.deploy.ssh.PARAMIKO_AVAILABLE", False), pytest.raises(
            ImportError, match="paramiko is required"
        ):
            SSHConnection(
                host="example.com",
                user="user",
                key_path="~/.ssh/id_rsa",
            )

    def test_connect_with_key(self, mock_paramiko, tmp_path):
        """Test SSH connection using key authentication."""
        # Create a fake key file
        key_file = tmp_path / ".ssh" / "id_rsa"
        key_file.parent.mkdir(parents=True)
        key_file.write_text("fake key")

        mock_client = MagicMock()
        mock_paramiko["SSHClient"].return_value = mock_client

        conn = SSHConnection(
            host="example.com",
            user="user",
            key_path=str(key_file),
        )

        conn.connect()

        # Verify SSH client setup
        mock_client.set_missing_host_key_policy.assert_called_once()
        assert mock_client.connect.called
        call_kwargs = mock_client.connect.call_args[1]
        assert call_kwargs["hostname"] == "example.com"
        assert call_kwargs["username"] == "user"
        assert str(key_file) in str(call_kwargs["key_filename"])

    def test_connect_with_password(self, mock_paramiko):
        """Test SSH connection using password authentication."""
        mock_client = MagicMock()
        mock_paramiko["SSHClient"].return_value = mock_client

        conn = SSHConnection(
            host="example.com",
            user="user",
            password="secret",
        )

        conn.connect()

        assert mock_client.connect.called
        call_kwargs = mock_client.connect.call_args[1]
        assert call_kwargs["password"] == "secret"

    def test_connect_failure(self, mock_paramiko):
        """Test SSH connection failure."""
        mock_client = MagicMock()
        mock_client.connect.side_effect = Exception("Connection failed")
        mock_paramiko["SSHClient"].return_value = mock_client

        conn = SSHConnection(
            host="example.com",
            user="user",
            key_path="/nonexistent/key",
            retry_attempts=1,
        )

        with pytest.raises(SSHConnectionError, match="Failed to connect"):
            conn.connect()

    def test_disconnect(self, mock_paramiko):
        """Test SSH disconnection."""
        mock_client = MagicMock()
        mock_paramiko["SSHClient"].return_value = mock_client

        conn = SSHConnection(
            host="example.com",
            user="user",
            key_path="/fake/key",
            retry_attempts=1,
        )

        # Mock successful connection
        conn._client = mock_client
        conn.disconnect()

        mock_client.close.assert_called_once()
        assert conn._client is None

    def test_disconnect_without_connection(self):
        """Test disconnect when not connected."""
        with patch("nodetool.deploy.ssh.PARAMIKO_AVAILABLE", True):
            conn = SSHConnection(
                host="example.com",
                user="user",
                key_path="~/.ssh/id_rsa",
            )

            # Should not raise an error
            conn.disconnect()

    def test_context_manager(self, mock_paramiko, tmp_path):
        """Test using SSHConnection as context manager."""
        # Create a fake key file
        key_file = tmp_path / ".ssh" / "id_rsa"
        key_file.parent.mkdir(parents=True)
        key_file.write_text("fake key")

        mock_client = MagicMock()
        mock_paramiko["SSHClient"].return_value = mock_client

        conn = SSHConnection(
            host="example.com",
            user="user",
            key_path=str(key_file),
            retry_attempts=1,
        )

        with conn:
            # Should be connected
            assert mock_client.set_missing_host_key_policy.called

        # Should be disconnected
        mock_client.close.assert_called()

    def test_execute_success(self, mock_paramiko):
        """Test executing a command successfully."""
        mock_client = MagicMock()
        mock_stdin = MagicMock()
        mock_stdout = MagicMock()
        mock_stderr = MagicMock()

        mock_stdout.read.return_value = b"output"
        mock_stderr.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0

        mock_client.exec_command.return_value = (mock_stdin, mock_stdout, mock_stderr)
        mock_client.get_transport.return_value.is_active.return_value = True
        mock_paramiko["SSHClient"].return_value = mock_client

        conn = SSHConnection(
            host="example.com",
            user="user",
            key_path="/fake/key",
            retry_attempts=1,
        )

        conn._client = mock_client
        exit_code, stdout, stderr = conn.execute("ls -la")

        assert exit_code == 0
        assert stdout == "output"
        assert stderr == ""

        mock_client.exec_command.assert_called_once_with("ls -la", timeout=30)

    def test_execute_failure_with_check(self, mock_paramiko):
        """Test executing a command that fails with check=True."""
        mock_client = MagicMock()
        mock_stdin = MagicMock()
        mock_stdout = MagicMock()
        mock_stderr = MagicMock()

        mock_stdout.read.return_value = b""
        mock_stderr.read.return_value = b"error message"
        mock_stdout.channel.recv_exit_status.return_value = 1

        mock_client.exec_command.return_value = (mock_stdin, mock_stdout, mock_stderr)
        mock_client.get_transport.return_value.is_active.return_value = True
        mock_paramiko["SSHClient"].return_value = mock_client

        conn = SSHConnection(
            host="example.com",
            user="user",
            key_path="/fake/key",
            retry_attempts=1,
        )

        conn._client = mock_client

        with pytest.raises(SSHCommandError) as exc_info:
            conn.execute("false", check=True)

        assert exc_info.value.exit_code == 1
        assert exc_info.value.stderr == "error message"

    def test_execute_failure_without_check(self, mock_paramiko):
        """Test executing a command that fails with check=False."""
        mock_client = MagicMock()
        mock_stdin = MagicMock()
        mock_stdout = MagicMock()
        mock_stderr = MagicMock()

        mock_stdout.read.return_value = b""
        mock_stderr.read.return_value = b"error message"
        mock_stdout.channel.recv_exit_status.return_value = 1

        mock_client.exec_command.return_value = (mock_stdin, mock_stdout, mock_stderr)
        mock_client.get_transport.return_value.is_active.return_value = True
        mock_paramiko["SSHClient"].return_value = mock_client

        conn = SSHConnection(
            host="example.com",
            user="user",
            key_path="/fake/key",
            retry_attempts=1,
        )

        conn._client = mock_client
        exit_code, _stdout, stderr = conn.execute("false", check=False)

        assert exit_code == 1
        assert stderr == "error message"

    def test_execute_without_connection(self, mock_paramiko):
        """Test executing command without being connected."""
        mock_client = MagicMock()
        mock_client.get_transport.return_value = None  # Not connected
        mock_paramiko["SSHClient"].return_value = mock_client

        conn = SSHConnection(
            host="example.com",
            user="user",
            password="test",  # Use password to avoid key file check
            retry_attempts=1,
        )

        # Mock execute to fail connection
        mock_client.connect.side_effect = Exception("Connection refused")

        with pytest.raises(SSHConnectionError, match="Failed to connect"):
            conn.execute("ls")

    def test_upload_file(self, mock_paramiko, tmp_path):
        """Test uploading a file."""
        # Create a test file
        local_file = tmp_path / "test.txt"
        local_file.write_text("test content")

        mock_client = MagicMock()
        mock_sftp = MagicMock()
        mock_client.open_sftp.return_value = mock_sftp
        mock_client.get_transport.return_value.is_active.return_value = True
        mock_paramiko["SSHClient"].return_value = mock_client

        conn = SSHConnection(
            host="example.com",
            user="user",
            key_path="/fake/key",
            retry_attempts=1,
        )

        conn._client = mock_client
        conn.upload_file(str(local_file), "/remote/path/test.txt")

        mock_sftp.put.assert_called_once_with(
            str(local_file),
            "/remote/path/test.txt",
        )

    def test_upload_file_nonexistent(self, mock_paramiko):
        """Test uploading nonexistent file."""
        mock_client = MagicMock()
        mock_client.get_transport.return_value.is_active.return_value = True
        mock_paramiko["SSHClient"].return_value = mock_client

        conn = SSHConnection(
            host="example.com",
            user="user",
            key_path="/fake/key",
            retry_attempts=1,
        )

        conn._client = mock_client

        with pytest.raises(FileNotFoundError, match="Local file not found"):
            conn.upload_file("/nonexistent/file.txt", "/remote/path/test.txt")

    def test_upload_string(self, mock_paramiko):
        """Test uploading string content as a file."""
        mock_client = MagicMock()
        mock_sftp = MagicMock()
        mock_file = MagicMock()
        mock_sftp.open.return_value.__enter__.return_value = mock_file
        mock_client.open_sftp.return_value = mock_sftp
        mock_client.get_transport.return_value.is_active.return_value = True
        mock_paramiko["SSHClient"].return_value = mock_client

        conn = SSHConnection(
            host="example.com",
            user="user",
            key_path="/fake/key",
            retry_attempts=1,
        )

        conn._client = mock_client
        conn.upload_string("test content", "/remote/path/test.txt")

        mock_sftp.open.assert_called_once_with("/remote/path/test.txt", "w")
        mock_file.write.assert_called_once_with("test content")

    def test_download_file(self, mock_paramiko, tmp_path):
        """Test downloading a file."""
        local_file = tmp_path / "test.txt"

        mock_client = MagicMock()
        mock_sftp = MagicMock()
        mock_client.open_sftp.return_value = mock_sftp
        mock_client.get_transport.return_value.is_active.return_value = True
        mock_paramiko["SSHClient"].return_value = mock_client

        conn = SSHConnection(
            host="example.com",
            user="user",
            key_path="/fake/key",
            retry_attempts=1,
        )

        conn._client = mock_client
        conn.download_file("/remote/path/test.txt", str(local_file))

        mock_sftp.get.assert_called_once_with(
            "/remote/path/test.txt",
            str(local_file),
        )

    def test_is_connected(self, mock_paramiko):
        """Test checking connection status."""
        mock_client = MagicMock()
        mock_transport = MagicMock()
        mock_transport.is_active.return_value = True
        mock_client.get_transport.return_value = mock_transport
        mock_paramiko["SSHClient"].return_value = mock_client

        conn = SSHConnection(
            host="example.com",
            user="user",
            key_path="/fake/key",
            retry_attempts=1,
        )

        assert not conn.is_connected()

        conn._client = mock_client
        assert conn.is_connected()

        conn._client = None
        assert not conn.is_connected()

    def test_file_exists(self, mock_paramiko):
        """Test checking if remote file exists."""
        mock_client = MagicMock()
        mock_sftp = MagicMock()
        mock_sftp.stat.return_value = MagicMock()  # Stat succeeds
        mock_client.open_sftp.return_value = mock_sftp
        mock_client.get_transport.return_value.is_active.return_value = True
        mock_paramiko["SSHClient"].return_value = mock_client

        conn = SSHConnection(
            host="example.com",
            user="user",
            key_path="/fake/key",
            retry_attempts=1,
        )

        conn._client = mock_client
        assert conn.file_exists("/remote/path/file.txt")

        # Test file doesn't exist
        mock_sftp.stat.side_effect = FileNotFoundError()
        assert not conn.file_exists("/remote/path/nonexistent.txt")

    def test_mkdir(self, mock_paramiko):
        """Test creating remote directory."""
        mock_client = MagicMock()
        mock_sftp = MagicMock()
        mock_client.open_sftp.return_value = mock_sftp
        mock_client.get_transport.return_value.is_active.return_value = True
        mock_paramiko["SSHClient"].return_value = mock_client

        conn = SSHConnection(
            host="example.com",
            user="user",
            key_path="/fake/key",
            retry_attempts=1,
        )

        conn._client = mock_client
        conn.mkdir("/remote/path/dir")

        # Should create parent directories
        assert mock_sftp.mkdir.called

    def test_retry_on_connection_failure(self, mock_paramiko, tmp_path):
        """Test connection retry logic."""
        # Create a fake key file
        key_file = tmp_path / ".ssh" / "id_rsa"
        key_file.parent.mkdir(parents=True)
        key_file.write_text("fake key")

        mock_client = MagicMock()

        # Fail twice, then succeed
        mock_client.connect.side_effect = [
            Exception("Connection refused"),
            Exception("Connection refused"),
            None,  # Success on third try
        ]

        mock_paramiko["SSHClient"].return_value = mock_client

        conn = SSHConnection(
            host="example.com",
            user="user",
            key_path=str(key_file),
            retry_attempts=3,
            retry_delay=0.01,
        )

        conn.connect()

        # Should have tried 3 times
        assert mock_client.connect.call_count == 3

    def test_retry_exhausted(self, mock_paramiko, tmp_path):
        """Test connection failure after retries exhausted."""
        # Create a fake key file
        key_file = tmp_path / ".ssh" / "id_rsa"
        key_file.parent.mkdir(parents=True)
        key_file.write_text("fake key")

        mock_client = MagicMock()
        mock_client.connect.side_effect = Exception("Connection refused")
        mock_paramiko["SSHClient"].return_value = mock_client

        conn = SSHConnection(
            host="example.com",
            user="user",
            key_path=str(key_file),
            retry_attempts=2,
            retry_delay=0.01,
        )

        with pytest.raises(SSHConnectionError, match="Failed to connect"):
            conn.connect()

        # Should have tried max_retries times
        assert mock_client.connect.call_count == 2

    def test_execute_script(self, mock_paramiko):
        """Test executing a multi-line script."""
        mock_client = MagicMock()
        mock_stdin = MagicMock()
        mock_stdout = MagicMock()
        mock_stderr = MagicMock()

        mock_stdout.read.return_value = b"output"
        mock_stderr.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0

        mock_client.exec_command.return_value = (mock_stdin, mock_stdout, mock_stderr)
        mock_client.get_transport.return_value.is_active.return_value = True
        mock_paramiko["SSHClient"].return_value = mock_client

        conn = SSHConnection(
            host="example.com",
            user="user",
            key_path="/fake/key",
            retry_attempts=1,
        )

        conn._client = mock_client
        script = "echo hello\necho world"
        exit_code, _stdout, _stderr = conn.execute_script(script)

        assert exit_code == 0
        # Should have wrapped in bash -c
        assert mock_client.exec_command.called
