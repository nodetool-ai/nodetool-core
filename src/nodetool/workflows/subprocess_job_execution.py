"""
Subprocess job execution strategy for isolated workflow execution.
"""

import asyncio
import json
import os
import platform
import shutil
import sys
import tempfile
from asyncio import subprocess as aio_subprocess
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from nodetool.config.logging_config import get_logger
from nodetool.models.job import Job
from nodetool.runtime.resources import ResourceScope
from nodetool.types.job import JobUpdate
from nodetool.workflows.job_execution import JobExecution
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.types import (
    Error as WorkflowError,
)
from nodetool.workflows.types import (
    ProcessingMessage,
)

log = get_logger(__name__)


def type_to_name(type: type[ProcessingMessage]) -> str:
    return type.__annotations__["type"].__args__[0]


MESSAGE_TYPE_MAP: dict[str, Any] = {
    type_to_name(message_type): message_type for message_type in ProcessingMessage.__args__
}


def _create_macos_sandbox_profile(
    allow_network: bool = True,
    allowed_read_paths: list[str] | None = None,
    allowed_write_paths: list[str] | None = None,
    enable_logging: bool = True,
) -> str:
    """
    Create a macOS sandbox profile for subprocess execution.

    Uses a "deny write by default" approach - allows reading most files but
    restricts writes to only necessary directories for workflow execution.

    Args:
        allow_network: Whether to allow network access (if False, network is denied)
        allowed_read_paths: Additional paths to allow reading from
        allowed_write_paths: Additional paths to allow writing to
        enable_logging: Whether to log sandbox violations to stderr

    Returns:
        Sandbox profile as a string
    """
    import sys

    home = str(Path.home())
    python_prefix = sys.prefix  # Conda/virtualenv root

    # Default paths where writes are allowed
    default_write_paths = [
        f"{home}/.cache",  # User cache directory
        f"{home}/.nodetool-workspaces",  # NodeTool workspaces directory
        f"{home}/.config/nodetool",  # NodeTool config directory
        f"{home}/.local/share/nodetool",  # NodeTool data directory
        f"{home}/Library/Caches",  # macOS cache location
        "/tmp",  # System temp
        "/private/var/folders",  # macOS temp directory
        "/dev/null",  # Allow writing to /dev/null
    ]

    if allowed_write_paths:
        default_write_paths.extend(allowed_write_paths)

    # Build sandbox profile using Scheme-like syntax
    profile_lines = [
        "(version 1)",
    ]

    # Add debug/logging if enabled
    if enable_logging:
        profile_lines.extend(
            [
                "",
                ";; Enable violation logging for debugging",
                "(debug deny)",  # Log denials to stderr
            ]
        )

    profile_lines.extend(
        [
            "",
            ";; Allow most operations by default for compatibility",
            "(allow default)",
            "",
            ";; === WRITE RESTRICTIONS ===",
            ";; Deny all file writes by default, then allow specific paths",
            "(deny file-write*)",
            "",
            ";; Allow writes to user cache and config directories",
        ]
    )

    for path in default_write_paths:
        profile_lines.append(f'(allow file-write* (subpath "{path}"))')

    profile_lines.extend(
        [
            "",
            ";; === READ RESTRICTIONS ===",
            ";; Deny read access to sensitive system files",
            "(deny file-read*",
            '  (literal "/etc/master.passwd")',
            '  (literal "/etc/sudoers")',
            '  (subpath "/private/var/root")',
            '  (subpath "/private/var/db/sudo"))',
            "",
            ";; === SYSTEM OPERATION RESTRICTIONS ===",
            ";; Block kernel extension operations",
            "(deny system-kext*)",
            "",
            ";; Block privileged operations",
            "(deny system-privilege)",
        ]
    )

    # While reads are allowed by default (except for the sensitive paths denied
    # above), include any explicitly requested read paths to keep the profile
    # self-documenting and forward compatible if stricter read rules are added.
    if allowed_read_paths:
        profile_lines.extend(
            [";; Additional allowed read paths (explicitly requested)"]
            + [f'(allow file-read* (subpath "{path}"))' for path in allowed_read_paths]
        )

    # Network access control - only apply if explicitly disabled
    if not allow_network:
        profile_lines.extend(
            [
                "",
                ";; Deny network access (disabled by configuration)",
                "(deny network-outbound)",
                "(deny network-inbound)",
                "(deny network*)",
            ]
        )

    # Add note about security model
    profile_lines.extend(
        [
            "",
            ";; === SECURITY MODEL ===",
            ";; READS: Allowed everywhere except sensitive system files",
            ";; WRITES: Denied by default, allowed only in:",
            f";;   - {home}/.cache (user cache)",
            f";;   - {home}/.config/nodetool (config)",
            f";;   - {home}/.local/share/nodetool (data)",
            f";;   - {python_prefix} (Python environment)",
            ";;   - /tmp and /private/var/folders (temp directories)",
            ";; EXECUTION: Allowed (processes, forking, IPC)",
            ";; LIBRARIES: Allowed (dynamic loading, mmap)",
            f";; NETWORK: {'ALLOWED' if allow_network else 'DENIED'}",
        ]
    )

    return "\n".join(profile_lines)


def _should_use_sandbox() -> bool:
    """
    Determine if sandbox-exec should be used.

    Returns True if:
    - Running on macOS
    - NODETOOL_USE_SANDBOX env var is not set to "0" or "false"
    - sandbox-exec command is available
    """
    if platform.system() != "Darwin":
        return False

    # Check environment variable override
    use_sandbox = os.environ.get("NODETOOL_USE_SANDBOX", "1").lower()
    if use_sandbox in ("0", "false", "no"):
        return False

    # Verify sandbox-exec is available
    try:
        import shutil

        return shutil.which("sandbox-exec") is not None
    except Exception:
        return False


def _extract_workflow_write_paths(graph: Any) -> list[str]:
    """
    Extract folder paths from Save*File nodes in the workflow graph.

    These are paths the user explicitly wants to write to via their workflow,
    so they should be allowed by the sandbox.

    Args:
        graph: Workflow graph (Graph object or dict)

    Returns:
        List of expanded folder paths to allow for writing
    """
    write_paths = []

    # Handle both Graph objects and dicts
    nodes = []
    if hasattr(graph, "nodes"):
        nodes = graph.nodes
    elif isinstance(graph, dict) and "nodes" in graph:
        nodes = graph["nodes"]

    # Node types that write to user-specified folders
    SAVE_NODE_TYPES = {
        "nodetool.text.SaveTextFile",
        "nodetool.document.SaveDocumentFile",
        "nodetool.audio.SaveAudioFile",
        "nodetool.video.SaveVideoFile",
        "nodetool.image.SaveImageFile",
        "lib.bytes.SaveBytesFile",
        "nodetool.data.SaveCSVDataframeFile",
        "nodetool.dictionary.SaveCSVFile",
    }

    for node in nodes:
        # Get node type
        node_type = None
        if hasattr(node, "type"):
            node_type = node.type
        elif isinstance(node, dict) and "type" in node:
            node_type = node["type"]

        if not node_type or node_type not in SAVE_NODE_TYPES:
            continue

        # Get node data/properties
        node_data = None
        if isinstance(node, dict) and "data" in node:
            node_data = node["data"]
        elif hasattr(node, "data"):
            node_data = node.data  # type: ignore

        if not node_data:
            continue

        # Extract folder path
        folder = None
        if isinstance(node_data, dict) and "folder" in node_data:
            folder = node_data["folder"]
        elif hasattr(node_data, "folder"):
            folder = node_data.folder  # type: ignore

        if folder and isinstance(folder, str) and folder.strip():
            # Expand user paths like ~/Documents
            expanded = os.path.expanduser(folder.strip())
            if expanded not in write_paths:
                write_paths.append(expanded)
                log.info(f"Allowing sandbox write access to workflow output: {expanded}")

    return write_paths


def _wrap_command_with_sandbox(
    cmd: list[str], workflow_write_paths: list[str] | None = None
) -> tuple[list[str], str | None]:
    """
    Wrap a command with sandbox-exec on macOS.

    Args:
        cmd: Original command to execute
        workflow_write_paths: Additional write paths extracted from workflow nodes

    Returns:
        Tuple of (wrapped_command, temp_profile_path)
        temp_profile_path will be None if sandbox is not used
    """
    if not _should_use_sandbox():
        return cmd, None

    try:
        # Get configuration from environment
        allow_network = os.environ.get("NODETOOL_SANDBOX_ALLOW_NETWORK", "1").lower() not in ("0", "false", "no")

        # Check if debug logging is enabled
        enable_logging = os.environ.get("NODETOOL_SANDBOX_DEBUG", "1").lower() not in (
            "0",
            "false",
            "no",
        )

        # Parse additional allowed paths from environment
        allowed_read_paths = []
        if read_paths := os.environ.get("NODETOOL_SANDBOX_READ_PATHS"):
            allowed_read_paths = [p.strip() for p in read_paths.split(":") if p.strip()]

        allowed_write_paths = []
        if write_paths := os.environ.get("NODETOOL_SANDBOX_WRITE_PATHS"):
            allowed_write_paths = [p.strip() for p in write_paths.split(":") if p.strip()]

        # Add workflow-specific write paths (from Save*File nodes)
        if workflow_write_paths:
            allowed_write_paths.extend(workflow_write_paths)

        # Create sandbox profile
        profile = _create_macos_sandbox_profile(
            allow_network=allow_network,
            allowed_read_paths=allowed_read_paths,
            allowed_write_paths=allowed_write_paths,
            enable_logging=enable_logging,
        )

        # Write profile to temporary file
        fd, profile_path = tempfile.mkstemp(suffix=".sb", prefix="nodetool_sandbox_")
        try:
            os.write(fd, profile.encode("utf-8"))
        finally:
            os.close(fd)

        # Wrap command with sandbox-exec. Violation logging is controlled by
        # the profile's (debug deny), not a CLI flag.
        wrapped_cmd = ["sandbox-exec", "-f", profile_path, *cmd]

        log.info(f"Wrapping command with sandbox-exec: {' '.join(wrapped_cmd)}")
        log.info(f"Sandbox profile path: {profile_path}")
        if enable_logging:
            log.info("Sandbox debug logging ENABLED - violations will be logged to stderr")
        log.debug(f"Sandbox profile:\n{profile}")

        return wrapped_cmd, profile_path

    except Exception as e:
        log.warning(f"Failed to setup sandbox, running without sandbox: {e}")
        return cmd, None


def _get_cpu_limit(resource_limits: Any | None = None) -> int | None:
    """
    Get CPU limit percentage from RunJobRequest or environment.

    Args:
        resource_limits: ResourceLimits from RunJobRequest

    Returns:
        CPU limit percentage or None if not configured
    """
    # Check per-job limit first
    if resource_limits and hasattr(resource_limits, "cpu_percent") and resource_limits.cpu_percent:
        return resource_limits.cpu_percent

    # Fall back to environment variable
    if cpu_limit := os.environ.get("NODETOOL_SUBPROCESS_CPU_LIMIT"):
        try:
            return int(cpu_limit)
        except ValueError:
            log.warning(f"Invalid CPU limit value: {cpu_limit}")

    return None


def _cpu_percent_to_taskpolicy_class(cpu_percent: int) -> str:
    """
    Map CPU percentage to macOS taskpolicy class.

    Args:
        cpu_percent: Desired CPU percentage (1-100)

    Returns:
        taskpolicy class string ('background' or 'utility')
    """
    # taskpolicy classes:
    # - background: ~5% CPU, very low priority
    # - utility: reduced priority, suitable for background tasks
    #
    # Map percentages to classes:
    # - < 25%: background (very low priority)
    # - >= 25%: utility (reduced priority)
    if cpu_percent < 25:
        return "background"
    else:
        return "utility"


def _wrap_command_with_cpu_limit(
    cmd: list[str], resource_limits: Any | None = None
) -> tuple[list[str], dict[str, Any]]:
    """
    Wrap command with CPU limit using taskpolicy on macOS.

    On macOS, uses taskpolicy to set CPU scheduling class based on the
    requested CPU percentage:
    - < 25%: background class (~5% CPU)
    - >= 25%: utility class (reduced priority)
    - No limit or >= 100%: no throttling

    Args:
        cmd: Original command to execute
        resource_limits: ResourceLimits from RunJobRequest

    Returns:
        Tuple of (wrapped_command, resource_info)
        resource_info contains details about applied limits
    """
    cpu_percent = _get_cpu_limit(resource_limits)

    if not cpu_percent or cpu_percent >= 100:
        # No CPU limit configured or no throttling needed
        return cmd, {}

    # Only use taskpolicy on macOS
    if platform.system() != "Darwin":
        log.warning(f"CPU limit requested ({cpu_percent}%) but taskpolicy is only available on macOS")
        return cmd, {"taskpolicy_warning": "taskpolicy not available (not macOS)"}

    # Check if taskpolicy is available
    if not shutil.which("taskpolicy"):
        log.warning("CPU limit requested but taskpolicy not available")
        return cmd, {"taskpolicy_warning": "taskpolicy not available"}

    # Map CPU percentage to taskpolicy class
    taskpolicy_class = _cpu_percent_to_taskpolicy_class(cpu_percent)

    # Wrap command with taskpolicy
    wrapped_cmd = ["taskpolicy", "-c", taskpolicy_class, *cmd]

    log.info(f"Applying CPU limit: {cpu_percent}% -> taskpolicy class '{taskpolicy_class}'")
    log.debug(f"CPU-limited command: {' '.join(wrapped_cmd)}")

    resource_info = {
        "cpu_percent": cpu_percent,
        "taskpolicy_class": taskpolicy_class,
        "taskpolicy_used": True,
    }

    return wrapped_cmd, resource_info


def _deserialize_processing_message(payload: dict[str, Any]) -> Any:
    """Deserialize a processing message from a JSON payload."""
    msg_type: str | None = payload.get("type")
    if msg_type is None:
        return None
    cls = MESSAGE_TYPE_MAP.get(msg_type)
    if cls is None:
        return None
    try:
        if hasattr(cls, "model_validate"):
            return cls.model_validate(payload)  # type: ignore[attr-defined]
        return cls(**payload)
    except Exception:
        log.exception("Failed to deserialize message from subprocess", extra={"payload": payload})
        return None


class SubprocessJobExecution(JobExecution):
    """
    Job execution using a subprocess.

    This execution strategy runs workflows in a separate subprocess,
    providing better isolation and resource management.

    On macOS, subprocess execution can optionally use sandbox-exec for
    additional security isolation, restricting file system access and
    network operations.

    Additional Attributes:
        process: subprocess.Process instance
        _stdout_task: asyncio.Task streaming stdout JSONL
        _stderr_task: asyncio.Task streaming stderr logs
        _completed_event: asyncio.Event signalling process exit
        _sandbox_profile_path: Path to temporary sandbox profile (macOS only)
    """

    def __init__(
        self,
        job_id: str,
        context: ProcessingContext,
        request: RunJobRequest,
        job_model: Job,
        process: asyncio.subprocess.Process,
        sandbox_profile_path: str | None = None,
    ):
        super().__init__(job_id, context, request, job_model, runner=None)
        self.process = process
        self._stdout_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._completed_event = asyncio.Event()
        self._status = "running"
        self._sandbox_profile_path = sandbox_profile_path

    def push_input_value(self, input_name: str, value: Any, source_handle: str) -> None:
        """Push an input value to the job execution."""
        raise NotImplementedError("SubprocessJobExecution does not support push_input_value")

    def is_running(self) -> bool:
        """Check if the subprocess is still running."""
        if self._status in {"completed", "failed", "cancelled", "error"}:
            return False
        return self.process.returncode is None

    def is_completed(self) -> bool:
        """Check if the subprocess has completed."""
        if self._status in {"completed", "failed", "cancelled", "error"}:
            return True
        return self.process.returncode is not None

    def cancel(self) -> bool:
        """Cancel the running subprocess."""
        if self.is_completed():
            return False

        with suppress(ProcessLookupError):
            self.process.terminate()
        self._status = "cancelled"
        return True

    def cleanup_resources(self) -> None:
        """Clean up subprocess resources."""
        # Ensure subprocess terminated
        if self.process.returncode is None:
            with suppress(ProcessLookupError):
                self.process.kill()

        # Cancel background tasks
        for task in (self._stdout_task, self._stderr_task):
            if task is not None:
                task.cancel()

        # Clean up sandbox profile file if it exists
        if self._sandbox_profile_path:
            with suppress(OSError):
                os.unlink(self._sandbox_profile_path)
                log.debug(f"Cleaned up sandbox profile: {self._sandbox_profile_path}")

    async def _stream_stdout(self):
        """Stream and parse JSONL messages from subprocess stdout."""
        assert self.process.stdout is not None

        try:
            while True:
                line_bytes = await self.process.stdout.readline()
                if not line_bytes:
                    break

                line = line_bytes.decode("utf-8").strip()

                # Always log raw stdout for debugging
                if line:
                    log.info(line)

                if not line or not line.startswith("{"):
                    continue

                try:
                    payload = json.loads(line)
                    msg = _deserialize_processing_message(payload)
                    if msg is not None:
                        # Update internal status from job_update messages
                        if isinstance(msg, JobUpdate):
                            self._status = msg.status
                        # Forward message to context
                        self.context.post_message(msg)
                except json.JSONDecodeError:
                    log.warning(f"Failed to parse JSON from subprocess stdout: {line[:100]}")
                except Exception:
                    log.exception("Error processing subprocess message")

        except asyncio.CancelledError:
            log.debug(f"Subprocess stdout streaming cancelled for job {self.job_id}")
        except Exception:
            log.exception(f"Error streaming stdout for job {self.job_id}")

    async def _stream_stderr(self):
        """Stream subprocess stderr for logging."""
        assert self.process.stderr is not None

        try:
            while True:
                line_bytes = await self.process.stderr.readline()
                if not line_bytes:
                    break

                line = line_bytes.decode("utf-8").rstrip()
                if line:
                    log.info(line)

        except asyncio.CancelledError:
            log.debug(f"Subprocess stderr streaming cancelled for job {self.job_id}")
        except Exception:
            log.exception(f"Error streaming stderr for job {self.job_id}")

    async def _monitor_completion(self):
        """Monitor subprocess completion and update job status."""
        try:
            returncode = await self.process.wait()

            # Update status based on return code
            if returncode == 0:
                # Successful completion - update both internal status and database
                self._status = "completed"
                await self.job_model.update(status="completed", finished_at=datetime.now())
                log.info(f"Subprocess job {self.job_id} completed successfully")
            else:
                # Failed completion - the subprocess may not have sent a proper update
                # We need to handle this case and send the failure update
                self._status = "failed"

                # Provide more detailed error messages for common exit codes
                if returncode == -6:
                    error_msg = (
                        "Subprocess terminated (exit code -6) - likely sandbox policy violation. "
                        "Check stderr logs for sandbox denial messages. "
                        "Set NODETOOL_SANDBOX_DEBUG=1 for detailed sandbox logging, "
                        "or disable sandbox with NODETOOL_USE_SANDBOX=0."
                    )
                    if self._sandbox_profile_path:
                        error_msg += f" Sandbox profile: {self._sandbox_profile_path}"
                elif returncode == -9:
                    error_msg = "Subprocess killed (exit code -9) - likely out of memory or exceeded resource limits"
                elif returncode == -11:
                    error_msg = "Subprocess crashed (exit code -11) - segmentation fault"
                elif returncode < 0:
                    error_msg = f"Subprocess terminated by signal {-returncode}"
                else:
                    error_msg = f"Subprocess exited with code {returncode}"

                # Track error locally for fallback reporters
                self._error = error_msg
                await self.job_model.update(status="failed", error=error_msg, finished_at=datetime.now())
                self.context.post_message(JobUpdate(job_id=self.job_id, status="failed", error=error_msg))
                log.error(f"Subprocess job {self.job_id} failed with code {returncode}: {error_msg}")

            self._completed_event.set()

        except asyncio.CancelledError:
            self._status = "cancelled"
            await self.job_model.update(status="cancelled", finished_at=datetime.now())
            self._completed_event.set()
            log.info(f"Subprocess job {self.job_id} was cancelled")
        except Exception as e:
            self._status = "failed"
            import traceback

            error_msg = f"Error monitoring subprocess: {str(e)}"
            tb_text = traceback.format_exc()
            # Track error locally for fallback reporters
            self._error = error_msg
            await self.job_model.update(status="failed", error=error_msg, finished_at=datetime.now())
            self._completed_event.set()
            log.error(f"Error monitoring subprocess job {self.job_id}: {e}")
            # Provide a final update with traceback for better visibility
            with suppress(Exception):
                self.context.post_message(
                    JobUpdate(
                        job_id=self.job_id,
                        status="failed",
                        error=error_msg,
                        traceback=tb_text,
                    )
                )

    @classmethod
    async def create_and_start(cls, request: RunJobRequest, context: ProcessingContext) -> "SubprocessJobExecution":
        """
        Create and start a new subprocess-based job.

        This factory method:
        - Creates job ID and database record
        - Spawns subprocess running run_workflow_cli
        - Writes request JSON to subprocess stdin
        - Starts stdout/stderr streaming
        - Returns SubprocessJobExecution instance

        Args:
            request: Job request with workflow details
            context: Processing context for the job

        Returns:
            SubprocessJobExecution instance with execution started
        """
        job_id = uuid4().hex

        log.info(f"Starting subprocess job {job_id} for workflow {request.workflow_id}")

        # Create job record in database
        # Use a temporary ResourceScope for the initial database operation
        job_model = Job(
            id=job_id,
            workflow_id=request.workflow_id,
            user_id=request.user_id,
            job_type=request.job_type,
            status="running",
            graph=request.graph.model_dump() if request.graph else {},
            params=request.params or {},
        )

        # In test mode, inherit db_path from current scope if available
        async with ResourceScope():
            await job_model.save()

        # Prepare request JSON
        request_dict = request.model_dump()
        # Convert Graph to dict if needed
        if request.graph:
            request_dict["graph"] = request.graph.model_dump()

        request_json = json.dumps(request_dict)

        # Spawn subprocess using 'nodetool run --stdin --jsonl' CLI command
        # This will read the request JSON from stdin and output JSONL
        cmd = ["nodetool", "run", "--stdin", "--jsonl"]

        # Apply CPU limit using cpulimit if available
        cmd, resource_info = _wrap_command_with_cpu_limit(cmd, request.resource_limits)
        if resource_info:
            log.info(f"CPU limit applied for job {job_id}: {resource_info}")

        # Extract write paths from workflow Save*File nodes
        workflow_write_paths = []
        if request.graph:
            workflow_write_paths = _extract_workflow_write_paths(request.graph)

        # Wrap command with sandbox-exec on macOS if enabled
        cmd, sandbox_profile_path = _wrap_command_with_sandbox(cmd, workflow_write_paths)

        process = await aio_subprocess.create_subprocess_exec(
            *cmd,
            stdout=aio_subprocess.PIPE,
            stderr=aio_subprocess.PIPE,
            stdin=aio_subprocess.PIPE,
        )

        # Write request JSON to stdin and close it
        if process.stdin:
            process.stdin.write(request_json.encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close()

        # Create the job instance
        job_instance = cls(
            job_id=job_id,
            context=context,
            request=request,
            job_model=job_model,
            process=process,
            sandbox_profile_path=sandbox_profile_path,
        )

        # Start streaming tasks
        job_instance._stdout_task = asyncio.create_task(job_instance._stream_stdout())
        job_instance._stderr_task = asyncio.create_task(job_instance._stream_stderr())

        # Start monitoring completion
        job_instance._monitor_task = asyncio.create_task(job_instance._monitor_completion())

        log.info(f"Subprocess job {job_id} started with PID {process.pid}")

        return job_instance


async def _test_subprocess_execution():
    """
    Test subprocess job execution with CPU limits.

    This creates a simple workflow and runs it in a subprocess with
    CPU limits applied. Useful for testing on macOS with sandbox
    and CPU limiting features.
    """
    import logging

    from nodetool.types.graph import Edge, Graph
    from nodetool.types.graph import Node as GraphNode
    from nodetool.workflows.run_job_request import ResourceLimits

    os.environ["NODETOOL_SANDBOX_DEBUG"] = "1"

    # Enable INFO logging - subprocess stdout/stderr are logged at INFO/WARNING level
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create a workflow graph that saves a file to test sandbox permissions
    # This demonstrates that Save*File nodes automatically get write access
    test_folder = os.path.expanduser("~/Documents")

    print("=" * 60)
    print("Testing SubprocessJobExecution with Sandbox & File Save")
    print("=" * 60)
    print()
    print("📋 This test will:")
    print(f"   1. Save a text file to: {test_folder}")
    print("   2. Demonstrate sandbox auto-allows SaveTextFile paths")
    print("   3. Show subprocess stdout/stderr logging")
    print()
    print("📋 All subprocess stdout (INFO) and stderr (WARNING) will be shown")
    print("    Sandbox violations will be highlighted if they occur")
    print()

    graph = Graph(
        nodes=[
            GraphNode(
                id="text_1",
                type="nodetool.input.StringInput",
                data={
                    "value": "Hello from sandboxed subprocess!",
                },
            ),
            GraphNode(
                id="save_1",
                type="nodetool.text.SaveTextFile",
                data={
                    "folder": test_folder,
                    "name": "nodetool_sandbox_test_%Y%m%d_%H%M%S.txt",
                },
            ),
            GraphNode(
                id="output_1",
                type="nodetool.output.StringOutput",
                data={
                    "name": "result",
                },
            ),
        ],
        edges=[
            Edge(
                source="text_1",
                target="save_1",
                sourceHandle="output",
                targetHandle="text",
            ),
            Edge(
                source="save_1",
                target="output_1",
                sourceHandle="output",
                targetHandle="value",
            ),
        ],
    )

    # Configure CPU limit for testing (requires cpulimit)
    resource_limits = ResourceLimits(
        # cpu_percent=50,  # Limit to 50% CPU (requires cpulimit)
    )

    # Create RunJobRequest with resource limits
    request = RunJobRequest(
        workflow_id="test_workflow",
        user_id="test_user",
        auth_token="test_token",
        graph=graph,
        params={},
        job_type="workflow",
        resource_limits=resource_limits,
    )

    # Check sandbox status
    if _should_use_sandbox():
        print("\nSandbox: ENABLED (macOS sandbox-exec)")
        print(f"  Network: {'ALLOWED' if os.environ.get('NODETOOL_SANDBOX_ALLOW_NETWORK', '1') != '0' else 'DENIED'}")
        debug_enabled = os.environ.get("NODETOOL_SANDBOX_DEBUG", "1") != "0"
        print(f"  Debug Logging: {'ENABLED' if debug_enabled else 'DISABLED'}")
        if debug_enabled:
            print("    (Sandbox violations will be logged with '🔒 Sandbox' prefix)")
        print("  To disable sandbox: NODETOOL_USE_SANDBOX=0")
        print("  To disable sandbox debug: NODETOOL_SANDBOX_DEBUG=0")
    else:
        print("\nSandbox: DISABLED")
        if platform.system() == "Darwin":
            print("  (Set NODETOOL_USE_SANDBOX=1 to enable)")

    print("\n" + "-" * 60)
    print("Starting workflow execution...")
    print("-" * 60 + "\n")

    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id="test_workflow",
    )

    try:
        # Start the subprocess execution
        job_execution = await SubprocessJobExecution.create_and_start(request, context)

        print(f"\nJob started with ID: {job_execution.job_id}")
        print(f"Process PID: {job_execution.process.pid}")
        if job_execution._sandbox_profile_path:
            print(f"Sandbox profile: {job_execution._sandbox_profile_path}")
            print("  (You can inspect this file to see the sandbox policy)")
        print("Waiting for completion...\n")

        # Wait for the job to complete (with timeout)
        await asyncio.wait_for(job_execution._completed_event.wait(), timeout=60.0)

        print("\n" + "-" * 60)
        print("Workflow execution completed!")
        print("-" * 60)

        # Print results
        print(f"\nFinal Status: {job_execution._status}")
        print(f"Return Code: {job_execution.process.returncode}")

        # Check if sandbox profile still exists
        if job_execution._sandbox_profile_path:
            if os.path.exists(job_execution._sandbox_profile_path):
                print("\nSandbox profile still available at:")
                print(f"  {job_execution._sandbox_profile_path}")
                print(f"\nTo inspect it: cat {job_execution._sandbox_profile_path}")
            else:
                print("\nSandbox profile was cleaned up")

        # Note about stderr
        if job_execution.process.returncode == -6:
            print("\n" + "⚠️ " * 20)
            print("Exit code -6 detected. Look above for [STDERR] lines.")
            print("If no stderr was captured, the sandbox might be killing")
            print("the process before it can write output.")
            print("⚠️ " * 20)

        # Cleanup resources last
        job_execution.cleanup_resources()

        print("\n✅ Test completed!")
        return 0

    except TimeoutError:
        print("\n❌ Test timed out!")
        if "job_execution" in locals():
            job_execution.cancel()
            job_execution.cleanup_resources()
        return 1

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        log.exception("Test failed")
        if "job_execution" in locals():
            job_execution.cleanup_resources()
        return 1


def main():
    """Main entry point for testing."""
    return asyncio.run(_test_subprocess_execution())


if __name__ == "__main__":
    sys.exit(main())
