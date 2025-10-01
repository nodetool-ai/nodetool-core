"""
Docker-based job execution for workflow isolation.

This module provides job execution using Docker containers for maximum isolation.
Each workflow runs in its own container using the 'nodetool' Docker image.
"""

import asyncio
import json
import logging
import os
from typing import Any
from uuid import uuid4
from docker.types import DeviceRequest

from nodetool.code_runners.runtime_base import StreamRunnerBase
from nodetool.config.logging_config import get_logger
from nodetool.config.settings import load_settings
from nodetool.models.job import Job
from nodetool.workflows.job_execution import JobExecution
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.types import (
    JobUpdate,
    NodeUpdate,
    NodeProgress,
)

log = get_logger(__name__)
log.setLevel(logging.DEBUG)


class NodetoolDockerRunner(StreamRunnerBase):
    """
    Docker runner for executing nodetool workflows with resource constraints.

    Uses the 'nodetool' Docker image to run `nodetool run --json <request>`.
    The user_code parameter is expected to contain the JSON-encoded RunJobRequest.

    Resource limits can be configured to prevent one job from monopolizing resources
    in a multi-user environment.
    """

    def __init__(
        self,
        mem_limit: str = "2g",
        nano_cpus: int = 2_000_000_000,  # 2 CPUs
        gpu_device_ids: list[int] | None = None,
        gpu_memory_reservation: str | None = None,
        **kwargs,
    ):
        """
        Initialize the Docker runner with resource constraints.

        Args:
            mem_limit: Memory limit (e.g., "2g", "512m")
            nano_cpus: CPU quota in nanoseconds (1_000_000_000 = 1 CPU)
            gpu_device_ids: List of GPU device IDs to make available (e.g., [0, 1])
                          None means all GPUs, [] means no GPUs
            gpu_memory_reservation: GPU memory reservation per device (e.g., "4g", "2048m")
                                   This helps prevent VRAM monopolization
        """
        super().__init__(
            image="nodetool",
            mode="docker",
            timeout_seconds=0,  # No timeout for workflow execution
            network_disabled=False,  # Workflows may need network access
            mem_limit=mem_limit,
            nano_cpus=nano_cpus,
            **kwargs,
        )
        self.gpu_device_ids = gpu_device_ids
        self.gpu_memory_reservation = gpu_memory_reservation

    def _create_container(
        self,
        client: Any,
        image: str,
        command: list[str] | None,
        environment: dict[str, str],
        context: Any,
        stdin_stream: Any,
    ) -> Any:
        """Create a container with GPU resource constraints if configured.

        Overrides base class to add GPU device requests for VRAM control.
        """
        import docker

        # Build device_requests for GPU allocation
        device_requests = None
        if self.gpu_device_ids is not None:
            if len(self.gpu_device_ids) > 0:
                # Specific GPU devices requested
                device_ids_str = ",".join(str(i) for i in self.gpu_device_ids)
                capabilities = [["gpu"]]

                device_request = DeviceRequest(
                    device_ids=[device_ids_str],
                    capabilities=capabilities,
                )

                # Add memory reservation if specified
                if self.gpu_memory_reservation:
                    # Docker doesn't directly support memory limits per GPU in device_request,
                    # but we can set it via environment variables that PyTorch/TF respect
                    # This is a soft limit enforced by the ML frameworks
                    environment["PYTORCH_CUDA_ALLOC_CONF"] = (
                        f"max_split_size_mb={self._parse_memory_mb(self.gpu_memory_reservation)}"
                    )

                device_requests = [device_request]
                log.info(
                    f"Allocating GPUs {device_ids_str} with {self.gpu_memory_reservation or 'no'} memory limit"
                )
            # else: gpu_device_ids is empty list, meaning no GPUs (device_requests stays None)
        # else: gpu_device_ids is None, meaning all available GPUs (handled by not specifying device_requests)

        log.debug("creating container with resource limits")
        container = client.containers.create(
            image=image,
            command=command,
            network_disabled=self.network_disabled,
            mem_limit=self.mem_limit,
            nano_cpus=self.nano_cpus,
            volumes={
                getattr(context, "workspace_dir", "/tmp"): {
                    "bind": "/workspace",
                    "mode": "rw",
                }
            },
            working_dir="/workspace",
            stdin_open=stdin_stream is not None,
            tty=False,
            detach=True,
            environment=environment,
            ipc_mode=self.ipc_mode,
            device_requests=device_requests,
        )

        log.debug(
            f"container created: id={getattr(container, 'id', None)} "
            f"mem={self.mem_limit} cpus={self.nano_cpus/1e9:.1f} "
            f"gpus={device_requests is not None}"
        )
        return container

    def _parse_memory_mb(self, mem_str: str) -> int:
        """Parse memory string (e.g., '2g', '512m') to MB."""
        mem_str = mem_str.lower().strip()
        if mem_str.endswith("g"):
            return int(float(mem_str[:-1]) * 1024)
        elif mem_str.endswith("m"):
            return int(float(mem_str[:-1]))
        else:
            # Assume bytes, convert to MB
            return int(float(mem_str) / (1024 * 1024))

    def build_container_command(
        self, user_code: str, env_locals: dict[str, Any]
    ) -> list[str]:
        """Build command to run nodetool CLI inside container.

        Args:
            user_code: JSON-encoded RunJobRequest string
            env_locals: Not used

        Returns:
            Command list with JSON passed as workflow argument
        """
        # The 'workflow' argument can accept a RunJobRequest JSON string directly
        return ["nodetool", "run", user_code, "--jsonl"]

    def build_container_environment(self, env: dict[str, Any]) -> dict[str, str]:
        """Build the environment dict for Docker, including all settings and secrets.

        This override loads all settings and secrets from the configuration system
        and makes them available as environment variables in the container.

        Args:
            env: Base environment mapping (usually empty for nodetool workflows)

        Returns:
            A string-to-string dictionary with settings, secrets, and env merged
        """
        # Start with base implementation
        result = super().build_container_environment(env)

        try:
            # Load settings and secrets from YAML files
            settings, secrets = load_settings()

            # Merge settings (non-sensitive configuration)
            for key, value in settings.items():
                if value is not None and str(value).strip():
                    try:
                        result[str(key)] = str(value)
                    except Exception:
                        log.debug(f"Could not convert setting {key} to string")

            # Merge secrets (API keys, tokens, passwords)
            for key, value in secrets.items():
                if value is not None and str(value).strip():
                    try:
                        result[str(key)] = str(value)
                    except Exception:
                        log.debug(f"Could not convert secret {key} to string")
                    value = os.getenv(key)
                    if value:
                        result[key] = value

            # Also include current environment variables (highest priority)
            # This allows runtime overrides
            for key in [
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "GEMINI_API_KEY",
                "HF_TOKEN",
                "REPLICATE_API_TOKEN",
                "ELEVENLABS_API_KEY",
                "FAL_API_KEY",
                "AIME_USER",
                "AIME_API_KEY",
                "GOOGLE_MAIL_USER",
                "GOOGLE_APP_PASSWORD",
                "SERPAPI_API_KEY",
                "BROWSER_URL",
                "DATA_FOR_SEO_LOGIN",
                "DATA_FOR_SEO_PASSWORD",
                "FONT_PATH",
                "COMFY_FOLDER",
                "CHROMA_PATH",
            ]:
                env_value = os.getenv(key)
                if env_value is not None and env_value.strip():
                    result[key] = env_value

            log.debug(
                f"Loaded {len(result)} environment variables for Docker container"
            )

        except Exception as e:
            log.warning(f"Error loading settings/secrets for Docker environment: {e}")

        return result


def _deserialize_processing_message(msg_dict: dict[str, Any]) -> Any:
    """Deserialize a processing message from dict to appropriate type."""
    msg_type = msg_dict.get("type")
    if msg_type is None:
        return None

    MESSAGE_TYPE_MAP = {
        "job_update": JobUpdate,
        "node_update": NodeUpdate,
        "node_progress": NodeProgress,
    }

    msg_class = MESSAGE_TYPE_MAP.get(msg_type)
    if msg_class:
        try:
            return msg_class(**msg_dict)
        except Exception as e:
            log.warning(f"Failed to deserialize {msg_type}: {e}")
            return msg_dict
    return msg_dict


class DockerJobExecution(JobExecution):
    """
    Execute a job in an isolated Docker container using proven StreamRunnerBase.

    This class manages workflow execution using Docker containers for maximum
    isolation. Each job runs in a fresh container using the 'nodetool' image.

    Features:
    - Complete process isolation via proven Docker communication patterns
    - Reliable stdout/stderr streaming via StreamRunnerBase
    - Automatic container cleanup
    - Real-time message forwarding
    - Cancellation support

    The container executes 'nodetool run --stdin --jsonl' and receives the
    RunJobRequest JSON via stdin.
    """

    def __init__(
        self,
        job_id: str,
        context: ProcessingContext,
        request: RunJobRequest,
        job_model: Job,
        runner: NodetoolDockerRunner,
    ):
        """
        Initialize Docker job execution.

        Args:
            job_id: Unique job identifier
            context: Processing context for the job
            request: Job request with workflow details
            job_model: Database model for the job
            runner: Docker runner instance
        """
        super().__init__(job_id, context, request, job_model)
        self._runner = runner
        self._job_model = job_model
        self._context = context
        self._execution_task: asyncio.Task | None = None

    def push_input_value(self, input_name: str, value: Any, source_handle: str) -> None:
        """Push an input value to the job execution."""
        raise NotImplementedError(
            "DockerJobExecution does not support push_input_value"
        )

    async def _execute_workflow(self, request_json: str) -> None:
        """
        Execute the workflow using the Docker runner.

        Args:
            request_json: Serialized RunJobRequest to pass as CLI argument
        """
        try:
            # Create a dummy node for the runner (required by API but not used)
            class DummyNode:
                id = f"docker-job-{self.job_id}"

            # Stream output from Docker container
            # Environment variables (settings and secrets) are automatically loaded
            # by NodetoolDockerRunner.build_container_environment()
            async for slot, line in self._runner.stream(
                user_code=request_json,  # JSON passed as command argument
                env_locals={},  # Not used for nodetool workflows
                context=self._context,
                node=DummyNode(),  # type: ignore
                stdin_stream=None,  # No stdin needed
            ):
                print(f"slot: {slot}, line: {line}")
                # Only process stdout (JSONL messages)
                if slot == "stdout":
                    await self._process_output_line(line.strip())
                elif slot == "stderr":
                    log.debug(f"Container stderr: {line.strip()}")

            # If we get here without errors, mark as completed if not already
            if self._status == "running":
                self._status = "completed"
                if self._job_model:
                    self._job_model.status = "completed"
                    await self._job_model.save()

        except Exception as e:
            log.error(f"Docker execution error: {e}")
            self._status = "error"
            self._error = str(e)
            if self._job_model:
                self._job_model.status = "error"
                self._job_model.error = str(e)
                await self._job_model.save()
            raise

    async def _process_output_line(self, line: str) -> None:
        """
        Process a single line of JSONL output from the container.

        Args:
            line: JSON line to process
        """
        if not line or not line.startswith("{"):
            return

        try:
            msg_dict = json.loads(line)
            msg = _deserialize_processing_message(msg_dict)

            if msg:
                # Send message to context
                self._context.post_message(msg)

                # Update status from job_update messages
                if isinstance(msg, JobUpdate):
                    self._status = msg.status
                    if msg.status == "completed":
                        self._result = msg.result
                    elif msg.status == "error":
                        self._error = msg.error

                    # Update database
                    if self._job_model:
                        try:
                            self._job_model.status = self._status
                            if self._error:
                                self._job_model.error = self._error
                            await self._job_model.save()
                        except Exception as db_err:
                            log.debug(f"Failed to update job model: {db_err}")

        except json.JSONDecodeError:
            log.debug(f"Non-JSON output: {line[:100]}")

    async def cancel(self) -> None:
        """Cancel the job by stopping the Docker runner."""
        if self.is_completed():
            log.warning(f"Job {self.job_id} already finished")
            return

        log.info(f"Cancelling Docker job {self.job_id}")
        self._status = "cancelled"

        try:
            # Stop the runner (which handles Docker cleanup)
            self._runner.stop()

            # Cancel the execution task if running
            if self._execution_task and not self._execution_task.done():
                self._execution_task.cancel()
                try:
                    await self._execution_task
                except asyncio.CancelledError:
                    pass

            # Update database
            if self._job_model:
                self._job_model.status = "cancelled"
                await self._job_model.save()

        except Exception as e:
            log.error(f"Error cancelling job: {e}")

    async def cleanup_resources(self) -> None:
        """Clean up Docker resources."""
        try:
            # Stop the runner first (handles container cleanup and breaks the stream)
            self._runner.stop()

            # Give a moment for the runner to actually stop
            await asyncio.sleep(0.1)

            # Now cancel and wait for execution task to finish
            if self._execution_task and not self._execution_task.done():
                self._execution_task.cancel()
                try:
                    await asyncio.wait_for(self._execution_task, timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    log.debug("Execution task cancelled or timed out during cleanup")

            log.info(f"Cleaned up Docker job {self.job_id}")

        except Exception as e:
            log.error(f"Error cleaning up: {e}")

    @classmethod
    async def create_and_start(
        cls,
        request: RunJobRequest,
        context: ProcessingContext,
        mem_limit: str | None = None,
        cpu_limit: float | None = None,
        gpu_device_ids: list[int] | None = None,
        gpu_memory_limit: str | None = None,
    ) -> "DockerJobExecution":
        """
        Create and start a new Docker-based job with resource constraints.

        This factory method:
        - Creates job ID and database record
        - Creates Docker runner with resource limits
        - Starts async execution task
        - Returns DockerJobExecution instance

        Args:
            request: Job request with workflow details
            context: Processing context for the job
            mem_limit: Memory limit (e.g., "2g", "512m").
                      Defaults to DOCKER_MEM_LIMIT env var or "2g"
            cpu_limit: CPU limit in cores (e.g., 2.0 for 2 CPUs).
                      Defaults to DOCKER_CPU_LIMIT env var or 2.0
            gpu_device_ids: List of GPU device IDs to allocate (e.g., [0, 1]).
                           None = all GPUs, [] = no GPUs.
                           Defaults to DOCKER_GPU_DEVICES env var (comma-separated)
            gpu_memory_limit: GPU memory limit per device (e.g., "4g").
                            Defaults to DOCKER_GPU_MEMORY_LIMIT env var

        Returns:
            DockerJobExecution instance with execution started
        """
        job_id = uuid4().hex

        # Load resource limits from environment if not specified
        mem_limit = mem_limit or os.getenv("DOCKER_MEM_LIMIT", "2g")
        cpu_limit = cpu_limit or float(os.getenv("DOCKER_CPU_LIMIT", "2.0"))

        # Parse GPU device IDs from environment if not specified
        if gpu_device_ids is None:
            gpu_devices_env = os.getenv("DOCKER_GPU_DEVICES", "")
            if gpu_devices_env:
                try:
                    gpu_device_ids = [
                        int(x.strip()) for x in gpu_devices_env.split(",") if x.strip()
                    ]
                except ValueError:
                    log.warning(f"Invalid DOCKER_GPU_DEVICES value: {gpu_devices_env}")
                    gpu_device_ids = None

        gpu_memory_limit = gpu_memory_limit or os.getenv("DOCKER_GPU_MEMORY_LIMIT")

        log.info(
            f"Starting Docker job {job_id} for workflow {request.workflow_id} "
            f"with limits: mem={mem_limit}, cpu={cpu_limit}, "
            f"gpus={gpu_device_ids}, gpu_mem={gpu_memory_limit}"
        )

        # Create job record in database
        job_model = Job(
            id=job_id,
            workflow_id=request.workflow_id,
            user_id=request.user_id,
            job_type=request.job_type,
            status="running",
            graph=request.graph.model_dump() if request.graph else {},
            params=request.params or {},
        )
        await job_model.save()

        # Prepare request JSON
        request_dict = request.model_dump()
        if request.graph:
            request_dict["graph"] = request.graph.model_dump()
        request_json = json.dumps(request_dict)

        # Create Docker runner with resource constraints
        nano_cpus = int(cpu_limit * 1_000_000_000)  # Convert to nanoseconds
        runner = NodetoolDockerRunner(
            mem_limit=mem_limit,
            nano_cpus=nano_cpus,
            gpu_device_ids=gpu_device_ids,
            gpu_memory_reservation=gpu_memory_limit,
        )

        # Create job instance
        job_instance = cls(
            job_id=job_id,
            context=context,
            request=request,
            job_model=job_model,
            runner=runner,
        )

        # Set status to running
        job_instance._status = "running"

        # Start execution task
        job_instance._execution_task = asyncio.create_task(
            job_instance._execute_workflow(request_json)
        )

        log.info(f"Started Docker job {job_id}")

        return job_instance

    def is_running(self) -> bool:
        """Check if the job is currently running."""
        return self._status == "running"

    def is_completed(self) -> bool:
        """Check if the job has completed successfully."""
        return self._status == "completed"

    @property
    def container_id(self) -> str | None:
        """Get the Docker container ID (managed internally by runner)."""
        # Container ID is managed by the runner, return the active one if available
        return getattr(self._runner, "_active_container_id", None)


# ---- Manual CLI for smoke testing ----
if __name__ == "__main__":
    import argparse
    from nodetool.types.graph import Graph, Node, Edge

    def _build_simple_workflow_graph() -> Graph:
        """Build a simple workflow graph for testing."""
        return Graph(
            nodes=[
                Node(
                    id="input_text",
                    type="nodetool.input.StringInput",
                    data={
                        "name": "text",
                        "label": "Input Text",
                        "value": "",
                    },
                ),
                Node(
                    id="format_text",
                    type="nodetool.text.FormatText",
                    data={
                        "template": "Docker test: {{ text }}",
                        "inputs": {
                            "text": {"node_id": "input_text", "output": "value"}
                        },
                    },
                ),
                Node(
                    id="output_result",
                    type="nodetool.output.StringOutput",
                    data={
                        "name": "result",
                        "value": "",
                        "inputs": {
                            "value": {"node_id": "format_text", "output": "value"}
                        },
                    },
                ),
            ],
            edges=[
                Edge(
                    id="edge_input_to_format",
                    source="input_text",
                    sourceHandle="output",
                    target="format_text",
                    targetHandle="text",
                ),
                Edge(
                    id="edge_format_to_output",
                    source="format_text",
                    sourceHandle="output",
                    target="output_result",
                    targetHandle="value",
                ),
            ],
        )

    async def _main() -> None:
        import logging

        parser = argparse.ArgumentParser(
            description="Smoke-test DockerJobExecution with a simple workflow"
        )
        parser.add_argument(
            "--text",
            default="Hello from Docker!",
            help="Text to process in the workflow",
        )
        parser.add_argument(
            "--timeout",
            type=int,
            default=30,
            help="Timeout in seconds to wait for completion",
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug logging",
        )

        args = parser.parse_args()

        # Enable debug logging if requested
        if args.debug:
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            )
            print("Debug logging enabled")

        print(f"Running Docker workflow with text: {args.text}")
        print("-" * 60)

        # Create workflow request
        request = RunJobRequest(
            user_id="smoke_test_user",
            auth_token="smoke_test_token",
            graph=_build_simple_workflow_graph(),
            params={"text": args.text},
        )

        # Create processing context
        context = ProcessingContext(
            user_id=request.user_id,
            auth_token=request.auth_token,
        )

        # Start the job
        job = await DockerJobExecution.create_and_start(request, context)

        print(f"Job ID: {job.job_id}")
        print(f"Container ID: {job.container_id}")
        print(f"Status: {job.status}")
        print("-" * 60)

        try:
            # Wait for completion
            print(f"Waiting up to {args.timeout} seconds for completion...")
            for i in range(args.timeout):
                print(
                    f"[{i}s] Status: {job.status}, Running: {job.is_running()}, Completed: {job.is_completed()}, Container: {job.container_id}"
                )

                if job.is_completed():
                    print(f"\n✓ Job completed after {i} seconds")
                    break
                elif job.status == "error":
                    print(f"\n✗ Job failed: {job.error}")
                    break

                # Show progress
                if i % 5 == 0 and i > 0:
                    print(f"  Still running... ({i}s elapsed, status: {job.status})")

                await asyncio.sleep(1)
            else:
                print(
                    f"\n⚠ Timeout after {args.timeout} seconds - final status: {job.status}"
                )

            # Print final status
            print("-" * 60)
            print(f"Final Status: {job.status}")
            if job.result:
                print(f"Result: {job.result}")
            if job.error:
                print(f"Error: {job.error}")

            # Check database record
            db_job = await Job.get(job.job_id)
            if db_job:
                print(f"Database status: {db_job.status}")
            else:
                print("⚠ No database record found")

        finally:
            print("-" * 60)
            print("Cleaning up...")
            await job.cleanup_resources()
            print("✓ Cleanup complete")

    asyncio.run(_main())
