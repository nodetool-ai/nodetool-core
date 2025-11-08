"""
Chain of Thought (CoT) Agent implementation with tool calling capabilities.

This module implements a Chain of Thought reasoning agent that can use large language
models (LLMs) from various providers (OpenAI, Anthropic, Ollama) to solve problems
step by step. The agent can leverage external tools to perform actions like mathematical
calculations, web browsing, file operations, and shell command execution.

The implementation provides:
1. A TaskPlanner class that creates a task list with dependencies
2. A TaskExecutor class that executes tasks in the correct order
3. An Agent class that combines planning and execution
4. Integration with the existing provider and tool system
5. Support for streaming results during reasoning
"""

import datetime
from nodetool.config.logging_config import get_logger
import json
import os
import asyncio
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import AsyncGenerator, List, Sequence, Any, Optional

from nodetool.agents.tools.code_tools import ExecutePythonTool
from nodetool.code_runners.runtime_base import StreamRunnerBase
from nodetool.config.settings import get_log_path
from nodetool.workflows.types import (
    Chunk,
    SubTaskResult,
    TaskUpdate,
    TaskUpdateEvent,
)
from nodetool.agents.task_executor import TaskExecutor
from nodetool.providers import BaseProvider
from nodetool.agents.task_planner import TaskPlanner
from nodetool.agents.tools.base import Tool
from nodetool.metadata.types import (
    Task,
    ToolCall,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.ui.console import AgentConsole
from nodetool.agents.base_agent import BaseAgent


log = get_logger(__name__)


def _create_macos_sandbox_profile(
    allow_network: bool = True,
    allowed_read_paths: list[str] | None = None,
    allowed_write_paths: list[str] | None = None,
    enable_logging: bool = True,
) -> str:
    """
    Create a macOS sandbox profile for agent execution.

    Uses a "deny write by default" approach - allows reading most files but
    restricts writes to only necessary directories for agent execution.

    Args:
        allow_network: Whether to allow network access (if False, network is denied)
        allowed_read_paths: Additional paths to allow reading from
        allowed_write_paths: Additional paths to allow writing to
        enable_logging: Whether to log sandbox violations to stderr

    Returns:
        Sandbox profile as a string
    """
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

    read_paths = [
        f"{home}/Library/Application Support",  # macOS app support
        f"{python_prefix}",  # Python environment (for __pycache__, etc.)
    ]

    # iterate over all python packages and add the site-packages directory
    for package in sys.path:
        read_paths.append(package)

    # add allowed read paths
    if allowed_read_paths:
        read_paths.extend(allowed_read_paths)

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

    for path in read_paths:
        profile_lines.append(f'(allow file-read* (subpath "{path}"))')

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
        return shutil.which("sandbox-exec") is not None
    except Exception:
        return False


def _wrap_command_with_sandbox(
    cmd: list[str], workspace_dir: str | None = None
) -> tuple[list[str], str | None]:
    """
    Wrap a command with sandbox-exec on macOS.

    Args:
        cmd: Original command to execute
        workspace_dir: Workspace directory to allow writing to

    Returns:
        Tuple of (wrapped_command, temp_profile_path)
        temp_profile_path will be None if sandbox is not used
    """
    if not _should_use_sandbox():
        return cmd, None

    try:
        enable_logging = True
        allow_network = True

        # Parse additional allowed paths from environment
        allowed_read_paths = []
        allowed_write_paths = []

        # Add workspace directory to write paths
        if workspace_dir:
            allowed_read_paths.append(workspace_dir)
            allowed_write_paths.append(workspace_dir)

        # Create sandbox profile
        profile = _create_macos_sandbox_profile(
            allow_network=allow_network,
            allowed_read_paths=allowed_read_paths,
            allowed_write_paths=allowed_write_paths,
            enable_logging=enable_logging,
        )

        # Write profile to temporary file
        fd, profile_path = tempfile.mkstemp(
            suffix=".sb", prefix="nodetool_agent_sandbox_"
        )
        try:
            os.write(fd, profile.encode("utf-8"))
        finally:
            os.close(fd)

        # Wrap command with sandbox-exec
        wrapped_cmd = ["sandbox-exec", "-f", profile_path] + cmd

        log.info(f"Wrapping agent command with sandbox-exec: {' '.join(wrapped_cmd)}")
        log.info(f"Sandbox profile path: {profile_path}")
        if enable_logging:
            log.info(
                "Sandbox debug logging ENABLED - violations will be logged to stderr"
            )
        log.debug(f"Sandbox profile:\n{profile}")

        return wrapped_cmd, profile_path

    except Exception as e:
        log.warning(f"Failed to setup sandbox, running without sandbox: {e}")
        return cmd, None


def _get_cpu_limit(resource_limits: Any | None = None) -> int | None:
    """
    Get CPU limit percentage from resource limits or environment.

    Args:
        resource_limits: ResourceLimits configuration

    Returns:
        CPU limit percentage or None if not configured
    """
    # Check per-agent limit first
    if resource_limits and hasattr(resource_limits, "cpu_percent"):
        if resource_limits.cpu_percent:
            return resource_limits.cpu_percent

    # Fall back to environment variable
    if cpu_limit := os.environ.get("NODETOOL_AGENT_CPU_LIMIT"):
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
        resource_limits: ResourceLimits configuration

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
        log.warning(
            f"CPU limit requested ({cpu_percent}%) but taskpolicy is only available on macOS"
        )
        return cmd, {"taskpolicy_warning": "taskpolicy not available (not macOS)"}

    # Check if taskpolicy is available
    if not shutil.which("taskpolicy"):
        log.warning("CPU limit requested but taskpolicy not available")
        return cmd, {"taskpolicy_warning": "taskpolicy not available"}

    # Map CPU percentage to taskpolicy class
    taskpolicy_class = _cpu_percent_to_taskpolicy_class(cpu_percent)

    # Wrap command with taskpolicy
    wrapped_cmd = ["taskpolicy", "-c", taskpolicy_class] + cmd

    log.info(
        f"Applying CPU limit to agent: {cpu_percent}% -> taskpolicy class '{taskpolicy_class}'"
    )
    log.debug(f"CPU-limited command: {' '.join(wrapped_cmd)}")

    resource_info = {
        "cpu_percent": cpu_percent,
        "taskpolicy_class": taskpolicy_class,
        "taskpolicy_used": True,
    }

    return wrapped_cmd, resource_info


class AgentRunner(StreamRunnerBase):
    """Runner for executing agents with streaming output.

    Supports Docker containers and subprocess execution with optional sandboxing
    on macOS using sandbox-exec.
    """

    def __init__(
        self,
        timeout_seconds: int = 600,
        image: str = "nodetool",
        mem_limit: str = "2g",
        nano_cpus: int = 2_000_000_000,
        network_disabled: bool = False,
        mode: str = "docker",
        resource_limits: Any | None = None,
    ):
        """
        Initialize the AgentRunner.

        Args:
            timeout_seconds: Max time in seconds before the container is force removed
            image: Docker image to use for execution
            mem_limit: Docker memory limit (e.g., "2g")
            nano_cpus: CPU quota in Docker nano-CPUs (2e9 = 2 CPUs)
            network_disabled: Whether to disable network access
            mode: Execution mode ("docker" or "subprocess")
            resource_limits: Optional resource limits for CPU throttling
        """
        super().__init__(
            timeout_seconds=timeout_seconds,
            image=image,
            mem_limit=mem_limit,
            nano_cpus=nano_cpus,
            network_disabled=network_disabled,
            mode=mode,
        )
        self.resource_limits = resource_limits

    def build_container_command(
        self, user_code: str, env_locals: dict[str, Any]
    ) -> list[str]:
        """
        Build the command to run the agent inside the container.

        Args:
            user_code: Path to the agent configuration file
            env_locals: Additional environment variables

        Returns:
            Command list for the container
        """
        return [
            "python",
            "-m",
            "nodetool.agents.docker_runner",
            user_code,
        ]

    def wrap_subprocess_command(
        self, command: list[str], context: ProcessingContext
    ) -> tuple[list[str], Any]:
        """
        Wrap subprocess command with sandbox-exec and CPU limiting on macOS.

        Args:
            command: The command to wrap
            context: Processing context (provides workspace_dir)

        Returns:
            Tuple of (wrapped_command, cleanup_data)
            cleanup_data contains the sandbox profile path for cleanup
        """
        # Apply CPU limiting first (innermost wrapper)
        command, resource_info = _wrap_command_with_cpu_limit(
            command, self.resource_limits
        )
        if resource_info:
            log.info(f"CPU limit applied to agent: {resource_info}")

        # Apply sandboxing (outermost wrapper)
        workspace_dir = getattr(context, "workspace_dir", None)
        command, sandbox_profile_path = _wrap_command_with_sandbox(
            command, workspace_dir
        )

        return command, sandbox_profile_path

    def cleanup_subprocess_wrapper(self, cleanup_data: Any) -> None:
        """
        Clean up sandbox profile file if it was created.

        Args:
            cleanup_data: Sandbox profile path returned by wrap_subprocess_command
        """
        if cleanup_data:
            try:
                os.unlink(cleanup_data)
                log.debug(f"Cleaned up sandbox profile: {cleanup_data}")
            except OSError:
                pass


def sanitize_file_path(file_path: str) -> str:
    """
    Sanitize a file path by replacing spaces and slashes with underscores.

    Args:
        file_path (str): The file path to sanitize.

    Returns:
        str: The sanitized file path.
    """
    return file_path.replace(" ", "_").replace("/", "_").replace("\\", "_")


class Agent(BaseAgent):
    """
    🤖 Orchestrates AI-driven task execution using Language Models and Tools.

    The Agent class acts as a high-level controller that takes a complex objective,
    breaks it down into a step-by-step plan, and then executes that plan using
    a specified Language Model (LLM) and a set of available tools.

    Think of it as an intelligent assistant that can understand your goal, figure out
    the necessary actions (like searching the web, reading files, performing calculations,
    or running code), and carry them out autonomously to achieve the objective.

    Key Capabilities:
    - **Planning:** Decomposes complex objectives into manageable subtasks.
    - **Execution:** Runs the subtasks in the correct order, handling dependencies.
    - **Tool Integration:** Leverages specialized tools to interact with external
      systems or perform specific actions (e.g., file operations, web browsing,
      code execution).
    - **LLM Agnostic:** Works with different LLM providers (OpenAI, Anthropic, Ollama).
    - **Progress Tracking:** Can stream updates as the task progresses.
    - **Input/Output Management:** Handles input files and collects final results.

    Use this class to automate workflows that require reasoning, planning, and
    interaction with various data sources or tools.
    """

    def __init__(
        self,
        name: str,
        objective: str,
        provider: BaseProvider,
        model: str,
        planning_model: str | None = None,
        reasoning_model: str | None = None,
        tools: Optional[Sequence[Tool]] = None,
        description: str = "",
        inputs: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        max_subtasks: int = 10,
        max_steps: int = 50,
        max_subtask_iterations: int = 5,
        max_token_limit: int | None = None,
        output_schema: dict | None = None,
        enable_analysis_phase: bool = True,
        enable_data_contracts_phase: bool = True,
        task: Task | None = None,  # Add optional task parameter
        verbose: bool = True,  # Add verbose flag
        docker_image: str | None = None,
        use_sandbox: bool = False,
        resource_limits: Any | None = None,
        display_manager: AgentConsole | None = None,
    ):
        """
        Initialize the base agent.

        Args:
            name (str): The name of the agent
            objective (str): The objective of the agent
            description (str): The description of the agent
            provider (ChatProvider): An LLM provider instance
            model (str): The model to use with the provider
            reasoning_model (str, optional): The model to use for reasoning, defaults to the same as the provider model
            planning_model (str, optional): The model to use for planning, defaults to the same as the provider model
            tools (List[Tool]): List of tools available for this agent
            inputs (dict[str, Any], optional): Inputs to use for the agent
            system_prompt (str, optional): Custom system prompt
            max_steps (int, optional): Maximum reasoning steps
            max_subtask_iterations (int, optional): Maximum iterations per subtask
            max_token_limit (int, optional): Maximum token limit before summarization
            max_subtasks (int, optional): Maximum number of subtasks to be created
            output_schema (dict, optional): JSON schema for the final task output
            enable_analysis_phase (bool, optional): Whether to run the analysis phase (PHASE 2)
            enable_data_contracts_phase (bool, optional): Whether to run the data contracts phase (PHASE 3)
            task (Task, optional): Pre-defined task to execute, skipping planning
            verbose (bool, optional): Enable/disable console output (default: True)
            docker_image (str, optional): If set, execute the agent inside this Docker image.
            use_sandbox (bool, optional): If True and docker_image is not set, run in subprocess mode with sandbox (macOS only, default: False)
            resource_limits (optional): Resource limits for CPU throttling
        """
        super().__init__(
            name=name,
            objective=objective,
            provider=provider,
            model=model,
            tools=tools or [],
            inputs=inputs or {},
            system_prompt=system_prompt,
            max_token_limit=max_token_limit,
        )
        self.description = description
        self.planning_model = planning_model or model
        self.reasoning_model = reasoning_model or model
        self.max_steps = max_steps
        self.max_subtask_iterations = max_subtask_iterations
        self.max_subtasks = max_subtasks
        self.output_schema = output_schema
        self.enable_analysis_phase = enable_analysis_phase
        self.enable_data_contracts_phase = enable_data_contracts_phase
        self.initial_task = task
        if self.initial_task:
            self.task = self.initial_task
        self.verbose = verbose
        self.docker_image = docker_image
        self.use_sandbox = use_sandbox
        self.resource_limits = resource_limits
        self.display_manager = display_manager

    async def execute(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[Any, None]:
        """
        Execute the agent using the task plan.

        Args:
            context (ProcessingContext): The processing context

        Yields:
            Union[Message, Chunk, ToolCall]: Execution progress
        """
        # Execute in isolated environment if requested
        if self.docker_image:
            async for item in self._execute_in_isolated_env(context, mode="docker"):
                yield item
            return
        elif self.use_sandbox:
            async for item in self._execute_in_isolated_env(context, mode="subprocess"):
                yield item
            return

        tools = list(self.tools)
        task_planner_instance: Optional[TaskPlanner] = (
            None  # Keep track of planner instance
        )

        if self.task:  # If self.task is already set (e.g. by initial_task in __init__)
            # If self.task was set by initial_task, we skip planning.
            # We need to ensure it passes the None check for subsequent operations.
            pass
        else:
            if self.display_manager:
                log.debug(
                    "Agent '%s' planning task for objective: %s",
                    self.name,
                    self.objective,
                )
            self.provider.log_file = str(
                get_log_path(
                    sanitize_file_path(
                        f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}__{self.name}__planner.jsonl"
                    )
                )
            )

            task_planner_instance = TaskPlanner(
                provider=self.provider,
                model=self.planning_model,
                reasoning_model=self.reasoning_model,
                objective=self.objective,
                workspace_dir=context.workspace_dir,
                execution_tools=tools,
                inputs=self.inputs,
                output_schema=self.output_schema,
                enable_analysis_phase=self.enable_analysis_phase,
                enable_data_contracts_phase=self.enable_data_contracts_phase,
                verbose=self.verbose,
                display_manager=self.display_manager,
            )

            async for chunk in task_planner_instance.create_task(
                context, self.objective
            ):
                yield chunk

            if (
                task_planner_instance.task_plan
                and task_planner_instance.task_plan.tasks
            ):
                self.task = task_planner_instance.task_plan.tasks[0]

            assert (
                self.task is not None
            ), "Task was not created by planner and was not provided initially."

            yield TaskUpdate(
                task=self.task,
                event=TaskUpdateEvent.TASK_CREATED,
            )

        if self.output_schema and len(self.task.subtasks) > 0:
            self.task.subtasks[-1].output_schema = json.dumps(self.output_schema)

        tool_calls: List[ToolCall] = []

        # Start live display managed by AgentConsole
        if self.display_manager:
            self.display_manager.start_live(
                self.display_manager.create_execution_tree(
                    title=self.name, task=self.task, tool_calls=tool_calls
                )
            )

        try:
            executor = TaskExecutor(
                provider=self.provider,
                model=self.model,
                processing_context=context,
                tools=list(self.tools),  # Ensure it's a list of Tool
                task=self.task,
                system_prompt=self.system_prompt,
                inputs=self.inputs,
                max_steps=self.max_steps,
                max_subtask_iterations=self.max_subtask_iterations,
                max_token_limit=self.max_token_limit,
            )

            # Execute all subtasks within this task and yield results
            async for item in executor.execute_tasks(context):
                # Update tool_calls list if item is a ToolCall
                if isinstance(item, ToolCall):
                    tool_calls.append(item)

                # Create the updated table and update the live display
                if self.display_manager:
                    new_table = self.display_manager.create_execution_tree(
                        title=f"Task:\\n{self.objective}",
                        task=self.task,
                        tool_calls=tool_calls,
                    )
                    self.display_manager.update_live(new_table)

                # Yield the item
                if isinstance(item, ToolCall):
                    if item.name == "finish_task":
                        self.results = item.args["result"]
                        yield TaskUpdate(
                            task=self.task,
                            event=TaskUpdateEvent.TASK_COMPLETED,
                        )
                    if item.name == "finish_subtask" or item.name == "finish_task":
                        for subtask in self.task.subtasks:
                            if subtask.id == item.subtask_id and "result" in item.args:
                                yield SubTaskResult(
                                    subtask=subtask,
                                    result=item.args["result"],
                                    is_task_result=item.name == "finish_task",
                                )
                elif isinstance(item, TaskUpdate):
                    yield item
                    # Update provider log file when a subtask starts/completes
                    if item.event in [
                        TaskUpdateEvent.SUBTASK_STARTED,
                        TaskUpdateEvent.SUBTASK_COMPLETED,
                    ]:
                        timestamp = datetime.datetime.now().strftime(
                            "%Y-%m-%d_%H-%M-%S"
                        )
                        assert item.subtask is not None
                        self.provider.log_file = str(
                            get_log_path(
                                sanitize_file_path(
                                    f"{timestamp}__{self.name}__{item.subtask.id}.jsonl"
                                )
                            )
                        )
                elif isinstance(
                    item, (Chunk, ToolCall)
                ):  # Yield chunks and other tool calls too
                    yield item

        finally:
            # Ensure live display is stopped
            if self.display_manager:
                self.display_manager.stop_live()

        if self.display_manager:
            log.debug("Provider usage: %s", self.provider.usage)

    def get_results(self) -> Any:
        """
        Get the results produced by this agent.
        If a final result exists from finish_task, return that.
        Otherwise, return all collected results.

        Returns:
            List[Any]: Results with priority given to finish_task output
        """
        return self.results

    async def _execute_in_isolated_env(
        self,
        context: ProcessingContext,
        mode: str = "docker",
    ) -> AsyncGenerator[Chunk, None]:
        """
        Run the agent in an isolated environment (Docker or subprocess with sandbox).

        Args:
            context: Processing context
            mode: Execution mode ("docker" or "subprocess")
        """
        workspace = context.workspace_dir

        # Prepare workspace paths based on mode
        if mode == "docker":
            workspace_path = "/workspace"
            result_path = "/workspace/docker_result.json"
            config_filename = "docker_agent_config.json"
        else:
            workspace_path = workspace
            result_path = os.path.join(workspace, "subprocess_result.json")
            config_filename = "subprocess_agent_config.json"

        config = {
            "name": self.name,
            "objective": self.objective,
            "provider": self.provider.provider_name,
            "model": self.model,
            "planning_model": self.planning_model,
            "reasoning_model": self.reasoning_model,
            "tools": [t.__class__.__name__ for t in self.tools],
            "description": self.description,
            "inputs": self.inputs,
            "system_prompt": self.system_prompt,
            "max_subtasks": self.max_subtasks,
            "max_steps": self.max_steps,
            "max_subtask_iterations": self.max_subtask_iterations,
            "max_token_limit": self.max_token_limit,
            "output_schema": self.output_schema,
            "enable_analysis_phase": self.enable_analysis_phase,
            "enable_data_contracts_phase": self.enable_data_contracts_phase,
            "verbose": self.verbose,
            "workspace_dir": workspace_path,
            "result_path": result_path,
        }

        host_config = os.path.join(workspace, config_filename)
        with open(host_config, "w") as f:
            json.dump(config, f)

        # Collect environment variables from provider and tools
        env_vars: dict[str, str] = {}
        env_vars.update(self.provider.get_container_env(context))
        for tool in self.tools:
            env_vars.update(tool.get_container_env(context))

        # Create the runner with appropriate settings for the mode
        if mode == "docker":
            assert self.docker_image is not None, "Docker image is not set"
            runner = AgentRunner(
                timeout_seconds=600,
                image=self.docker_image,
                mem_limit="2g",
                nano_cpus=2_000_000_000,
                network_disabled=False,
                mode="docker",
                resource_limits=self.resource_limits,
            )
            config_path = "/workspace/" + config_filename
        else:
            # Subprocess mode with sandbox
            runner = AgentRunner(
                timeout_seconds=600,
                mode="subprocess",
                resource_limits=self.resource_limits,
            )
            config_path = host_config

        # Stream output from the runner
        async for slot, value in runner.stream(
            user_code=config_path,
            env_locals=env_vars,
            context=context,
            node=None,
        ):
            if slot == "stdout":
                yield Chunk(content=value)
            elif slot == "stderr":
                yield Chunk(content=f"[stderr] {value}")

        # Read results if available
        if mode == "docker":
            result_file = os.path.join(workspace, "docker_result.json")
        else:
            result_file = os.path.join(workspace, "subprocess_result.json")

        if os.path.exists(result_file):
            with open(result_file) as f:
                self.results = json.load(f)

        mode_name = "docker" if mode == "docker" else "sandbox"
        yield Chunk(content=f"\n[{mode_name} completed]\n", done=True)


async def test_docker_feature():
    """
    Smoke test for the Docker feature in Agent.
    Tests that an Agent can be initialized with a docker_image parameter.
    """
    from nodetool.providers.openai_provider import OpenAIProvider

    # Create a mock provider
    provider = OpenAIProvider()

    # Test that Agent can be initialized with docker_image parameter
    agent = Agent(
        name="test-docker-agent",
        objective="Write python code to calculate fibonacci numbers",
        provider=provider,
        model="gpt-4o-mini",
        enable_analysis_phase=False,
        enable_data_contracts_phase=False,
        docker_image="nodetool",
        tools=[ExecutePythonTool()],
    )

    context = ProcessingContext()

    async for item in agent.execute(context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    print(f"\nWorkspace: {context.workspace_dir}")
    print(f"Results: {agent.results}")
    print("✓ Docker feature smoke test passed")


async def test_sandbox_feature():
    """
    Smoke test for the Sandbox feature in Agent.
    Tests that an Agent can be initialized with use_sandbox=True parameter.
    """
    from nodetool.providers.openai_provider import OpenAIProvider

    # Create a mock provider
    provider = OpenAIProvider()

    # Test that Agent can be initialized with use_sandbox parameter
    agent = Agent(
        name="test-sandbox-agent",
        objective="Write python code to calculate prime numbers up to 100",
        provider=provider,
        model="gpt-4o-mini",
        enable_analysis_phase=False,
        enable_data_contracts_phase=False,
        use_sandbox=True,
        tools=[ExecutePythonTool()],
    )

    context = ProcessingContext()

    async for item in agent.execute(context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    print(f"\nWorkspace: {context.workspace_dir}")
    print(f"Results: {agent.results}")
    print("✓ Sandbox feature smoke test passed")


if __name__ == "__main__":
    # Run the smoke tests when the module is executed directly
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "sandbox":
        asyncio.run(test_sandbox_feature())
    else:
        asyncio.run(test_docker_feature())
