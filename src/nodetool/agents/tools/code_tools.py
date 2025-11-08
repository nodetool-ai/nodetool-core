from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from nodetool.agents.tools.base import Tool
from nodetool.code_runners.runtime_base import ContainerFailureError, StreamRunnerBase
from nodetool.code_runners.python_runner import PythonDockerRunner
from nodetool.code_runners.javascript_runner import JavaScriptDockerRunner
from nodetool.code_runners.bash_runner import BashDockerRunner
from nodetool.code_runners.ruby_runner import RubyDockerRunner
from nodetool.workflows.processing_context import ProcessingContext


RunnerMode = Literal["docker", "subprocess"]


class RunnerExecutionError(RuntimeError):
    """Wraps runner failures so partial stdout/stderr can be surfaced."""

    def __init__(
        self,
        message: str,
        *,
        stdout_lines: list[str],
        stderr_lines: list[str],
        mode: RunnerMode,
        original: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.stdout_lines = stdout_lines
        self.stderr_lines = stderr_lines
        self.mode = mode
        self.original = original


@lru_cache(maxsize=1)
def _docker_available() -> bool:
    """Return True when the Docker SDK can reach the daemon."""
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


class RunnerExecutionTool(Tool):
    """Base class for code-execution tools backed by StreamRunnerBase."""

    runner_cls: type[StreamRunnerBase]
    language_label: str = "code"
    network_disabled: bool = False

    input_schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Source code to execute",
            },
        },
        "required": ["code"],
    }

    docker_error_signatures = (
        "Docker daemon is not available",
        "Error while fetching server API version",
        "Cannot connect to the Docker daemon",
        "docker.errors",
        "No module named 'docker'",
    )

    def __init__(
        self,
        prefer_docker: bool | None = None,
        timeout_seconds: int = 60,
        runner_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.prefer_docker = prefer_docker
        self.timeout_seconds = timeout_seconds
        self.runner_kwargs = runner_kwargs or {}

    async def process(self, context: ProcessingContext, params: dict):
        code_to_execute = params.get("code", "")
        if not isinstance(code_to_execute, str) or not code_to_execute.strip():
            return {
                "success": False,
                "error": f"No {self.language_label} provided.",
            }

        mode = self._determine_mode(params)
        runner = self._build_runner(mode, params)
        env_locals = self.build_env_locals(context, runner)
        try:
            stdout_lines, stderr_lines = await self._execute_with_runner(
                runner,
                code_to_execute,
                context,
                env_locals,
            )
            return self._format_success(stdout_lines, stderr_lines, mode)
        except RunnerExecutionError as exc:
            if self._should_fallback_to_subprocess(exc):
                try:
                    fallback_mode: RunnerMode = "subprocess"
                    fallback_runner = self._build_runner(fallback_mode, params)
                    fallback_env = self.build_env_locals(context, fallback_runner)
                    stdout_lines, stderr_lines = await self._execute_with_runner(
                        fallback_runner,
                        code_to_execute,
                        context,
                        fallback_env,
                    )
                    return self._format_success(
                        stdout_lines,
                        stderr_lines,
                        fallback_mode,
                        fallback_reason=str(exc),
                    )
                except RunnerExecutionError as fallback_exc:
                    return self._format_error(fallback_exc)
            return self._format_error(exc)
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
            }

    def user_message(self, params: dict):
        code = params.get("code")
        msg = f"Executing {self.language_label}..."
        if code and len(code) < 50:
            msg = f"Executing {self.language_label}: '{code[:40]}...'"
        return msg

    def _determine_mode(self, params: dict[str, Any] | None = None) -> RunnerMode:
        if self.prefer_docker is True:
            return "docker"
        if self.prefer_docker is False:
            return "subprocess"
        return "docker" if _docker_available() else "subprocess"

    async def _execute_with_runner(
        self,
        runner: StreamRunnerBase,
        code: str,
        context: ProcessingContext,
        env_locals: dict[str, Any],
    ) -> tuple[list[str], list[str]]:
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        try:
            async for slot, value in runner.stream(
                user_code=code,
                env_locals=env_locals,
                context=context,
                node=None,
            ):
                if slot == "stdout":
                    stdout_lines.append(value)
                elif slot == "stderr":
                    stderr_lines.append(value)
        except ContainerFailureError as exc:
            print(f"Container execution failed with exit code {exc.exit_code}: {exc.args[0]}")
            print(f"mode: {self._runner_mode(runner)}")
            print(f"original: {exc}")
            stdout = "".join(stdout_lines)
            stderr = "".join(stderr_lines)
            print(f"stdout: {stdout}")
            print(f"stderr: {stderr}")
            raise RunnerExecutionError(
                f"Container execution failed with exit code {exc.exit_code}: {exc.args[0]}",
                stdout_lines=stdout_lines,
                stderr_lines=stderr_lines,
                mode=self._runner_mode(runner),
                original=exc,
            ) from exc
        except Exception as exc:
            raise RunnerExecutionError(
                str(exc),
                stdout_lines=stdout_lines,
                stderr_lines=stderr_lines,
                mode=self._runner_mode(runner),
                original=exc,
            ) from exc

        return stdout_lines, stderr_lines

    def _build_runner(
        self, mode: RunnerMode, params: dict[str, Any] | None
    ) -> StreamRunnerBase:
        workspace_mount = "host" if mode == "docker" else None
        call_options = self.runner_options_for_call(mode, params or {})
        runner_kwargs = {
            **self.runner_kwargs,
            **(call_options or {}),
            "mode": mode,
            "timeout_seconds": self.timeout_seconds,
            "network_disabled": self.network_disabled,
            "workspace_mount_path": workspace_mount,
        }
        return self.runner_cls(**runner_kwargs)

    def runner_options_for_call(
        self, mode: RunnerMode, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Hook for subclasses to tweak runner options per invocation."""
        return {}

    def build_env_locals(
        self, context: ProcessingContext, runner: StreamRunnerBase
    ) -> dict[str, Any]:
        workspace_path = runner.resolve_execution_workspace_path(context)
        if not workspace_path:
            workspace_path = context.workspace_dir
        env = {
            "WORKSPACE_DIR": workspace_path,
        }
        env.update(self.additional_env_locals(context, runner))
        return env

    def additional_env_locals(
        self, context: ProcessingContext, runner: StreamRunnerBase
    ) -> dict[str, Any]:
        return {}

    def _runner_mode(self, runner: StreamRunnerBase) -> RunnerMode:
        return "docker" if runner.mode == "docker" else "subprocess"

    def _format_success(
        self,
        stdout_lines: list[str],
        stderr_lines: list[str],
        mode: RunnerMode,
        fallback_reason: str | None = None,
    ) -> dict[str, Any]:
        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)
        response: dict[str, Any] = {
            "success": True,
            "result": stdout,
            "stdout": stdout,
            "stderr": stderr,
            "runner_mode": mode,
        }
        if fallback_reason:
            response["runner_note"] = (
                f"Fell back to subprocess after Docker error: {fallback_reason}"
            )
        return response

    def _format_error(self, error: RunnerExecutionError) -> dict[str, Any]:
        stdout = "".join(error.stdout_lines)
        stderr = "".join(error.stderr_lines)
        return {
            "success": False,
            "error": str(error),
            "stdout": stdout,
            "stderr": stderr,
            "runner_mode": error.mode,
        }

    def _should_fallback_to_subprocess(self, exc: RunnerExecutionError) -> bool:
        if exc.mode != "docker" or self.prefer_docker is True:
            return False
        original_message = ""
        if exc.original is not None:
            original_message = str(exc.original)
        message = (original_message or str(exc)).lower()
        stderr_text = "".join(exc.stderr_lines).lower()
        stdout_text = "".join(exc.stdout_lines).lower()
        combined = "\n".join([message, stderr_text, stdout_text])
        if any(sig.lower() in combined for sig in self.docker_error_signatures):
            return True
        if "modulenotfounderror" in combined or "importerror" in combined:
            return True
        return False


class ExecutePythonTool(RunnerExecutionTool):
    """
    Execute Python code in a sandboxed environment and return its output.
    """

    runner_cls = PythonDockerRunner
    language_label = "Python"
    name = "execute_python"
    description = """Execute Python code in a sandboxed environment and return its output.
    The code will be executed in the workspace directory. Choose the optional
    'datascience' environment to run inside the jupyter/datascience-notebook
    Docker image, which ships with a full data science stack (pandas, NumPy,
    SciPy, scikit-learn, matplotlib, etc.).
    You have following python libraries available:
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - scikit-learn
    - scipy
    - requests
    - beautifulsoup4
    """

class ExecuteDatascienceTool(RunnerExecutionTool):
    """Execute Python code in the jupyter/datascience-notebook Docker image."""

    runner_cls = PythonDockerRunner
    language_label = "Python"
    name = "execute_datascience"
    description = "Execute Python code in the jupyter/datascience-notebook Docker image."

    def runner_options_for_call(
        self, mode: RunnerMode, params: dict[str, Any]
    ) -> dict[str, Any]:
        return {"image": "jupyter/datascience-notebook"}

class ExecuteJavaScriptTool(RunnerExecutionTool):
    """Execute JavaScript (Node.js) code and stream stdout/stderr."""

    runner_cls = JavaScriptDockerRunner
    language_label = "JavaScript"
    name = "execute_javascript"
    description = (
        "Execute JavaScript (Node.js) code inside an isolated environment and "
        "stream stdout/stderr output."
    )


class ExecuteBashTool(RunnerExecutionTool):
    """Execute Bash shell commands and scripts."""

    runner_cls = BashDockerRunner
    language_label = "Bash"
    name = "execute_bash"
    description = (
        "Execute Bash shell commands in an isolated environment and stream stdout/stderr output."
    )


class ExecuteRubyTool(RunnerExecutionTool):
    """Execute Ruby scripts."""

    runner_cls = RubyDockerRunner
    language_label = "Ruby"
    name = "execute_ruby"
    description = (
        "Execute Ruby scripts in an isolated environment and stream stdout/stderr output."
    )
