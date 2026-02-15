import asyncio
from contextlib import suppress
from typing import AsyncGenerator
from uuid import uuid4

from nodetool.config.logging_config import get_logger
from nodetool.models.workspace import Workspace
from nodetool.runtime.resources import ResourceScope
from nodetool.types.job import JobUpdate
from nodetool.workflows.comfy_workflow_runner import run_comfy_workflow, should_use_comfy_runner
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.threaded_event_loop import ThreadedEventLoop
from nodetool.workflows.types import Error, ProcessingMessage
from nodetool.workflows.workflow_runner import WorkflowRunner

log = get_logger(__name__)


async def _resolve_workspace_dir(user_id: str, workspace_id: str | None) -> str | None:
    """
    Resolve the workspace directory path from a workspace_id.

    Args:
        user_id: The user ID to validate workspace ownership.
        workspace_id: The workspace ID to look up, can be None.

    Returns:
        The workspace path if found and accessible, None otherwise.
    """
    if not workspace_id:
        return None

    try:
        workspace = await Workspace.find(user_id, workspace_id)
        if workspace and workspace.is_accessible():
            return workspace.path
        elif workspace:
            log.warning(f"Workspace {workspace_id} exists but is not accessible")
        else:
            log.warning(f"Workspace {workspace_id} not found for user {user_id}")
    except Exception as e:
        log.error(f"Error resolving workspace {workspace_id}: {e}")

    return None


async def handle_runner_error(
    exception: Exception,
    runner: WorkflowRunner,
) -> AsyncGenerator[ProcessingMessage, None]:
    """Yield standardized error messages when the runner fails."""

    if runner.status == "running":
        runner.status = "error"

    error_message = str(exception)

    yield Error(message=error_message)
    yield JobUpdate(job_id=runner.job_id, status="failed", error=error_message)


async def process_message(
    context: ProcessingContext,
) -> AsyncGenerator[ProcessingMessage, None]:
    """
    Helper method to process and send individual messages.
    Yields the message to the caller.

    Args:
        context (ProcessingContext): The processing context
        req (RunJobRequest): The request object for the job.
    """
    msg = await context.pop_message_async()
    yield msg


async def process_workflow_messages(
    context: ProcessingContext,
    runner: WorkflowRunner,
    sleep_interval: float = 0.01,
    _explicit_types: bool = False,
) -> AsyncGenerator[ProcessingMessage, None]:
    """
    Process messages from a running workflow.

    Args:
        context (ProcessingContext): The processing context
        runner (WorkflowRunner): The workflow runner
        message_handler: Async function to handle messages
        sleep_interval (float): Time to sleep between message checks
        explicit_types (bool): Whether to wrap primitive types in explicit types
    """
    error_occurred = False
    try:
        log.debug("Starting workflow message processing")
        while runner.is_running():
            if context.has_messages():
                async for msg in process_message(context):
                    yield msg
            else:
                await asyncio.sleep(sleep_interval)

        # Process remaining messages
        while context.has_messages():
            async for msg in process_message(context):
                yield msg

        log.debug("Finished processing workflow messages")
    except Exception as e:
        log.exception(e)
        error_occurred = True
        async for msg in handle_runner_error(e, runner):
            yield msg
    finally:
        # Always drain pending messages, even if an error occurred
        while context.has_messages():
            async for msg in process_message(context):
                yield msg
        if error_occurred:
            raise


async def run_workflow(
    request: RunJobRequest,
    runner: WorkflowRunner | None = None,
    context: ProcessingContext | None = None,
    use_thread: bool = True,
    send_job_updates: bool = True,
    initialize_graph: bool = True,
    validate_graph: bool = True,
    event_loop: ThreadedEventLoop | None = None,
) -> AsyncGenerator[ProcessingMessage, None]:
    """
    Runs a workflow asynchronously, with the option to run in a separate thread.

    Args:
        request (RunJobRequest): The request object containing the necessary information for running the workflow.
        runner (WorkflowRunner | None): The workflow runner object. If not provided, a new instance will be created.
        context (ProcessingContext | None): The processing context object. If not provided, a new instance will be created.
        use_thread (bool): Whether to run the workflow in a separate thread. Defaults to False.
        send_job_updates (bool): Whether to send job updates to the client. Defaults to True.
        initialize_graph (bool): Whether to initialize the graph. Defaults to True.
        validate_graph (bool): Whether to validate the graph. Defaults to True.
        event_loop (ThreadedEventLoop | None): Optional persistent threaded event loop to schedule the workflow on
            when use_thread is True. If provided, this function will not create or close a new loop and will reuse
            the given loop. If not provided, a temporary loop is created and closed via context manager.

    Yields:
        Any: A generator that yields job updates and messages from the workflow.

    Raises:
        Exception: If an error occurs during the execution of the workflow.

    Returns:
        AsyncGenerator[Any, None]: An asynchronous generator that yields job updates and messages from the workflow.
    """
    # -- Resolve graph early so we can detect Comfy workflows before creating
    # a full WorkflowRunner / ProcessingContext. --
    run_mode: str | None = None
    if request.graph is None:
        from nodetool.models.workflow import Workflow as WorkflowModel

        wf = await WorkflowModel.find(request.user_id, request.workflow_id)
        if wf is not None:
            request.graph = wf.get_api_graph()
            run_mode = wf.run_mode

    # -- Route to Comfy runner when applicable --
    if should_use_comfy_runner(run_mode, request.graph):
        log.info(
            "Routing to Comfy backend runner: workflow_id=%s, run_mode=%s",
            request.workflow_id,
            run_mode,
        )
        job_id = uuid4().hex
        async for msg in run_comfy_workflow(
            graph=request.graph,  # type: ignore[arg-type]
            workflow_id=request.workflow_id,
            job_id=job_id,
        ):
            yield msg
        return

    if runner is None:
        runner = WorkflowRunner(job_id=uuid4().hex)

    if context is None:
        context = ProcessingContext(
            user_id=request.user_id,
            job_id=runner.job_id,
            auth_token=request.auth_token,
            workflow_id=request.workflow_id,
        )

    async def drain_pending_messages() -> AsyncGenerator[ProcessingMessage, None]:
        """Yield any queued messages that arrived after the runner stopped."""
        while context.has_messages():
            async for msg in process_message(context):
                yield msg

    async def run():
        try:
            if request.graph is None:
                log.info(f"Loading workflow graph for {request.workflow_id}")
                workflow = await context.get_workflow(request.workflow_id)
                if workflow is None:
                    raise Exception(f"Workflow {request.workflow_id} not found")
                # Support both API model and plain object with a 'graph' attribute
                if hasattr(workflow, "get_api_graph"):
                    request.graph = workflow.get_api_graph()  # type: ignore[attr-defined]
                elif hasattr(workflow, "graph"):
                    request.graph = workflow.graph
                else:
                    raise Exception("Workflow object does not provide a graph")

                # Set workspace_dir from workflow's workspace_id if available
                if context.workspace_dir is None and hasattr(workflow, "workspace_id"):
                    workspace_id = getattr(workflow, "workspace_id", None)
                    if workspace_id:
                        context.workspace_dir = await _resolve_workspace_dir(context.user_id, workspace_id)
                        if context.workspace_dir:
                            log.info(f"Using workspace_dir from workflow: {context.workspace_dir}")

            # Execute runner with configured options
            await runner.run(
                request,
                context,
                send_job_updates=send_job_updates,
                initialize_graph=initialize_graph,
                validate_graph=validate_graph,
            )
        except asyncio.CancelledError:
            if runner.status == "running":
                runner.status = "cancelled"
            raise
        except Exception:
            if runner.status == "running":
                runner.status = "error"
            raise

    if use_thread:
        # Running the workflow in a separate thread (via ThreadedEventLoop) is beneficial
        # in scenarios where:
        # 1. The calling environment is synchronous, or its asyncio event loop should not
        #    be blocked by the workflow's execution. This keeps the caller responsive.
        # 2. The workflow needs to be integrated into a larger, primarily synchronous
        #    multi-threaded application. ThreadedEventLoop provides a managed asyncio
        #    environment within a dedicated thread.
        # 3. The workflow's operations are potentially long-running or resource-intensive,
        #    and isolating them in a separate thread prevents interference with other
        #    operations in the main application thread or event loop.
        log.info(f"Running workflow in thread for {request.workflow_id}")
        if event_loop is not None:
            # Use provided persistent loop (do not close it here)
            if not event_loop.is_running:
                event_loop.start()
            run_future = event_loop.run_coroutine(run())

            try:
                async for msg in process_workflow_messages(context, runner):
                    yield msg
            except Exception as e:
                log.exception(e)
                run_future.cancel()
                async for msg in handle_runner_error(e, runner):
                    yield msg
                try:
                    run_future.result()
                except Exception as e:
                    log.exception(e)
                    async for msg in handle_runner_error(e, runner):
                        yield msg
                # Drain pending messages after error
                async for msg in drain_pending_messages():
                    yield msg
                raise
            else:
                try:
                    run_future.result()
                except Exception as e:
                    log.exception(e)
                    async for msg in handle_runner_error(e, runner):
                        yield msg
                # Drain pending messages after normal completion
                async for msg in drain_pending_messages():
                    yield msg
        else:
            # Backwards-compatible behavior: create a temporary loop for this run
            with ThreadedEventLoop() as tel:
                run_future = tel.run_coroutine(run())

                try:
                    async for msg in process_workflow_messages(context, runner):
                        yield msg
                except Exception as e:
                    log.exception(e)
                    run_future.cancel()
                    async for msg in handle_runner_error(e, runner):
                        yield msg
                    try:
                        run_future.result()
                    except Exception as e:
                        log.exception(e)
                        async for msg in handle_runner_error(e, runner):
                            yield msg
                    # Drain pending messages after error
                    async for msg in drain_pending_messages():
                        yield msg
                    raise
                else:
                    try:
                        run_future.result()
                    except Exception as e:
                        log.exception(e)
                        async for msg in handle_runner_error(e, runner):
                            yield msg
                    # Drain pending messages after normal completion
                    async for msg in drain_pending_messages():
                        yield msg

    else:
        async with ResourceScope():
            run_task = asyncio.create_task(run())

            try:
                async for msg in process_workflow_messages(context, runner):
                    yield msg
            except Exception as e:
                log.exception(e)
                run_task.cancel()
                with suppress(asyncio.CancelledError):
                    await run_task
                async for msg in handle_runner_error(e, runner):
                    yield msg
                async for msg in drain_pending_messages():
                    yield msg
                raise

            try:
                await run_task
            except Exception as exception:
                log.exception(exception)
                async for msg in handle_runner_error(exception, runner):
                    yield msg
                async for msg in drain_pending_messages():
                    yield msg
                raise
            async for msg in drain_pending_messages():
                yield msg
