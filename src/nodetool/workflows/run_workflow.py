import asyncio
from typing import AsyncGenerator, Any
from uuid import uuid4
from nodetool.types.graph import Edge, Node, Graph
from nodetool.common.environment import Environment
from nodetool.types.job import JobUpdate
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.types import Error, ProcessingMessage
from nodetool.workflows.workflow_runner import WorkflowRunner
from nodetool.workflows.threaded_event_loop import ThreadedEventLoop
from nodetool.models.workflow import Workflow as WorkflowModel

log = Environment.get_logger()


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
    if isinstance(msg, Error):
        raise Exception(msg.error)
    else:
        yield msg


async def process_workflow_messages(
    context: ProcessingContext,
    runner: WorkflowRunner,
    sleep_interval: float = 0.01,
    explicit_types: bool = False,
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
        raise


async def run_workflow(
    request: RunJobRequest,
    runner: WorkflowRunner | None = None,
    context: ProcessingContext | None = None,
    use_thread: bool = True,
    send_job_updates: bool = True,
    initialize_graph: bool = True,
    validate_graph: bool = True,
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

    Yields:
        Any: A generator that yields job updates and messages from the workflow.

    Raises:
        Exception: If an error occurs during the execution of the workflow.

    Returns:
        AsyncGenerator[Any, None]: An asynchronous generator that yields job updates and messages from the workflow.
    """
    if context is None:
        context = ProcessingContext(
            user_id=request.user_id,
            auth_token=request.auth_token,
            workflow_id=request.workflow_id,
        )

    if runner is None:
        runner = WorkflowRunner(job_id=uuid4().hex)

    async def run():
        if request.graph is None:
            log.info(f"Loading workflow graph for {request.workflow_id}")
            workflow = await context.get_workflow(request.workflow_id)
            if workflow is None:
                raise Exception(f"Workflow {request.workflow_id} not found")
            # Support both API model and plain object with a 'graph' attribute
            if hasattr(workflow, "get_api_graph"):
                request.graph = workflow.get_api_graph()  # type: ignore[attr-defined]
            elif hasattr(workflow, "graph"):
                request.graph = getattr(workflow, "graph")
            else:
                raise Exception("Workflow object does not provide a graph")
        # Call with minimal positional args for compatibility with test runners
        await runner.run(request, context)

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
        with ThreadedEventLoop() as tel:
            run_future = tel.run_coroutine(run())

            try:
                async for msg in process_workflow_messages(context, runner):
                    yield msg
            except Exception as e:
                log.exception(e)
                run_future.cancel()
                yield Error(error=str(e))
                yield JobUpdate(job_id=runner.job_id, status="failed", error=str(e))
            try:
                run_future.result()
            except Exception as e:
                log.exception(e)
                yield Error(error=str(e))
                yield JobUpdate(job_id=runner.job_id, status="failed", error=str(e))

    else:
        run_task = asyncio.create_task(run())

        try:
            async for msg in process_workflow_messages(context, runner):
                yield msg
        except Exception as e:
            log.exception(e)
            run_task.cancel()
            yield JobUpdate(job_id=runner.job_id, status="failed", error=str(e))

        exception = run_task.exception()
        if exception:
            log.exception(exception)
            yield JobUpdate(job_id=runner.job_id, status="failed", error=str(exception))


# Test nodes for the main method


async def main():
    """Run a simple test workflow that calculates (a * b) + (c * d)."""
    # import test nodes
    from nodetool.workflows.test_nodes import NumberInput, Multiply, Add, NumberOutput

    # Create nodes
    input1 = Node(
        id="input1",
        type="nodetool.workflows.test_nodes.NumberInput",
        data={"value": 5.0},
    )
    input2 = Node(
        id="input2",
        type="nodetool.workflows.test_nodes.NumberInput",
        data={"value": 3.0},
    )
    input3 = Node(
        id="input3",
        type="nodetool.workflows.test_nodes.NumberInput",
        data={"value": 2.0},
    )
    input4 = Node(
        id="input4",
        type="nodetool.workflows.test_nodes.NumberInput",
        data={"value": 4.0},
    )

    multiply1 = Node(id="multiply1", type="nodetool.workflows.test_nodes.Multiply")
    multiply2 = Node(id="multiply2", type="nodetool.workflows.test_nodes.Multiply")

    add_node = Node(id="add", type="nodetool.workflows.test_nodes.Add")

    output = Node(id="output", type="nodetool.workflows.test_nodes.NumberOutput")

    # Create edges
    edges = [
        # Connect inputs to first multiplication
        Edge(
            source="input1", sourceHandle="output", target="multiply1", targetHandle="a"
        ),
        Edge(
            source="input2", sourceHandle="output", target="multiply1", targetHandle="b"
        ),
        # Connect inputs to second multiplication
        Edge(
            source="input3", sourceHandle="output", target="multiply2", targetHandle="a"
        ),
        Edge(
            source="input4", sourceHandle="output", target="multiply2", targetHandle="b"
        ),
        # Connect multiplications to addition
        Edge(source="multiply1", sourceHandle="output", target="add", targetHandle="a"),
        Edge(source="multiply2", sourceHandle="output", target="add", targetHandle="b"),
        # Connect addition to output
        Edge(
            source="add", sourceHandle="output", target="output", targetHandle="value"
        ),
    ]

    # Create graph
    graph = Graph(
        nodes=[input1, input2, input3, input4, multiply1, multiply2, add_node, output],
        edges=edges,
    )

    # Create workflow request
    req = RunJobRequest(
        user_id="test_user",
        auth_token="test_token",
        workflow_id="test_workflow",
        graph=graph,
    )

    print("Running test workflow: (5 * 3) + (2 * 4) = ?")
    print("-" * 50)

    # Run the workflow
    async for msg in run_workflow(req, use_thread=False):
        print(msg)

    print("-" * 50)
    print("Workflow completed!")


if __name__ == "__main__":
    asyncio.run(main())
