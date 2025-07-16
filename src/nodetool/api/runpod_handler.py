import asyncio
import dotenv
import json
import os
import runpod
from nodetool.common.environment import Environment
from nodetool.types.job import JobUpdate
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.workflow_runner import WorkflowRunner
from nodetool.workflows.threaded_event_loop import ThreadedEventLoop
from nodetool.workflows.types import Error

dotenv.load_dotenv()


log = Environment.get_logger()


async def async_generator_handler(job):
    """
    Asynchronous generator handler for RunPod serverless workflow execution.
    
    This function processes NodeTool workflow jobs on RunPod infrastructure.
    It supports both embedded workflows (baked into the Docker image) and
    traditional workflows (provided in job input).
    
    Workflow Discovery:
    1. Checks for embedded workflow via EMBEDDED_WORKFLOW_PATH environment variable
    2. If embedded workflow exists, uses that data and ignores job input workflow data
    3. Otherwise, uses workflow data from job input (traditional mode)
    
    Job Input Format:
    {
        "input": {
            "auth_token": "optional_auth_token",     # API authentication (optional)
            "workflow_id": "workflow_id",            # Required if no embedded workflow
            "user_id": "user_id",                    # Required if no embedded workflow  
            "graph": {...}                           # Optional workflow graph data
        },
        "id": "job_id"
    }
    
    Args:
        job (dict): RunPod job dictionary containing input data and job ID
        
    Yields:
        dict: Job status updates, progress messages, and results as they occur
        
    The generator yields messages until the workflow completes or fails:
    - JobUpdate messages with status and progress information
    - Error messages if execution fails
    - Final completion status
    """
    # Extract workflow data from the job input
    job_data = job.get("input", {})
    job_id = job.get("id")

    # Check if we have an embedded workflow (from Docker image)
    # This allows the same handler to work for both embedded and dynamic workflows
    embedded_workflow_path = Environment.get_env_variable("EMBEDDED_WORKFLOW_PATH")
    if embedded_workflow_path and os.path.exists(embedded_workflow_path):
        log.info(f"Using embedded workflow from {embedded_workflow_path}")
        with open(embedded_workflow_path, 'r') as f:
            embedded_workflow = json.load(f)
        
        # Override job data with embedded workflow information
        # Preserve auth_token from job input if provided
        job_data = {
            "workflow_id": embedded_workflow["id"],
            "user_id": embedded_workflow["user_id"],
            "auth_token": job_data.get("auth_token"),  # Optional authentication
            "graph": embedded_workflow["graph"]
        }

    req = RunJobRequest.model_validate(job_data)

    # Create processing context for workflow execution
    # This context manages the workflow runtime environment
    context = ProcessingContext(
        user_id=req.user_id,
        auth_token=req.auth_token,
        workflow_id=req.workflow_id,
        upload_assets_to_s3=True,  # Store generated assets in S3 for persistence
    )

    # Initialize workflow runner and event loop for async execution
    runner = WorkflowRunner(job_id=job_id)
    event_loop = ThreadedEventLoop()

    # Execute workflow in a separate thread to avoid blocking
    with event_loop as tel:

        async def run():
            """Inner async function to run the workflow."""
            try:
                # If no graph provided, fetch it from the database
                if req.graph is None:
                    workflow = await context.get_workflow(req.workflow_id)
                    req.graph = workflow.graph
                
                # Execute the workflow
                await runner.run(req, context)
            except Exception as e:
                # Log the exception and post failure message
                log.exception(e)
                context.post_message(
                    JobUpdate(job_id=job_id, status="failed", error=str(e))
                )

        # Start workflow execution in the thread
        run_future = tel.run_coroutine(run())

        # Stream messages while workflow is running
        while runner.is_running():
            if context.has_messages():
                # Get next message from the context queue
                msg = await context.pop_message_async()

                # Convert message to dictionary for JSON serialization
                msg_dict = msg.model_dump()
                print(msg_dict)  # Log for debugging

                yield msg_dict

                # Stop streaming on error or failure
                if isinstance(msg, Error):
                    return

                if isinstance(msg, JobUpdate) and msg.status == "failed":
                    return

            else:
                # Brief sleep to prevent busy waiting
                await asyncio.sleep(0.1)

        # Process any remaining messages after workflow completion
        while context.has_messages():
            msg = await context.pop_message_async()
            msg_dict = msg.model_dump()
            yield msg_dict

        # Ensure the workflow execution completes
        run_future.result()


# Example usage for local testing (commented out for production):
# This shows how to test the handler locally with a sample workflow
#
# import os
# os.environ["ENV"] = "production"
# workflow = load_example("Stable Diffusion in Comfy.json").model_dump()
#
# async def main():
#     async for msg in async_generator_handler({"input": workflow}):
#         print(msg)
#
# asyncio.run(main())

# Start the RunPod serverless handler
# This is the main entry point for RunPod execution
runpod.serverless.start({"handler": async_generator_handler})
