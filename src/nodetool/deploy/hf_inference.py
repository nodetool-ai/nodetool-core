#!/usr/bin/env python3
"""
HuggingFace Inference Endpoint Handler for NodeTool

This module provides a custom handler compatible with HuggingFace Inference Endpoints.
It exposes NodeTool workflow execution through a standard inference interface.

The handler implements the HuggingFace custom handler interface:
- __init__: Initialize the handler
- __call__: Process inference requests

Usage:
    This handler is automatically loaded by HuggingFace Inference Endpoints
    when deployed as a custom container.

Environment Variables:
    PORT: Port to listen on (default: 8080, set by HF)
    HF_MODEL_DIR: Model directory (set by HF)
    NODETOOL_WORKFLOW_ID: Default workflow ID to run
    LOG_LEVEL: Logging level (default: INFO)
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any

# Configure logging
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class EndpointHandler:
    """
    Custom handler for HuggingFace Inference Endpoints.

    This handler processes inference requests by running NodeTool workflows.
    """

    def __init__(self, path: str = ""):
        """
        Initialize the handler.

        Args:
            path: Model directory path (provided by HuggingFace)
        """
        self.model_dir = path
        self.default_workflow_id = os.environ.get("NODETOOL_WORKFLOW_ID")
        self._initialized = False

        logger.info(f"Initializing EndpointHandler with model_dir: {path}")

        # Initialize NodeTool components
        self._setup_nodetool()

    def _setup_nodetool(self) -> None:
        """Set up NodeTool environment and components."""
        try:
            # Import NodeTool modules
            from nodetool.packages.registry import Registry

            # Initialize the registry to load available nodes
            registry = Registry.get_instance()
            logger.info(f"Registry initialized with {len(registry.list_nodes())} nodes")

            self._initialized = True
            logger.info("NodeTool setup completed successfully")

        except Exception as e:
            logger.error(f"Failed to initialize NodeTool: {e}")
            raise

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Process an inference request.

        Args:
            data: Input data with the following structure:
                {
                    "inputs": <workflow parameters>,
                    "workflow_id": <optional workflow ID>,
                    "parameters": <optional additional parameters>
                }

        Returns:
            dict with workflow results or error information
        """
        try:
            return asyncio.run(self._process_request(data))
        except Exception as e:
            logger.exception("Error processing request")
            return {"error": str(e), "status": "error"}

    async def _process_request(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Async implementation of request processing.

        Args:
            data: Input data

        Returns:
            dict with workflow results
        """
        from nodetool.types.graph import Graph
        from nodetool.workflows.processing_context import ProcessingContext
        from nodetool.workflows.run_job_request import RunJobRequest
        from nodetool.workflows.run_workflow import run_workflow

        # Extract workflow ID
        workflow_id = data.get("workflow_id") or self.default_workflow_id
        if not workflow_id:
            return {
                "error": "No workflow_id provided and NODETOOL_WORKFLOW_ID not set",
                "status": "error",
            }

        # Extract parameters
        inputs = data.get("inputs", {})
        parameters = data.get("parameters", {})

        # Merge inputs and parameters
        params = {**inputs, **parameters}

        logger.info(f"Running workflow {workflow_id} with params: {list(params.keys())}")

        # Create run request
        request = RunJobRequest(
            workflow_id=workflow_id,
            user_id="hf_inference",
            auth_token="hf_inference_token",
            params=params,
        )

        # Create processing context
        context = ProcessingContext(
            user_id="hf_inference",
            auth_token="hf_inference_token",
            workflow_id=workflow_id,
            job_id=None,
        )

        # Run the workflow and collect results
        results: dict[str, Any] = {}
        errors: list[str] = []

        try:
            async for message in run_workflow(
                request,
                context=context,
                use_thread=False,
                send_job_updates=True,
                initialize_graph=True,
                validate_graph=True,
            ):
                # Process workflow messages
                if hasattr(message, "type"):
                    msg_type = getattr(message, "type", "")

                    if msg_type == "job_update":
                        status = getattr(message, "status", "")
                        if status == "error":
                            error_msg = getattr(message, "error", "Unknown error")
                            errors.append(error_msg)

                    elif msg_type == "node_update":
                        status = getattr(message, "status", "")
                        if status == "completed":
                            result = getattr(message, "result", None)
                            node_id = getattr(message, "node_id", "")
                            if result is not None:
                                results[node_id] = result

                    elif msg_type == "output_update":
                        output_name = getattr(message, "output_name", "output")
                        value = getattr(message, "value", None)
                        if value is not None:
                            results[output_name] = value

        except Exception as e:
            logger.exception("Error during workflow execution")
            return {"error": str(e), "status": "error"}

        if errors:
            return {
                "error": "; ".join(errors),
                "status": "error",
                "partial_results": results,
            }

        return {
            "results": results,
            "status": "success",
            "workflow_id": workflow_id,
        }


# Flask app for HuggingFace Inference Endpoints
def create_app():
    """
    Create Flask application for HuggingFace Inference Endpoints.

    This is used when running as a standalone server rather than
    through the HuggingFace handler interface.
    """
    from flask import Flask, jsonify, request

    app = Flask(__name__)
    handler = EndpointHandler()

    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint."""
        return jsonify({"status": "healthy"})

    @app.route("/", methods=["POST"])
    def predict():
        """Main prediction endpoint."""
        try:
            data = request.get_json()
            result = handler(data)
            return jsonify(result)
        except Exception as e:
            logger.exception("Error processing prediction request")
            return jsonify({"error": str(e), "status": "error"}), 500

    @app.route("/info", methods=["GET"])
    def info():
        """Information endpoint."""
        return jsonify({
            "name": "NodeTool Inference Handler",
            "version": "1.0.0",
            "default_workflow_id": handler.default_workflow_id,
            "initialized": handler._initialized,
        })

    return app


def main():
    """
    Main entry point for running the handler as a standalone server.
    """
    port = int(os.environ.get("PORT", 8080))

    logger.info(f"Starting NodeTool HuggingFace Inference Handler on port {port}")

    app = create_app()
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
