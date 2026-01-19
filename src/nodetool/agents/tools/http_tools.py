import os
from typing import Any, ClassVar

import aiofiles
import aiohttp

from nodetool.agents.tools.base import Tool
from nodetool.workflows.processing_context import ProcessingContext


class DownloadFileTool(Tool):
    """
    A tool that downloads files from URLs and saves them to disk.

    This tool enables language models to retrieve files of any type from the web
    and save them to the workspace directory for further processing or analysis.
    Supports downloading multiple files in parallel.
    """

    name = "download_file"
    description = "Download a text or binaryfile from a URL and save it to disk"
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL of the file to download",
            },
            "output_file": {
                "type": "string",
                "description": "Workspace relative path where to save the file",
            },
        },
        "required": ["url", "output_file"],
    }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        """
        Downloads a file from a URL and saves it to the specified path.

        Args:
            context: The processing context
            params: Dictionary including:
                url (str): URL of the file to download
                output_file (str): Workspace relative path where to save the file
                headers (dict, optional): HTTP headers for the request
                timeout (int, optional): Timeout for the request in seconds

        Returns:
            dict: Result containing download status information
        """
        try:
            # Handle both single URL and list of URLs
            url = params.get("url")
            output_file = params.get("output_file")

            if not url:
                return {"error": "URL is required"}
            if not output_file:
                return {"error": "Output file is required"}

            headers = params.get("headers", {})
            default_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "Accept": "*/*",
                "Accept-Language": "en-US,en;q=0.9",
            }
            merged_headers = {**default_headers, **headers}
            timeout = params.get("timeout", 60)

            # Create a semaphore to limit concurrent downloads

            # Ensure the directory exists
            full_path = context.resolve_workspace_path(output_file)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            async with (
                aiohttp.ClientSession() as session,
                session.get(url, headers=merged_headers, timeout=timeout) as response,
            ):
                if response.status != 200:
                    return {
                        "url": url,
                        "output_file": output_file,
                        "success": False,
                        "error": f"HTTP request failed with status {response.status}",
                        "status_code": response.status,
                    }

                # Get content type and size
                content_type = response.headers.get("Content-Type", "unknown")
                content_length = response.headers.get("Content-Length")
                file_size = int(content_length) if content_length else None

                # Read the file data and write to disk
                async with aiofiles.open(full_path, "wb") as f:
                    await f.write(await response.read())

                return {
                    "url": url,
                    "output_file": output_file,
                    "success": True,
                    "content_type": content_type,
                    "file_size_bytes": file_size,
                }

        except Exception as e:
            return {"error": f"Error in download process: {str(e)}"}

    def user_message(self, params: dict) -> str:
        url = params.get("url", "a URL")
        output = params.get("output_file", "a file")
        msg = f"Downloading from {url} to {output}..."
        if len(msg) > 80:
            msg = f"Downloading file to {output}..."
        if len(msg) > 80:
            msg = "Downloading a file..."
        return msg
