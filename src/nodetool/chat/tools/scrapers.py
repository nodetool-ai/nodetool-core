"""
Web scraping tools using Brightdata's Datasets API.

This module provides tools for scraping data from multiple platforms including LinkedIn,
Facebook, Twitter, Instagram, TikTok, Reddit, Pinterest, Quora, Vimeo, and YouTube
via Brightdata's specialized Datasets API endpoints.
"""

import json
import traceback
from typing import Any, List, Dict, Optional
import asyncio

import aiohttp  # Add this import for asyncio.sleep()


from nodetool.common.environment import Environment
from nodetool.workflows.processing_context import ProcessingContext
from .base import Tool


def get_required_api_key(key_name, error_message=None):
    """Get a required API key from environment variables."""
    api_key = Environment.get(key_name)
    if not api_key:
        raise ValueError(
            error_message or f"{key_name} not found in environment variables."
        )
    return api_key


class BrightdataScraperTool(Tool):
    """
    A tool that allows scraping data from multiple platforms using Brightdata's Datasets API.

    This tool enables language models to retrieve data from various social media and content
    platforms without directly interacting with a browser, using Brightdata's specialized
    Datasets API endpoints with a three-step process: trigger collection, monitor progress,
    and download results.
    """

    name = "brightdata_scraper"
    description = "Scrape data from multiple platforms (LinkedIn, Facebook, Twitter, Instagram, TikTok, Reddit, Pinterest, Quora, Vimeo, YouTube) using Brightdata's Datasets API"
    input_schema = {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of URLs to scrape from supported platforms",
            },
            "max_wait_time": {
                "type": "integer",
                "description": "Maximum time to wait for results in seconds",
                "default": 300,
            },
            "poll_interval": {
                "type": "integer",
                "description": "Interval between status checks in seconds",
                "default": 2,
            },
        },
        "required": ["urls"],
    }

    def __init__(self, workspace_dir: str):
        super().__init__(workspace_dir)
        self.session = None

    async def _ensure_session(self):
        """Ensure an HTTP session exists"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def _make_api_request(
        self, api_url: str, method: str = "GET", payload: Any = None
    ):
        """Make an API request and handle common error cases."""
        try:
            session = await self._ensure_session()
            api_key = get_required_api_key(
                "BRIGHTDATA_API_KEY",
                "Brightdata API key not found. Please provide it in the secrets as 'BRIGHTDATA_API_KEY'.",
            )
            if isinstance(api_key, dict) and "error" in api_key:
                return api_key

            # Brightdata-specific request preparation
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            async with session.request(
                method, api_url, headers=headers, json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return {
                        "error": f"API request failed with status {response.status}: {error_text} using url: {api_url} and payload: {payload}"
                    }

                return await response.json()
        except Exception as e:
            return {"error": f"Error making API request: {str(e)}"}

    async def _make_brightdata_request(
        self, endpoint: str, method: str = "GET", payload: Any = None
    ) -> Dict[str, Any]:
        """
        Generic function to make requests to Brightdata's Datasets API.

        Args:
            endpoint: Complete API endpoint path (including any ID values)
            method: HTTP method (GET, POST, etc.)
            payload: Data to send in the request body (for POST requests)

        Returns:
            dict: API response
        """
        # Construct the full URL
        api_url = f"https://api.brightdata.com/datasets/v3/{endpoint}"

        # Make the API request (API key handling is done in _make_api_request)
        return await self._make_api_request(
            api_url=api_url,
            method=method,
            payload=payload,
        )

    async def _trigger_data_collection(self, urls: List[str]) -> Dict[str, Any]:
        """
        Triggers data collection for the specified URLs.

        Args:
            urls: List of URLs to scrape

        Returns:
            dict: Response containing snapshot_id or error message
        """
        # Format payload as list of URL objects
        payload = [{"url": url} for url in urls]

        return await self._make_brightdata_request(
            endpoint="trigger", method="POST", payload=payload
        )

    async def _check_progress(self, snapshot_id: str) -> Dict[str, Any]:
        """
        Checks the progress of a data collection job.

        Args:
            snapshot_id: The ID of the snapshot to check

        Returns:
            dict: Response containing job status information
        """
        return await self._make_brightdata_request(endpoint=f"progress/{snapshot_id}")

    async def _download_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        """
        Downloads the data from a completed snapshot.

        Args:
            snapshot_id: The ID of the snapshot to download

        Returns:
            dict: Response containing the scraped data
        """
        return await self._make_brightdata_request(endpoint=f"snapshot/{snapshot_id}")

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        """
        Execute a multi-platform scraping operation using Brightdata's Datasets API.

        Args:
            context: The processing context which may contain API credentials
            params: Dictionary including:
                urls (List[str]): URLs to scrape from supported platforms
                max_wait_time (int): Maximum time to wait for results in seconds
                poll_interval (int): Interval between status checks in seconds

        Returns:
            dict: Scraped data or error message
        """
        try:
            urls = params.get("urls", [])
            if not urls:
                return {"error": "At least one URL is required"}

            max_wait_time = params.get("max_wait_time", 300)  # Default: 5 minutes
            poll_interval = params.get("poll_interval", 2)  # Default: 2 seconds

            # Step 1: Trigger data collection
            trigger_response = await self._trigger_data_collection(urls)
            if "error" in trigger_response:
                return trigger_response

            # Parse response and extract snapshot_id
            try:
                trigger_data = json.loads(trigger_response.get("body", "{}"))
                snapshot_id = trigger_data.get("snapshot_id")
                if not snapshot_id:
                    return {
                        "error": "Failed to obtain snapshot_id from API response",
                        "response": trigger_data,
                    }
            except (json.JSONDecodeError, AttributeError) as e:
                return {
                    "error": f"Failed to parse trigger response: {str(e)}",
                    "body": trigger_response,
                }

            # Step 2: Monitor progress
            elapsed_time = 0
            while elapsed_time < max_wait_time:
                progress_response = await self._check_progress(snapshot_id)
                if "error" in progress_response:
                    return progress_response

                try:
                    progress_data = json.loads(progress_response.get("body", "{}"))
                    status = progress_data.get("status")

                    if status == "completed":
                        break
                    elif status == "failed":
                        return {
                            "error": "Data collection failed",
                            "details": progress_data,
                        }

                except (json.JSONDecodeError, AttributeError) as e:
                    return {
                        "error": f"Failed to parse progress response: {str(e)}",
                        "body": progress_response,
                    }

                # Wait before polling again
                await asyncio.sleep(poll_interval)
                elapsed_time += poll_interval

            if elapsed_time >= max_wait_time:
                return {"error": "Timed out waiting for data collection to complete"}

            # Step 3: Download results
            download_response = await self._download_snapshot(snapshot_id)
            if "error" in download_response:
                return download_response

            try:
                return json.loads(download_response.get("body", "{}"))
            except json.JSONDecodeError as e:
                return {
                    "error": f"Failed to parse downloaded data: {str(e)}",
                    "body": download_response.get("body", "")[:500],
                }

        except Exception as e:
            traceback.print_exc()
            return {"error": f"Error performing web scraping: {str(e)}"}
        finally:
            # Close the session if it exists
            if self.session and not self.session.closed:
                await self.session.close()
