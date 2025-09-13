import asyncio
import os
from typing import Any

from langchain_openai import ChatOpenAI
from nodetool.agents.tools.base import Tool
from nodetool.config.environment import Environment
from nodetool.workflows.processing_context import ProcessingContext

# Browseer Use
os.environ["ANONYMIZED_TELEMETRY"] = "false"


class BrowserUseTool(Tool):
    """
    Browser agent tool that uses browser_use under the hood.

    This tool provides functionality for running browser-based agents using the browser_use library.
    The agent can perform complex web automation tasks like form filling, navigation, data extraction,
    and multi-step workflows using natural language instructions.

    Use cases:
    - Perform complex web automation tasks based on natural language.
    - Automate form filling and data entry.
    - Scrape data after complex navigation or interaction sequences.
    - Automate multi-step web workflows.
    """

    name = "browser_agent"  # Name aligned with display name intention
    description = "Run a browser agent to perform complex web automation tasks based on natural language instructions."
    input_schema = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Natural language description of the browser task to perform. Can include complex multi-step instructions like 'Compare prices between websites', 'Fill out forms', or 'Extract specific data'.",
            },
            "timeout": {
                "type": "integer",
                "description": "Maximum time in seconds to allow for task completion. Complex tasks may require longer timeouts.",
                "default": 300,
                "minimum": 1,
                "maximum": 3600,
            },
            "use_remote_browser": {
                "type": "boolean",
                "description": "Use a remote browser instead of a local one. Requires BROWSER_URL.",
                "default": True,
            },
        },
        "required": ["task"],
    }
    example = """
    browser_agent(
        task="Go to google.com, search for 'best AI tools', and return the first 3 organic results including their titles and URLs.",
        model="gpt-4o",
        timeout=600
    )
    """

    def get_container_env(self) -> dict[str, str]:
        env_vars = {}
        api_key = Environment.get("OPENAI_API_KEY")
        if api_key:
            env_vars["OPENAI_API_KEY"] = api_key
        endpoint = Environment.get("BROWSER_URL")
        if endpoint:
            env_vars["BROWSER_URL"] = endpoint
        return env_vars

    def user_message(self, params: dict) -> str:
        task = params.get("task", "a browser task")
        msg = f"Running browser agent for task: '{task}'..."
        if len(msg) > 80:
            msg = "Running browser agent..."
        return msg

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        """
        Execute a browser agent task using browser_use.
        """
        from browser_use import Agent, Browser as BrowserUse, BrowserConfig

        task = params.get("task")
        if not task:
            return {"success": False, "task": task, "error": "Task is required"}

        timeout = params.get("timeout", 300)
        use_remote_browser = params.get("use_remote_browser", True)

        llm = None
        # Select LLM based on model name
        openai_api_key = Environment.get("OPENAI_API_KEY")
        if not openai_api_key:
            return {
                "success": False,
                "task": task,
                "error": "OpenAI API key not found in environment variables (OPENAI_API_KEY).",
            }
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)

        browser_instance = None
        try:
            if use_remote_browser:
                browser_endpoint = Environment.get("BROWSER_URL")
                if not browser_endpoint:
                    raise ValueError(
                        "Browser endpoint not found in environment variables (BROWSER_URL)."
                    )
                browser_instance = BrowserUse(
                    config=BrowserConfig(
                        headless=True,
                        wss_url=browser_endpoint,
                        # Removed browser_timeout
                    )
                )
            else:
                # Use local Playwright browser
                browser_instance = BrowserUse(
                    config=BrowserConfig(
                        headless=True,  # Default to headless for local as well
                        # Removed browser_timeout
                    )
                )

            # Create and run the agent
            agent = Agent(
                task=task,
                llm=llm,
                browser=browser_instance,
                # Consider adding agent-specific timeouts or configurations if needed
            )

            try:
                # Use asyncio.wait_for for overall task timeout
                result = await asyncio.wait_for(agent.run(), timeout=timeout)
                return {
                    "success": True,
                    "task": task,
                    "result": result,
                    "error": None,
                }
            except asyncio.TimeoutError:
                return {
                    "success": False,
                    "task": task,
                    "error": f"Task timed out after {timeout} seconds",
                    "result": None,
                }
            except Exception as agent_run_e:
                return {
                    "success": False,
                    "task": task,
                    "error": f"Browser agent execution failed: {str(agent_run_e)}",
                    "result": None,
                }

        except Exception as setup_e:
            return {
                "success": False,
                "task": task,
                "error": f"Browser agent setup failed: {str(setup_e)}",
                "result": None,
            }
        finally:
            if browser_instance and hasattr(browser_instance, "close"):
                try:
                    await browser_instance.close()
                except Exception as close_e:
                    print(f"Error closing browser instance: {close_e}")


# Example usage for testing
async def main():
    # Load environment variables (ensure OPENAI_API_KEY and potentially BROWSER_URL are set)
    # Environment.load_env() # Removed this line as it caused a linter error

    tool = BrowserUseTool()
    context = ProcessingContext()  # Create a dummy context for testing
    params = {
        "task": "Go to https://news.ycombinator.com/news and get the top 10 stories.",
        "timeout": 120,
        "use_remote_browser": False,  # Set to True if you have BrightData configured
    }

    print(f"Running test task: {params['task']}")
    result = await tool.process(context, params)
    print("Result:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
