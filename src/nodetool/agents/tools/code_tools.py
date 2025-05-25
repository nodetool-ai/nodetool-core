from nodetool.agents.tools.base import Tool
from nodetool.workflows.processing_context import ProcessingContext
import subprocess
import shlex


class ExecutePythonTool(Tool):
    """
    Execute Python code in a sandboxed environment and return its output
    """

    name = "execute_python"
    description = """Execute Python code in a sandboxed environment and return its output.
    The code will be executed in the workspace directory.
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
    input_schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute",
            },
        },
        "required": ["code"],
    }

    async def process(self, context: ProcessingContext, params: dict):
        try:
            # Extract the code to execute
            code_to_execute = params.get("code", "")
            workspace_dir = context.workspace_dir

            # Prepare the command to execute the code
            command = f"python -c {shlex.quote(code_to_execute)}"

            # Execute the command in a subprocess with the specified working directory
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, cwd=workspace_dir
            )

            # Check if the execution was successful
            if result.returncode == 0:
                return {"success": True, "result": result.stdout}
            else:
                return {"error": result.stderr}
        except Exception as e:
            return {"error": str(e)}

    def user_message(self, params: dict):
        code = params.get("code")
        msg = "Executing Python code..."
        if code and len(code) < 50:
            msg = f"Executing Python: '{code[:40]}...'"
        return msg
