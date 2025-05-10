import os
import subprocess
import shlex
import asyncio  # Added for running async functions

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.tools.base import Tool


class GitStatusTool(Tool):
    """
    Retrieves the status of the Git repository, listing changed, new, deleted, and untracked files.
    """

    name = "git_status"
    description = "Retrieves the status of the Git repository, listing changed, new, deleted, and untracked files."
    input_schema = {
        "type": "object",
        "properties": {},  # No specific inputs for basic status
    }

    def __init__(self, repo_path: str | None = None):
        super().__init__()
        self.repo_path = repo_path

    async def process(self, context: ProcessingContext, params: dict):
        try:
            command = "git status --porcelain=v1 -uall"
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.repo_path or context.workspace_dir,
                check=False,
            )

            if result.returncode != 0:
                print(f"Git status command failed: {result.stderr}")
                return {
                    "success": False,
                    "error": f"Git status command failed: {result.stderr}",
                }

            files_status = []
            lines = result.stdout.strip().split("\n")
            for line in lines:
                if not line:
                    continue

                # Format: XY PATH or XY ORIG_PATH -> NEW_PATH
                status_xy = line[:2]
                path_part = line[3:]

                x_status = status_xy[0]
                y_status = status_xy[1]

                # Ignore ignored files
                if x_status == "!" and y_status == "!":
                    continue

                path = path_part
                orig_path = None

                if " -> " in path_part:
                    orig_path, path = path_part.split(" -> ", 1)

                files_status.append(
                    {
                        "path": path,
                        "x_status": x_status,  # Status of index
                        "y_status": y_status,  # Status of working tree
                        "orig_path": orig_path,  # For renames/copies
                    }
                )

            return {"success": True, "files": files_status}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def user_message(self, params: dict):
        return "Retrieving Git repository status..."


class GitDiffTool(Tool):
    """
    Shows content changes (diffs) for specified files or for all staged/unstaged changes.
    """

    name = "git_diff"
    description = "Shows content changes (diffs) for specified files or for all staged/unstaged changes in the Git repository."
    input_schema = {
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of file paths to diff. If empty, shows all changes based on 'staged' flag.",
            },
            "staged": {
                "type": "boolean",
                "description": "Set to true to show staged changes, false for unstaged. Defaults to false (unstaged).",
                "default": False,
            },
        },
    }

    def __init__(self, repo_path: str | None = None):
        super().__init__()
        self.repo_path = repo_path

    async def process(self, context: ProcessingContext, params: dict):
        try:
            files_to_diff = params.get("files", [])
            staged = params.get("staged", False)

            cmd_parts = ["git", "diff"]
            if staged:
                cmd_parts.append("--cached")

            if files_to_diff:
                cmd_parts.append("--")
                cmd_parts.extend(files_to_diff)

            command = shlex.join(cmd_parts)
            print(f"Running command: {command}")

            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.repo_path or context.workspace_dir,
                check=False,
            )
            print(f"Result: {result}")

            # git diff exits with 0 if no diff, 1 if diff. We don't treat 1 as an error here.
            if result.returncode > 1:
                return {
                    "success": False,
                    "error": f"Git diff command failed: {result.stderr}",
                }

            return {"success": True, "diff": result.stdout}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def user_message(self, params: dict):
        target = "staged changes" if params.get("staged") else "unstaged changes"
        if params.get("files"):
            target = f"files: {', '.join(params.get('files', []))}"
            if params.get("staged"):
                target += " (staged)"
            else:
                target += " (unstaged)"
        return f"Retrieving Git diff for {target}..."


class GitCommitTool(Tool):
    """
    Stages the specified files as a group and creates a Git commit with them and the given message.
    """

    name = "git_commit"
    description = "Creates a Git commit with the given message."
    input_schema = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The commit message.",
            },
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "A list of file paths (relative to workspace root) to include in this commit.",
            },
            "dry_run": {
                "type": "boolean",
                "description": "If true, performs a dry run of the commit without actually creating it. Defaults to false.",
                "default": False,
            },
        },
        "required": ["message", "files"],
    }

    def __init__(self, repo_path: str | None = None):
        super().__init__()
        self.repo_path = repo_path

    async def process(self, context: ProcessingContext, params: dict):
        try:
            message = params["message"]
            files_to_commit = params["files"]
            dry_run = params.get("dry_run", False)

            if not files_to_commit:
                return {"success": False, "error": "No files specified for commit."}

            # Stage the specified files
            add_cmd_parts = ["git", "add", "--"]
            add_cmd_parts.extend(files_to_commit)
            add_command = shlex.join(add_cmd_parts)

            add_result = subprocess.run(
                add_command,
                shell=False,
                capture_output=True,
                text=True,
                cwd=self.repo_path or context.workspace_dir,
                check=False,
            )

            if add_result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Git add command failed for files {files_to_commit}: {add_result.stderr}",
                }

            # Commit the staged files
            commit_cmd_parts = ["git", "commit", "-m", message]
            if dry_run:
                commit_cmd_parts.append("--dry-run")
            # No need to specify files again for commit if they are already staged by the add command above.
            # If we wanted to commit ONLY these files, we would add them here.
            # However, 'git commit -m message' will commit all currently staged changes.
            # This is fine as we just staged the files we care about.
            commit_command = shlex.join(commit_cmd_parts)

            commit_result = subprocess.run(
                commit_command,
                shell=False,
                capture_output=True,
                text=True,
                cwd=self.repo_path or context.workspace_dir,
                check=False,
            )

            if commit_result.returncode != 0:
                # Check if it's "nothing to commit" which might mean files were already committed or not really changed
                if (
                    "nothing to commit" in commit_result.stdout
                    or "no changes added to commit" in commit_result.stdout
                ):
                    return {
                        "success": True,
                        "message": "Commit successful (or no changes to commit for the specified files).",
                        "details": commit_result.stdout,
                    }
                return {
                    "success": False,
                    "error": f"Git commit command failed: {commit_result.stderr}",
                    "stdout": commit_result.stdout,
                }

            return {
                "success": True,
                "message": "Commit successful.",
                "details": commit_result.stdout,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def user_message(self, params: dict):
        num_files = len(params.get("files", []))
        message_param = params.get("message", "")
        dry_run = params.get("dry_run", False)
        message_preview = message_param[:30]
        if len(message_param) > 30:
            message_preview += "..."
        action = "Dry running commit" if dry_run else "Committing"
        return f"{action} {num_files} file(s) with message: '{message_preview}'"


if __name__ == "__main__":
    """
    Smoke tests for Git tools.
    """

    async def run_tests():
        context = ProcessingContext(workspace_dir=".")

        print("\n=== Testing GitStatusTool ===")
        status_tool = GitStatusTool()
        status_result = await status_tool.process(context, {})
        print(f"Status Result: {status_result}")

        print("\n=== Testing GitDiffTool ===")
        diff_tool = GitDiffTool()
        # Get diff for this file
        diff_result = await diff_tool.process(context, {})
        print(f"Diff Result Success: {diff_result['success']}")
        print(f"Diff Length: {len(diff_result.get('diff', ''))}")

        print("\n=== Testing GitCommitTool (Dry Run) ===")
        commit_tool = GitCommitTool()
        # Use dry_run to avoid actually creating a commit
        commit_result = await commit_tool.process(
            context,
            {
                "message": "Test commit (dry run)",
                "files": ["src/nodetool/agents/tools/git_tools.py"],
                "dry_run": True,
            },
        )
        print(f"Commit Result: {commit_result}")

        print("\nAll smoke tests completed.")

    # Run the async tests
    asyncio.run(run_tests())
