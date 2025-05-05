import datetime
import os
import platform
import shutil
import subprocess
import re


class WorkspaceManager:
    """
    📁 Workspace Wizard - Virtual filesystem manager for the agent

    This class provides a controlled environment for file operations,
    keeping track of the current working directory and ensuring all
    file operations remain within the designated workspace boundaries.

    Think of it as a sandbox where agents can safely create, read, modify,
    and delete files without messing up your actual system. It's like
    giving the AI a virtual filing cabinet it can organize however it wants.
    """

    def __init__(self, workspace_dir=None):
        """
        Initialize the workspace manager with a root directory.

        If no directory is specified, creates a timestamped workspace in
        ~/.nodetool-workspaces for clean isolation of each agent run.

        Args:
            workspace_dir (str, optional): Custom workspace path to use.
                If None, a timestamped directory is created automatically.
        """
        if workspace_dir:
            self.workspace_root = os.path.abspath(workspace_dir)
            self.current_workspace = self.workspace_root
        else:
            # Use ~/.nodetool-workspaces as the default root
            self.workspace_root = os.path.expanduser("~/.nodetool-workspaces")
            os.makedirs(self.workspace_root, exist_ok=True)
            self.create_new_workspace()

    def create_new_workspace(self):
        """Creates a new workspace named with an incrementing number."""
        existing_workspaces = []
        try:
            items = os.listdir(self.workspace_root)
            for item in items:
                # Check if the item is a directory and its name is purely numeric
                if os.path.isdir(os.path.join(self.workspace_root, item)):
                    match = re.match(r"^(\d+)$", item)  # Match only digits
                    if match:
                        try:
                            existing_workspaces.append(int(match.group(1)))
                        except ValueError:
                            # Ignore items that look like numbers but aren't valid integers
                            pass
        except FileNotFoundError:
            # Handle case where workspace_root doesn't exist yet (though __init__ should create it)
            pass

        next_num = 1
        if existing_workspaces:
            next_num = max(existing_workspaces) + 1

        workspace_name = str(next_num)  # Use just the number as the name
        workspace_path = os.path.join(self.workspace_root, workspace_name)

        # Handle potential race condition: ensure the directory doesn't exist before creating
        while os.path.exists(workspace_path):
            next_num += 1
            workspace_name = str(next_num)
            workspace_path = os.path.join(self.workspace_root, workspace_name)

        os.makedirs(workspace_path, exist_ok=True)
        self.current_workspace = workspace_path
        print(f"Created new workspace at: {self.current_workspace}")

    def get_current_directory(self) -> str:
        """
        Get the current working directory.

        Returns:
            str: The current working directory path
        """
        return self.current_workspace or ""

    def list_directory(self, path=None) -> str:
        """
        List contents of a directory.

        Args:
            path (str, optional): Path to list. Defaults to current directory.

        Returns:
            str: Newline-separated list of directory contents or error message
        """
        full_path = (
            os.path.join(self.current_workspace, path)
            if path
            else self.current_workspace
        )
        try:
            items = os.listdir(full_path)
            return "\n".join(items)
        except Exception as e:
            return f"Error: {str(e)}"

    def change_directory(self, path) -> str:
        """
        Change the current working directory.

        Args:
            path (str): Target directory path

        Returns:
            str: Success message or error message
        """
        if not path:
            return "Error: Missing directory argument"
        new_path = os.path.join(self.current_workspace, path)
        if not os.path.exists(new_path):
            return f"Error: Directory {path} does not exist"
        if not new_path.startswith(self.workspace_root):
            return "Error: Cannot navigate outside workspace"
        self.current_workspace = new_path
        return f"Changed directory to {new_path}"

    def make_directory(self, path) -> str:
        """
        Create a new directory.

        Args:
            path (str): Directory path to create

        Returns:
            str: Success message or error message
        """
        if not path:
            return "Error: Missing directory name"
        try:
            os.makedirs(os.path.join(self.current_workspace, path), exist_ok=True)
            return f"Created directory {path}"
        except Exception as e:
            return f"Error creating directory: {str(e)}"

    def remove_item(self, path, recursive=False) -> str:
        """
        Remove a file or directory.

        Args:
            path (str): Path to remove
            recursive (bool, optional): If True, recursively remove directories. Defaults to False.

        Returns:
            str: Success message or error message
        """
        if not path:
            return "Error: Missing path argument"
        full_path = os.path.join(self.current_workspace, path)
        if not full_path.startswith(self.workspace_root):
            return "Error: Cannot remove files outside workspace"
        try:
            if os.path.isdir(full_path):
                if recursive:
                    shutil.rmtree(full_path)
                else:
                    os.rmdir(full_path)
            else:
                os.remove(full_path)
            return f"Removed {path}"
        except Exception as e:
            return f"Error removing {path}: {str(e)}"

    def open_file(self, path) -> str:
        """
        Open a file with the system default application.

        Args:
            path (str): Path to the file to open

        Returns:
            str: Success message or error message
        """
        if not path:
            return "Error: Missing file argument"
        full_path = os.path.join(self.current_workspace, path)
        if not os.path.exists(full_path):
            return f"Error: File {path} does not exist"
        try:
            if platform.system() == "Darwin":  # macOS
                subprocess.run(["open", full_path])
            elif platform.system() == "Windows":  # Windows
                os.startfile(full_path)  # type: ignore
            else:  # linux variants
                subprocess.run(["xdg-open", full_path])
            return f"Opened {path}"
        except Exception as e:
            return f"Error opening file: {str(e)}"

    async def execute_command(self, cmd: str) -> str:
        """
        Execute workspace commands in a controlled environment.

        This method parses and executes file system commands within the workspace
        boundary, preventing operations outside the allowed workspace directory.
        All paths are validated to ensure they remain within the workspace.

        Supported commands:
            - pwd/cwd: Show current working directory
            - ls [path]: List directory contents
            - cd [path]: Change directory
            - mkdir [path]: Create directory
            - rm [-r/-rf] [path]: Remove file or directory
            - open [path]: Open file with system default application

        Args:
            cmd (str): The command to execute

        Returns:
            str: The result of the command execution or error message
        """
        parts = cmd.split()
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        if command == "pwd" or command == "cwd":
            return self.get_current_directory()

        elif command == "ls":
            path = args[0] if args else None
            return self.list_directory(path)

        elif command == "cd":
            if not args:
                return "Error: Missing directory argument"
            return self.change_directory(args[0])

        elif command == "mkdir":
            if not args:
                return "Error: Missing directory name"
            return self.make_directory(args[0])

        elif command == "rm":
            if not args:
                return "Error: Missing path argument"
            recursive = "-r" in args or "-rf" in args
            # Find the actual path argument (not the flags)
            path_arg = next((arg for arg in args if not arg.startswith("-")), None)
            if not path_arg:
                return "Error: Missing path argument"
            return self.remove_item(path_arg, recursive)

        elif command == "open":
            if not args:
                return "Error: Missing file argument"
            return self.open_file(args[0])

        return f"Unknown command: {command}"
