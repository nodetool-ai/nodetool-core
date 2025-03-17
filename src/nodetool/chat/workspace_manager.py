import datetime
import os
import platform
import shutil
import subprocess


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
            self.workspace_root = workspace_dir
            self.current_workspace = workspace_dir
        else:
            # Use ~/.nodetool-workspaces as the default root
            self.workspace_root = os.path.expanduser("~/.nodetool-workspaces")
            os.makedirs(self.workspace_root, exist_ok=True)
            self.create_new_workspace()

    def create_new_workspace(self):
        """Creates a new workspace with a unique name"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        workspace_name = f"workspace_{timestamp}"
        workspace_path = os.path.join(self.workspace_root, workspace_name)
        os.makedirs(workspace_path, exist_ok=True)
        self.current_workspace = workspace_path
        print(f"Created new workspace at: {self.current_workspace}")

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
            return self.current_workspace or ""

        elif command == "ls":
            path = (
                os.path.join(self.current_workspace, *args)
                if args
                else self.current_workspace
            )
            try:
                items = os.listdir(path)
                return "\n".join(items)
            except Exception as e:
                return f"Error: {str(e)}"

        elif command == "cd":
            if not args:
                return "Error: Missing directory argument"
            new_path = os.path.join(self.current_workspace, args[0])
            if not os.path.exists(new_path):
                return f"Error: Directory {args[0]} does not exist"
            if not new_path.startswith(self.workspace_root):
                return "Error: Cannot navigate outside workspace"
            self.current_workspace = new_path
            return f"Changed directory to {new_path}"

        elif command == "mkdir":
            if not args:
                return "Error: Missing directory name"
            try:
                os.makedirs(
                    os.path.join(self.current_workspace, args[0]), exist_ok=True
                )
                return f"Created directory {args[0]}"
            except Exception as e:
                return f"Error creating directory: {str(e)}"

        elif command == "rm":
            if not args:
                return "Error: Missing path argument"
            path = os.path.join(self.current_workspace, args[0])
            if not path.startswith(self.workspace_root):
                return "Error: Cannot remove files outside workspace"
            try:
                if os.path.isdir(path):
                    if "-r" in args or "-rf" in args:
                        shutil.rmtree(path)
                    else:
                        os.rmdir(path)
                else:
                    os.remove(path)
                return f"Removed {args[0]}"
            except Exception as e:
                return f"Error removing {args[0]}: {str(e)}"

        elif command == "open":
            if not args:
                return "Error: Missing file argument"
            path = os.path.join(self.current_workspace, args[0])
            if not os.path.exists(path):
                return f"Error: File {args[0]} does not exist"
            try:
                if platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", path])
                elif platform.system() == "Windows":  # Windows
                    os.startfile(path)  # type: ignore
                else:  # linux variants
                    subprocess.run(["xdg-open", path])
                return f"Opened {args[0]}"
            except Exception as e:
                return f"Error opening file: {str(e)}"

        return f"Unknown command: {command}"
