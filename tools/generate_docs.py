import os
from typing import List

import nodetool.cli
import nodetool.workflows
from nodetool.packages.gen_docs import process_module


def generate_cli_docs(output_path: str) -> None:
    """Generate documentation for the nodetool CLI."""
    lines: List[str] = ["# nodetool CLI", ""]
    lines.append("Available commands:")
    lines.append("")
    for cmd_name, cmd in nodetool.cli.cli.commands.items():
        lines.append(f"- **{cmd_name}**: {cmd.help}")
        if hasattr(cmd, "commands"):
            for sub_name, sub_cmd in cmd.commands.items():
                lines.append(f"  - **{cmd_name} {sub_name}**: {sub_cmd.help}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_workflow_docs(output_dir: str) -> None:
    """Generate documentation for the workflow modules."""
    os.makedirs(output_dir, exist_ok=True)
    process_module(nodetool.workflows, output_dir, "nodetool")


if __name__ == "__main__":
    generate_cli_docs(os.path.join("docs", "cli.md"))
    generate_workflow_docs(os.path.join("docs", "api-reference"))
    print("Documentation generated.")
