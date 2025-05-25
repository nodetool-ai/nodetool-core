import asyncio
import json
from typing import Dict

import websockets

from nodetool.types.workflow import Workflow


def workflow_to_dsl_code(workflow: Workflow, websocket_url: str = "ws://localhost:8000/chat") -> str:
    """Generate Python DSL code for a workflow that runs via the chat WebSocket.

    Parameters
    ----------
    workflow : Workflow
        The workflow to export.
    websocket_url : str, optional
        The chat WebSocket URL, by default "ws://localhost:8000/chat".

    Returns
    -------
    str
        A Python script as a string.
    """

    imports = {
        "base": [
            "import asyncio",
            "import json",
            "import websockets",
            "from nodetool.dsl.graph import graph",
        ],
        "nodes": set(),
    }

    node_var_by_id: Dict[str, str] = {}
    node_lines = []
    connect_lines = []

    for idx, node in enumerate(workflow.graph.nodes):
        var = f"n{idx}"
        node_var_by_id[node.id] = var
        parts = node.type.split(".")
        module = "nodetool.dsl." + ".".join(parts[:-1])
        cls = parts[-1]
        imports["nodes"].add(f"from {module} import {cls}")
        params = ", ".join(f"{k}={repr(v)}" for k, v in (node.data or {}).items())
        node_lines.append(f"{var} = {cls}({params})")

    for edge in workflow.graph.edges:
        src = node_var_by_id.get(edge.source)
        tgt = node_var_by_id.get(edge.target)
        if src and tgt:
            connect_lines.append(
                f"{tgt}.{edge.targetHandle} = ({src}, \"{edge.sourceHandle}\")"
            )

    graph_line = f"workflow_graph = graph({', '.join(node_var_by_id.values())})"

    run_lines = [
        "async def main():",
        f"    async with websockets.connect('{websocket_url}') as websocket:",
        "        message = {",
        "            'role': 'user',",
        "            'workflow_id': 'exported_workflow',",
        "            'graph': workflow_graph.model_dump(),",
        "        }",
        "        await websocket.send(json.dumps(message))",
        "        while True:",
        "            response = await websocket.recv()",
        "            print(response)",
        "",
        "asyncio.run(main())",
    ]

    lines = []
    lines.extend(imports["base"])
    lines.extend(sorted(imports["nodes"]))
    lines.append("")
    lines.extend(node_lines)
    if connect_lines:
        lines.append("")
        lines.extend(connect_lines)
    lines.append("")
    lines.append(graph_line)
    lines.append("")
    lines.extend(run_lines)

    return "\n".join(lines)
