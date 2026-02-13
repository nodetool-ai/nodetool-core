import os

from nodetool.api.server import create_app

static_folder = os.getenv("STATIC_FOLDER", "web/dist")
enable_mcp = os.getenv("NODETOOL_ENABLE_MCP", "0") == "1"

app = create_app(static_folder=static_folder, enable_mcp=enable_mcp)
