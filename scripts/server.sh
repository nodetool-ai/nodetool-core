#!/bin/bash
# Start NodeTool server (macOS/Linux)
#
exec conda run --live-stream -n nodetool nodetool serve --reload
