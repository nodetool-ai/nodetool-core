"""
MCP Tools Wrapper Module

This module provides a wrapper to expose all MCP (Model Context Protocol) tools
as Agent Tools, making them available to chat agents for controlling NodeTool.

The wrapper introspects the MCP server and dynamically creates Tool instances
for each MCP function, allowing agents to:
- Manage workflows (create, run, get status)
- Query nodes and search
- Handle assets and files
- Manage jobs and executions
- Query collections and databases
- And all other MCP server capabilities
"""

import inspect
from typing import Any

from nodetool.agents.tools.base import Tool
from nodetool.config.logging_config import get_logger
from nodetool.workflows.processing_context import ProcessingContext

log = get_logger(__name__)

# Lazy import to avoid circular dependencies and missing dependencies
_mcp_instance = None

# Cache for loaded MCP tools
# Note: This global cache is intentional for performance. MCP tools are immutable
# once loaded and the cache is safe for concurrent access (read-only after first load).
# In tests, the cache can be cleared by reimporting the module.
_mcp_tools_cache: dict[str, Tool] = {}


def _get_mcp_instance():
    """Get the MCP instance, importing lazily."""
    global _mcp_instance
    if _mcp_instance is None:
        try:
            from nodetool.api.mcp_server import mcp
            _mcp_instance = mcp
        except ImportError as e:
            log.error(f"Failed to import MCP server: {e}")
            raise
    return _mcp_instance


class MCPToolWrapper(Tool):
    """
    A dynamic wrapper that converts an MCP tool to an Agent Tool.
    
    This class wraps a single MCP tool function and provides the interface
    expected by the Agent system.
    """

    def __init__(self, mcp_tool_name: str, mcp_function: Any, tool_metadata: dict[str, Any]):
        """
        Initialize an MCP tool wrapper.
        
        Args:
            mcp_tool_name: The name of the MCP tool
            mcp_function: The actual MCP function to call
            tool_metadata: Metadata including description and input schema
        """
        self.mcp_tool_name = mcp_tool_name
        self.mcp_function = mcp_function
        self._tool_metadata = tool_metadata
        
        # Set Tool class attributes
        self.name = mcp_tool_name
        self.description = tool_metadata.get("description", f"MCP tool: {mcp_tool_name}")
        self.input_schema = tool_metadata.get("inputSchema", {})
        self.example = ""

    async def process(self, context: ProcessingContext, params: dict[str, Any]) -> Any:
        """
        Execute the MCP tool with the given parameters.
        
        Args:
            context: The processing context (may be used for user_id, etc.)
            params: Parameters to pass to the MCP function
            
        Returns:
            The result from the MCP function
        """
        try:
            log.debug(f"Executing MCP tool '{self.mcp_tool_name}' with params: {params}")
            
            # Call the MCP function
            # Note: Some MCP functions may require Context, we need to handle this
            result = await self.mcp_function(**params)
            
            log.debug(f"MCP tool '{self.mcp_tool_name}' completed successfully")
            return result
            
        except Exception as e:
            log.error(f"Error executing MCP tool '{self.mcp_tool_name}': {e}", exc_info=True)
            return {"error": str(e), "tool": self.mcp_tool_name}

    def user_message(self, params: dict[str, Any]) -> str:
        """
        Generate a user-friendly message describing the tool execution.
        
        Args:
            params: The parameters being used
            
        Returns:
            A human-readable message
        """
        param_summary = ", ".join(f"{k}={v}" for k, v in list(params.items())[:3])
        if len(params) > 3:
            param_summary += ", ..."
        return f"Calling MCP tool '{self.mcp_tool_name}' ({param_summary})"


def _extract_function_schema(func: Any) -> dict[str, Any]:
    """
    Extract JSON schema from function signature and docstring.
    
    Args:
        func: The function to extract schema from
        
    Returns:
        JSON schema dict for the function parameters
    """
    sig = inspect.signature(func)
    properties = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        # Skip special parameters
        if param_name in ("self", "ctx"):
            continue
            
        param_type = "string"  # Default type
        param_description = f"Parameter {param_name}"
        
        # Try to infer type from annotation
        if param.annotation != inspect.Parameter.empty:
            annotation = param.annotation
            # Check if annotation is a type object
            if annotation == str:
                param_type = "string"
            elif annotation == int:
                param_type = "integer"
            elif annotation == bool:
                param_type = "boolean"
            elif annotation == float:
                param_type = "number"
            elif hasattr(annotation, "__origin__"):
                # Handle typing generics like list[str], dict[str, Any]
                origin = getattr(annotation, "__origin__", None)
                if origin == list:
                    param_type = "array"
                elif origin == dict:
                    param_type = "object"
        
        properties[param_name] = {
            "type": param_type,
            "description": param_description
        }
        
        # Add to required if no default value
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
    
    schema = {
        "type": "object",
        "properties": properties,
    }
    
    if required:
        schema["required"] = required
        
    return schema


async def get_all_mcp_tools() -> list[Tool]:
    """
    Get all MCP tools as a list of Agent Tool instances.
    
    This function introspects the MCP server and creates Tool wrappers
    for each registered MCP tool. Results are cached for efficiency.
    
    Returns:
        List of Tool instances wrapping MCP functions
    """
    global _mcp_tools_cache
    
    # Return cached tools if available
    if _mcp_tools_cache:
        return list(_mcp_tools_cache.values())
    
    tools: list[Tool] = []
    
    try:
        mcp = _get_mcp_instance()
    except Exception as e:
        log.error(f"Failed to get MCP instance: {e}")
        return tools
    
    # Use the FastMCP get_tools() method to retrieve all registered tools
    # Returns a dict: {tool_name: FunctionTool}
    try:
        mcp_tools_dict = await mcp.get_tools()
        
        for tool_name, mcp_tool in mcp_tools_dict.items():
            try:
                # mcp_tool is a FunctionTool object from FastMCP with attributes:
                # - name: str
                # - description: str  
                # - fn: callable
                # - parameters: dict (JSON schema)
                func = mcp_tool.fn
                description = mcp_tool.description or ""
                # Get parameters from MCP tool (FastMCP uses 'parameters' for JSON schema)
                parameters = getattr(mcp_tool, "parameters", {})
                
                if func is None:
                    log.warning(f"MCP tool '{tool_name}' has no function, skipping")
                    continue
                
                # Extract description from docstring if not provided
                if not description and hasattr(func, "__doc__") and func.__doc__:
                    description = func.__doc__.strip().split("\n")[0]
                
                # Generate input schema if not provided
                if not parameters or not parameters.get("properties"):
                    parameters = _extract_function_schema(func)
                
                # Create tool metadata
                tool_metadata = {
                    "description": description,
                    "inputSchema": parameters,
                }
                
                # Create and add the wrapper
                wrapper = MCPToolWrapper(tool_name, func, tool_metadata)
                tools.append(wrapper)
                _mcp_tools_cache[tool_name] = wrapper
                log.debug(f"Registered MCP tool: {tool_name}")
                
            except Exception as e:
                log.error(f"Failed to wrap MCP tool: {e}", exc_info=True)
                
    except Exception as e:
        log.error(f"Failed to get tools from MCP server: {e}", exc_info=True)
        return tools
    
    log.info(f"Loaded {len(tools)} MCP tools for agent use")
    return tools


async def get_mcp_tool_by_name(name: str) -> Tool | None:
    """
    Get a specific MCP tool by name.
    
    Args:
        name: The name of the MCP tool to retrieve
        
    Returns:
        Tool instance or None if not found
    """
    if not _mcp_tools_cache:
        await get_all_mcp_tools()  # Populate cache
    
    return _mcp_tools_cache.get(name)
