from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.types import Tool

import structlog

logger = structlog.get_logger(__name__)

class McpClientStreamableHttp:
    """
    Generic client for connecting to an MCP server over HTTP(S). It allows us to list the available
    tools, as well as execute them.
    """

    def __init__(self):
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_streamable_http_server(
        self, server_url: str, headers: dict | None = None
    ):
        """Connect to an MCP server running with HTTP Streamable transport"""
        httpClient = create_mcp_http_client(headers=headers)
        self._streams_context = streamable_http_client(
            url=server_url,
            http_client=httpClient or {},
        )
        read_stream, write_stream = await self._streams_context.__aenter__()

        self._session_context = ClientSession(read_stream, write_stream)
        self.session: ClientSession = await self._session_context.__aenter__()

        await self.session.initialize()

    async def execute_tool(self, tool_name: str, tool_args: dict, tool_call_id) -> Any:
        try:
            logger.info(f"[Calling tool {tool_name} with args {tool_args}]")
            # Execute tool call
            result = await self.session.call_tool(tool_name, tool_args)
        except Exception as e:
            logger.exception(f"Error calling tool {tool_name}", error=str(e))
            result = None
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": {e}
            }

        logger.info(f"[Calling tool {tool_name} returned with {result.content}]")
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result.content
        }

    async def list_tools(self):
        tools_response = await self.session.list_tools()
        return [ self.__convert_tool_format(t) for t in tools_response.tools ]

    async def cleanup(self):
        """Properly clean up the session and streams"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)
    
    def __convert_tool_format(self, tool: Tool) -> dict[str, Any]:
        properties = tool.input_schema.get("properties", {})
        required = tool.input_schema.get("required", [])
        
        # Compose the base parameters schema
        parameters_schema = {
            "type": "object",
            "properties": properties,
            "required": required
        }
        
        converted_tool = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters_schema
            }
        }
        
        return converted_tool
