from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from openai import OpenAI
import json
from mcp.shared._httpx_utils import (
    create_mcp_http_client,
)

class McpClientStreamableHttp:
    def __init__(self, agentUrl: str, agentKey: str, model: str):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI(
            base_url=agentUrl,
            api_key=agentKey
        )
        self.model = model

    async def connect_to_streamable_http_server(
        self, server_url: str, headers: Optional[dict] = None
    ):
        """Connect to an MCP server running with HTTP Streamable transport"""
        httpClient = create_mcp_http_client(headers=headers)
        self._streams_context = streamable_http_client(
            url=server_url,
            http_client=httpClient or {},
        )
        read_stream, write_stream, _ = await self._streams_context.__aenter__()

        self._session_context = ClientSession(read_stream, write_stream)
        self.session: ClientSession = await self._session_context.__aenter__()

        await self.session.initialize()

        # List available tools from the MCP server
        response = await self.session.list_tools()
        print("\nConnected to server with tools:", [tool.name for tool in response.tools])

        self.messages = []

    async def process_query(self, query: str, toolsToLoad: list[str]) -> str:

        self.messages.append({
            "role": "user",
            "content": query
        })

        response = await self.session.list_tools()

        # Filter tools to keep only those in toolsToLoad
        filtered_tools = response.tools if not toolsToLoad else [tool for tool in response.tools if tool.name in toolsToLoad]
        available_tools = [self.convert_tool_format(tool) for tool in filtered_tools]

        response = self.openai.chat.completions.create(
            model=self.model,
            tools=available_tools,
            messages=self.messages
        )
        self.messages.append(response.choices[0].message.model_dump())

        final_text = []
        content = response.choices[0].message
        if content.tool_calls is not None:
            tool_name = content.tool_calls[0].function.name
            tool_args = content.tool_calls[0].function.arguments
            tool_args = json.loads(tool_args) if tool_args else {}

            # Execute tool call
            try:
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
            except Exception as e:
                print(f"Error calling tool {tool_name}: {e}")
                result = None

            self.messages.append({
                "role": "tool",
                "tool_call_id": content.tool_calls[0].id,
                "name": tool_name,
                "content": result.content
            })

            response = self.openai.chat.completions.create(
                model=self.model,
                max_tokens=1000,
                messages=self.messages,
            )

            final_text.append(response.choices[0].message.content)
        else:
            final_text.append(content.content)

        return "\n".join(final_text)

    async def chat_loop(self, toolsToLoad: list[str]):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break
                result = await self.process_query(query, toolsToLoad)
                print("Result:")
                print(result)

            except Exception as e:
                print(f"Error: {str(e)}")
    
    def convert_tool_format(self, tool):
        properties = tool.inputSchema.get("properties", {})
        required = tool.inputSchema.get("required", [])
        
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

    async def cleanup(self):
        """Properly clean up the session and streams"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:  # pylint: disable=W0125
            await self._streams_context.__aexit__(None, None, None)