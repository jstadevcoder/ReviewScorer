from contextlib import AsyncExitStack
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.types import Tool
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from typing import Optional
import json


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

    async def process_query(self, query: str, toolsToLoad: list[str], strongType: type = None, freshSession: bool = False) -> any:
        if freshSession:
            self.messages.clear()

        self.messages.append({
            "role": "user",
            "content": query
        })

        response = await self.session.list_tools()

        # Filter tools to keep only those in toolsToLoad
        filtered_tools = response.tools if not toolsToLoad else [tool for tool in response.tools if tool.name in toolsToLoad]
        available_tools = [self.__convert_tool_format(tool) for tool in filtered_tools]

        return await self.__chat_execute(available_tools, strongType)

    async def chat_loop(self, toolsToLoad: list[str]):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break
                if query.lower() == 'clear':
                    self.messages.clear()
                    continue
                result = await self.process_query(query, toolsToLoad)
                print("Result:")
                print(result)

            except Exception as e:
                print(f"Error: {str(e)}")

    async def cleanup(self):
        """Properly clean up the session and streams"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)

    async def __chat_execute(self, available_tools: dict[str, any], strongType: type = None) -> any:
        continue_count: int = 0
        response: any = None
        while True:
            if continue_count == 5:
                print(f"Exited due to continue limit reached")
                break

            params = {
                "model": self.model,
                "tools": available_tools,
                "messages": self.messages,
            }
            if strongType is not None:
                params["output_type"] = strongType
            response = self.openai.chat.completions.create(**params)

            content = response.choices[0].message
            self.messages.append(response.choices[0].message.model_dump())

            if content.tool_calls is not None:
                await self.__tool_execute(content)
                continue
            else:
                print(response.choices[0].message.content)

            finish_reason = getattr(response.choices[0], 'finish_reason', None)
            # If the chat is complete, exit the loop
            if finish_reason == "stop":
                break
            
            # If the chat is not complete, prompt to continue
            else:
                self.messages.append({
                    "role": "user",
                    "content": "continue"
                })
                print("continue")
                continue_count+=1
                
        return response.choices[0].message.content

    async def __tool_execute(self, content: ChatCompletionMessage) -> None:
        tool_name = content.tool_calls[0].function.name
        tool_args = content.tool_calls[0].function.arguments
        tool_args = json.loads(tool_args) if tool_args else {}

        try:
            print(f"[Calling tool {tool_name} with args {tool_args}]")
            # Execute tool call
            result = await self.session.call_tool(tool_name, tool_args)
        except Exception as e:
            print(f"Error calling tool {tool_name}: {e}")
            result = None
            self.messages.append({
                "role": "tool",
                "tool_call_id": content.tool_calls[0].id,
                "name": tool_name,
                "content": {e}
            })
            print(f"Error calling tool {tool_name}:\n{e}")
            return

        self.messages.append({
            "role": "tool",
            "tool_call_id": content.tool_calls[0].id,
            "name": tool_name,
            "content": result.content
        })
        print(f"[Calling tool {tool_name} returned with {result.content}]")
    
    def __convert_tool_format(self, tool: Tool) -> dict[str, any]:
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