from contextlib import AsyncExitStack
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.types import Tool
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from pydantic import BaseModel
from typing import Any, List, Type, Optional
from agent_output import ReviewOutput
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
        read_stream, write_stream = await self._streams_context.__aenter__()

        self._session_context = ClientSession(read_stream, write_stream)
        self.session: ClientSession = await self._session_context.__aenter__()

        await self.session.initialize()

        # List available tools from the MCP server
        response = await self.session.list_tools()
        print("\nConnected to server with tools:", [tool.name for tool in response.tools])

        self.messages = []

    async def process_query(self, query: str, toolsToLoad: List[str], outputType: Optional[Type[BaseModel]] = None, freshSession: bool = False) -> Any:
        if freshSession:
            self.messages.clear()

        self.messages.append({
            "role": "user",
            "content": query
        })

        toolResponse: Any = await self.session.list_tools()

        # Filter tools to keep only those in toolsToLoad
        filtered_tools = toolResponse.tools if not toolsToLoad else [tool for tool in toolResponse.tools if tool.name in toolsToLoad]
        available_tools = [self.__convert_tool_format(tool) for tool in filtered_tools]

        response = ""
        output_type_counter: int = 0
        if query and query.strip():
            response = await self.__chat_execute(available_tools, None)

        # Convert query response to output type specified
        # Will retry some time before giving up
        output_type_error = ""
        if outputType is not None:
            while True:
                try:
                    if output_type_error == "":
                        toTypeSchema = outputType.model_json_schema()
                        toTypeStr: str = f"""For the result produced previously, return the result as a JSON array where each element matches the following schema
                        {json.dumps(toTypeSchema, indent=2)}"""
                        print(toTypeStr)
                        self.messages.append({
                            "role": "user",
                            "content": toTypeStr
                        })
                    else:
                        toTypeStr: str = f"""Result return schema incorrect: {output_type_error}. Try again."""
                        self.messages.append({
                            "role": "user",
                            "content": toTypeStr
                        })
                    response = await self.__chat_execute(available_tools, outputType)
                    break
                except Exception as e:
                    output_type_counter+=1
                    output_type_error = f"Error: {str(e)}"
                    if output_type_counter == 3:
                        print(f"Exited due to output type retry counter reached")
                        raise
        
        return response

    async def chat_loop(self, toolsToLoad: List[str]):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower().startswith('testing'):
                    query = query[len("testing"):]
                    result = await self.process_query(query, toolsToLoad, ReviewOutput)
                    continue
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
        self.openai.close()

    async def __chat_execute(self, available_tools: dict[str, Any], outputType: Optional[Type[BaseModel]] = None) -> Any:
        continue_count: int = 0
        response: Any = None
        fullChat: str = ""
        while True:
            if continue_count == 5:
                print(f"Exited due to continue limit reached")
                break

            params = {
                "model": self.model,
                "tools": available_tools,
                "messages": self.messages
            }
            response = self.openai.chat.completions.create(**params)

            content = response.choices[0].message
            self.messages.append(response.choices[0].message.model_dump())

            if content.tool_calls is not None:
                await self.__tool_execute(content)
                continue
            else:
                fullChat += response.choices[0].message.content
                print(response.choices[0].message.content)

            finish_reason = getattr(response.choices[0], 'finish_reason', None)
            # If the chat is complete, exit the loop
            if finish_reason == "stop":
                break
            
            # If the chat is not complete, prompt to continue
            elif finish_reason == "length":
                self.messages.append({
                    "role": "user",
                    "content": "continue"
                })
                print("continue")
                continue_count+=1

            # Something broke, i.e. insufficient funds
            else:
                return fullChat

        return self.__get_response(fullChat, outputType)

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

    def __get_response(self, fullChat: str, outputType: Optional[Type[BaseModel]]) -> Any:
        if outputType is not None:
            dataList = json.loads(fullChat)
            typed = [outputType.model_validate(data) for data in dataList]
            return typed
                
        return fullChat