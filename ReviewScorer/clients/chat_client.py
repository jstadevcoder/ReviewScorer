import json
from collections.abc import Callable
from typing import Any

from openai import OpenAI
from pydantic import BaseModel


class ChatClient:
    """
    Client for prompting an LLM with generic "chat" messages. This client is stateful; it maintains a
    context window of historical chat messages that is used & built upon with each message.
    """

    def __init__(self, url: str, api_key: str, model: str, tools: list, tool_callback: Callable[[str, dict], Any]):
        self.__url = url
        self.__api_key = api_key
        self.__model = model
        self.__tools = tools
        self.__tool_callback = tool_callback
        self.__openai = self._build_openai_client(url, api_key)
        self.__messages = []


    async def process_query(self, query: str, available_tools: list[str], outputType: type[BaseModel] | None = None) -> Any:

        self.__messages.append({
            "role": "user",
            "content": query
        })

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
                        self.__messages.append({
                            "role": "user",
                            "content": toTypeStr
                        })
                    else:
                        toTypeStr: str = f"""Result return schema incorrect: {output_type_error}. Try again."""
                        self.__messages.append({
                            "role": "user",
                            "content": toTypeStr
                        })
                    response = await self.__chat_execute(available_tools, outputType)
                    break
                except Exception as e:
                    output_type_counter+=1
                    output_type_error = f"Error: {e!s}"
                    if output_type_counter == 3:
                        print("Exited due to output type retry counter reached")
                        raise
        
        return response

    def clear_history(self):
        self.__messages.clear()

    
    def get_messages(self):
        return self.__messages


    def cleanup(self):
        self.__openai.close()


    def _build_openai_client(self, url, api_key):
        return OpenAI(base_url=url, api_key=api_key)

    async def __chat_execute(self, available_tools: dict[str, Any], outputType: type[BaseModel] | None = None) -> Any:
        continue_count: int = 0
        response: Any = None
        fullChat: str = ""
        while True:
            if continue_count == 5:
                print("Exited due to continue limit reached")
                break

            params = {
                "model": self.__model,
                "tools": available_tools,
                "messages": self.__messages
            }
            response = self.__openai.chat.completions.create(**params)

            content = response.choices[0].message
            self.__messages.append(response.choices[0].message.model_dump())

            if content.tool_calls is not None:
                tool_name = content.tool_calls[0].function.name
                tool_args = content.tool_calls[0].function.arguments
                tool_args = json.loads(tool_args) if tool_args else {}
                await self.__tool_callback(tool_name, tool_args)
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
                self.__messages.append({
                    "role": "user",
                    "content": "continue"
                })
                print("continue")
                continue_count+=1

            # Something broke, i.e. insufficient funds
            else:
                return fullChat

        return self.__get_response(fullChat, outputType)

    def __get_response(self, fullChat: str, outputType: type[BaseModel] | None) -> Any:
        if outputType is not None:
            dataList = json.loads(fullChat)
            typed = [outputType.model_validate(data) for data in dataList]
            return typed
                
        return fullChat


