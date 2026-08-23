import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from dotenv import load_dotenv

from ReviewScorer.clients.chat_client import ChatClient


def _message_response(message: str, finish_reason: str = "stop"):
    """Generate a response that vaguely looks like an OpenAI SDK response, for the fields we care about."""
    model_dump_func = lambda : SimpleNamespace(content = message)
    chat_completion_msg = SimpleNamespace(model_dump = model_dump_func, content = message, tool_calls = None)
    return SimpleNamespace(choices = [ SimpleNamespace(message = chat_completion_msg, finish_reason = finish_reason) ])


def _tool_response(name: str, args: list, finish_reason: str = "stop"):
    """Generate a response that vaguely looks like an OpenAI SDK response, for the fields we care about."""
    message = "generic-message-for-calling-tool"
    tool_function = SimpleNamespace(name = name, arguments = json.dumps(args))
    tool_calls = [ SimpleNamespace(function = tool_function) ]
    model_dump_func = lambda : SimpleNamespace(content = message)
    chat_completion_msg = SimpleNamespace(model_dump = model_dump_func, content = message, tool_calls = tool_calls)
    return SimpleNamespace(choices = [ SimpleNamespace(message = chat_completion_msg, finish_reason = finish_reason) ])


class MockOpenAiClient:
    """Very convoluted mock of the OpenAI client, because I don't know how to do this better."""
    def __init__(self, responses: list = []):
        chat_create_func = lambda model, tools, messages : self.__get_response()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create = chat_create_func))
        self.__index = -1
        self.__responses = responses

    def __get_response(self):
        # if the user provided responses when creating the mock, we return those, else we return a dummy response
        self.__index = self.__index + 1
        if self.__index < len(self.__responses):
            # if the user provided an exceptional response, raise the exception
            if isinstance(self.__responses[self.__index], Exception):
                raise self.__responses[self.__index]
            # else just return the response as-is
            else:
                return self.__responses[self.__index]
        else:
            return _message_response("dummy response")


@pytest.mark.asyncio
class TestChatClient:

    def __mock_openai_client(self, monkeypatch, responses: list|None = None):
        """Helper to monkey-patch our mock OpenAI client in"""
        if responses:
            patched_func = lambda a, b, c : MockOpenAiClient(responses)
        else:
            patched_func = lambda a, b, c : MockOpenAiClient()
        monkeypatch.setattr(ChatClient, "_build_openai_client", patched_func)

    async def test_process_simple_query(self, monkeypatch):
        """We can process a simple query, without throwing any errors."""
        self.__mock_openai_client(monkeypatch)
        chat_client = ChatClient('fake-url', 'fake-api-key', 'fake-model', [], None)

        await chat_client.process_query("ffs", [])

    async def test_message_appending(self, monkeypatch):
        """For each query we append two messages to our context, one each for query & response."""
        self.__mock_openai_client(monkeypatch)
        chat_client = ChatClient('fake-url', 'fake-api-key', 'fake-model', [], None)

        await chat_client.process_query("ffs", [])
        assert len(chat_client.get_messages()) == 2
        await chat_client.process_query("ffs", [])
        assert len(chat_client.get_messages()) == 4


    async def test_clear_history(self, monkeypatch):
        """Calling clear_history()...clears the history"""
        self.__mock_openai_client(monkeypatch)
        chat_client = ChatClient('fake-url', 'fake-api-key', 'fake-model', [], None)

        await chat_client.process_query("ffs", [])
        assert len(chat_client.get_messages()) == 2
        await chat_client.process_query("ffs", [])
        assert len(chat_client.get_messages()) == 4
        chat_client.clear_history()
        assert len(chat_client.get_messages()) == 0
        await chat_client.process_query("ffs", [])
        assert len(chat_client.get_messages()) == 2

    async def test_exceptional_response(self, monkeypatch):
        """If we get an exception from the OpenAI client, we just bubble it up."""
        self.__mock_openai_client(monkeypatch, responses=[RuntimeError("fake-runtime-error")])
        chat_client = ChatClient('fake-url', 'fake-api-key', 'fake-model', [], None)

        with pytest.raises(RuntimeError) as e_info:
            await chat_client.process_query("ffs", [])
        
        assert str(e_info.value) == "fake-runtime-error"

    async def test_process_tool_call(self, monkeypatch):
        """We can process a simple query, without throwing any errors."""
        fake_tool_call = AsyncMock()

        responses = [ _tool_response("my_tool", {"arg-1": "val-1", "arg-2": "val-2"}) ]
        self.__mock_openai_client(monkeypatch, responses)
        chat_client = ChatClient('fake-url', 'fake-api-key', 'fake-model', [], fake_tool_call)

        await chat_client.process_query("ffs", [])
        fake_tool_call.assert_awaited_once_with(
            "my_tool",
            {"arg-1": "val-1", "arg-2": "val-2"},
        )


@pytest.mark.skip(reason="These tests do not run at build time, but can be used for dev testing by commenting out this annotation.")
@pytest.mark.asyncio
class TestChatClientAgainstProd:


    @pytest.fixture
    def prod_chat_client(self):
        # load .env file and return the prod URL & creds
        load_dotenv(override=True)
        agent_api_url = os.getenv('AGENT_API_URL')
        agent_api_key = os.getenv('AGENT_API_KEY')
        agent_model = os.getenv('AGENT_MODEL')
        return ChatClient(agent_api_url, agent_api_key, agent_model, [], None)


    async def test_basic_prompt(self, prod_chat_client):
        """
        Basic method that just runs a single prompt against the LLM.
        """
        resp = await prod_chat_client.process_query("your face is silly", [])
        print(f"Response is '{resp}'")
