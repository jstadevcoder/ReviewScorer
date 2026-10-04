from dotenv import load_dotenv
from types import SimpleNamespace
from pydantic import BaseModel
from unittest.mock import AsyncMock, Mock, patch

import json
import os
import pytest

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
    tool_calls = [ SimpleNamespace(function = tool_function, id = 0) ]
    model_dump_func = lambda : SimpleNamespace(content = message)
    chat_completion_msg = SimpleNamespace(model_dump = model_dump_func, content = message, tool_calls = tool_calls)
    return SimpleNamespace(choices = [ SimpleNamespace(message = chat_completion_msg, finish_reason = finish_reason) ])

class Item(BaseModel):
    name: str
    count: int

def make_message(
    *,
    content=None,
    tool_calls=None,
    dumped=None,
):
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        model_dump=Mock(
            return_value=dumped
            or {
                "role": "assistant",
                "content": content,
            }
        ),
    )


def make_response(message, *, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=message,
                finish_reason=finish_reason,
            )
        ]
    )


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
            {"arg-1": "val-1", "arg-2": "val-2",},
            0
        )

    @pytest.fixture
    def openai_client(self):
        client = Mock()
        client.close = Mock()
        client.chat.completions.create = Mock()
        return client

    @pytest.fixture
    def tool_callback(self):
        return AsyncMock(
            return_value={
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "tool result",
            }
        )

    @pytest.fixture
    def client(self, openai_client, tool_callback):
        with patch(
            "ReviewScorer.clients.chat_client.OpenAI",
            return_value=openai_client,
        ) as openai:
            instance = ChatClient(
                url="https://example.test/v1",
                api_key="test-key",
                model="test-model",
                tools=[],
                tool_callback=tool_callback,
            )

        openai.assert_called_once_with(
            base_url="https://example.test/v1",
            api_key="test-key",
        )
        return instance

    @pytest.mark.asyncio
    async def test_process_query_returns_text_and_updates_history(
        self,
        client,
        openai_client,
    ):
        assistant_message = make_message(content="hello")
        openai_client.chat.completions.create.return_value = make_response(
            assistant_message
        )

        result = await client.process_query(
            query="Hi",
            available_tools=[],
        )

        assert result == "hello"
        assert client.get_messages() == [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "hello"},
        ]

        openai_client.chat.completions.create.assert_called_once_with(
            model="test-model",
            tools=[],
            messages=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "hello"},
            ],
        )

    @pytest.mark.asyncio
    async def test_empty_query_does_not_call_openai(
        self,
        client,
        openai_client,
    ):
        result = await client.process_query(
            query="   ",
            available_tools=[],
        )

        assert result == ""
        assert client.get_messages() == [
            {"role": "user", "content": "   "},
        ]
        openai_client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_clear_history_removes_all_messages(self, client):
        client.get_messages().extend(
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
            ]
        )

        client.clear_history()

        assert client.get_messages() == []

    @pytest.mark.asyncio
    async def test_cleanup_closes_openai_client(
        self,
        client,
        openai_client,
    ):
        await client.cleanup()

        openai_client.close.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_tool_call_is_executed_and_chat_continues(
        self,
        client,
        openai_client,
        tool_callback,
    ):
        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(
                name="get_weather",
                arguments='{"city": "Sydney"}',
            ),
        )

        tool_message = make_message(
            content=None,
            tool_calls=[tool_call],
            dumped={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Sydney"}',
                        },
                    }
                ],
            },
        )
        final_message = make_message(content="It is sunny.")

        openai_client.chat.completions.create.side_effect = [
            make_response(tool_message),
            make_response(final_message),
        ]

        result = await client.process_query(
            query="What is the weather?",
            available_tools=["get_weather"],
        )

        assert result == "It is sunny."

        tool_callback.assert_awaited_once_with(
            "get_weather",
            {"city": "Sydney"},
            "call-1",
        )

        assert client.get_messages() == [
            {"role": "user", "content": "What is the weather?"},
            tool_message.model_dump.return_value,
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "tool result",
            },
            final_message.model_dump.return_value,
        ]

    @pytest.mark.asyncio
    async def test_tool_call_with_empty_arguments_uses_empty_dict(
        self,
        client,
        openai_client,
        tool_callback,
    ):
        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(
                name="list_items",
                arguments="",
            ),
        )
        tool_message = make_message(
            content=None,
            tool_calls=[tool_call],
        )
        final_message = make_message(content="Done")

        openai_client.chat.completions.create.side_effect = [
            make_response(tool_message),
            make_response(final_message),
        ]

        result = await client.process_query(
            query="List items",
            available_tools=[],
        )

        assert result == "Done"
        tool_callback.assert_awaited_once_with(
            "list_items",
            {},
            "call-1",
        )

    @pytest.mark.asyncio
    async def test_length_response_prompts_model_to_continue(
        self,
        client,
        openai_client,
    ):
        first_message = make_message(content="first ")
        second_message = make_message(content="second")

        openai_client.chat.completions.create.side_effect = [
            make_response(first_message, finish_reason="length"),
            make_response(second_message, finish_reason="stop"),
        ]

        result = await client.process_query(
            query="Tell me something",
            available_tools=[],
        )

        assert result == "first second"
        assert client.get_messages()[-2:] == [
            {"role": "user", "content": "continue"},
            second_message.model_dump.return_value,
        ]
        assert openai_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_non_stop_non_length_finish_reason_returns_accumulated_text(
        self,
        client,
        openai_client,
    ):
        assistant_message = make_message(content="partial result")

        openai_client.chat.completions.create.return_value = make_response(
            assistant_message,
            finish_reason="content_filter",
        )

        result = await client.process_query(
            query="Answer",
            available_tools=[],
        )

        assert result == "partial result"
        openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_continue_limit_stops_after_five_continuations(
        self,
        client,
        openai_client,
    ):
        truncated_message = make_message(content="part ")

        openai_client.chat.completions.create.return_value = make_response(
            truncated_message,
            finish_reason="length",
        )

        result = await client.process_query(
            query="Answer",
            available_tools=[],
        )

        assert result == "part " * 5
        assert openai_client.chat.completions.create.call_count == 5

    @pytest.mark.asyncio
    async def test_structured_output_is_parsed_into_pydantic_models(
        self,
        client,
        openai_client,
    ):
        assistant_message = make_message(
            content=json.dumps(
                [
                    {"name": "one", "count": 1},
                    {"name": "two", "count": 2},
                ]
            )
        )
        openai_client.chat.completions.create.return_value = make_response(
            assistant_message
        )

        result = await client.process_query(
            query="Return items",
            available_tools=[],
            outputType=Item,
        )

        assert result == [
            Item(name="one", count=1),
            Item(name="two", count=2),
        ]

        assert client.get_messages()[1] == (
            assistant_message.model_dump.return_value
        )
        assert client.get_messages()[2]["role"] == "user"
        assert "schema" in client.get_messages()[2]["content"]

    @pytest.mark.asyncio
    async def test_structured_output_retries_after_invalid_json(
        self,
        client,
        openai_client,
    ):
        invalid_message = make_message(
            content="not valid json",
        )

        valid_message = make_message(
            content=json.dumps(
                [
                    {
                        "name": "valid",
                        "count": 3,
                    }
                ]
            ),
        )

        captured_requests = []

        def create_response(**kwargs):
            captured_requests.append(
                {
                    "model": kwargs["model"],
                    "tools": kwargs["tools"],
                    "messages": [
                        message.copy()
                        for message in kwargs["messages"]
                    ],
                }
            )

            if len(captured_requests) == 1:
                # Initial normal response.
                return make_response(
                    make_message(content="Initial response")
                )

            if len(captured_requests) == 2:
                # First structured-output attempt fails JSON parsing.
                return make_response(invalid_message)

            # Retry succeeds.
            return make_response(valid_message)

        openai_client.chat.completions.create.side_effect = (
            create_response
        )

        result = await client.process_query(
            query="Return an item",
            available_tools=[],
            outputType=Item,
        )

        assert result == [
            Item(
                name="valid",
                count=3,
            )
        ]

        assert len(captured_requests) == 3

        first_request = captured_requests[0]
        structured_request = captured_requests[1]
        retry_request = captured_requests[2]

        assert first_request["model"] == "test-model"
        assert first_request["tools"] == []
        assert first_request["messages"] == [
            {
                "role": "user",
                "content": "Return an item",
            },
        ]

        assert structured_request["model"] == "test-model"
        assert structured_request["tools"] == []

        assert any(
            message["role"] == "user"
            and "schema" in message["content"]
            for message in structured_request["messages"]
        )

        assert retry_request["model"] == "test-model"
        assert retry_request["tools"] == []

        assert any(
            message["role"] == "user"
            and message["content"].startswith(
                "Result return schema incorrect:"
            )
            for message in retry_request["messages"]
        )

    @pytest.mark.asyncio
    async def test_structured_output_retries_after_validation_error(
        self,
        client,
        openai_client,
    ):
        invalid_message = make_message(
            content=json.dumps(
                [
                    {"name": "missing count"},
                ]
            )
        )
        valid_message = make_message(
            content=json.dumps(
                [
                    {"name": "valid", "count": 4},
                ]
            )
        )

        openai_client.chat.completions.create.side_effect = [
            make_response(invalid_message),
            make_response(valid_message),
        ]

        result = await client.process_query(
            query="Return an item",
            available_tools=[],
            outputType=Item,
        )

        assert result == [Item(name="valid", count=4)]
        assert openai_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_structured_output_raises_after_three_failures(
        self,
        client,
        openai_client,
    ):
        invalid_message = make_message(content="invalid")
        openai_client.chat.completions.create.return_value = make_response(
            invalid_message
        )

        with pytest.raises(json.JSONDecodeError):
            await client.process_query(
                query="Return items",
                available_tools=[],
                outputType=Item,
            )

        assert openai_client.chat.completions.create.call_count == 4

    @pytest.mark.asyncio
    async def test_tool_callback_exception_is_propagated(
        self,
        client,
        openai_client,
        tool_callback,
    ):
        tool_callback.side_effect = RuntimeError("tool failed")

        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(
                name="broken_tool",
                arguments="{}",
            ),
        )
        tool_message = make_message(
            content=None,
            tool_calls=[tool_call],
        )
        openai_client.chat.completions.create.return_value = make_response(
            tool_message
        )

        with pytest.raises(RuntimeError, match="tool failed"):
            await client.process_query(
                query="Use the tool",
                available_tools=[],
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
