from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ReviewScorer.clients.mcp_client import McpClientStreamableHttp

class AsyncContextManagerMock:
    """Async context-manager mock for streamable_http_client()."""

    def __init__(self, enter_result):
        self.enter_result = enter_result
        self.enter_calls = 0
        self.exit_calls = 0

    async def __aenter__(self):
        self.enter_calls += 1
        return self.enter_result

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.exit_calls += 1
        return False

class TestMcpClient:
    @pytest.fixture
    def client(self):
        return McpClientStreamableHttp()

    @pytest.mark.asyncio
    async def test_connect_to_streamable_http_server(self, client):
        read_stream = Mock(name="read_stream")
        write_stream = Mock(name="write_stream")

        streams_context = AsyncContextManagerMock((read_stream, write_stream))

        session = SimpleNamespace(initialize=AsyncMock())
        session_context = AsyncContextManagerMock(session)

        with (
            patch(
                "ReviewScorer.clients.mcp_client.create_mcp_http_client",
                return_value=Mock(name="http_client"),
            ) as create_http_client,
            patch(
                "ReviewScorer.clients.mcp_client.streamable_http_client",
                return_value=streams_context,
            ) as streamable_http_client,
            patch(
                "ReviewScorer.clients.mcp_client.ClientSession",
                return_value=session_context,
            ) as client_session,
        ):
            await client.connect_to_streamable_http_server(
                "https://testingmcpclient.com/mcp",
                headers={"Authorization": "Bearer token"},
            )

        create_http_client.assert_called_once_with(
            headers={"Authorization": "Bearer token"}
        )

        streamable_http_client.assert_called_once_with(
            url="https://testingmcpclient.com/mcp",
            http_client=create_http_client.return_value,
        )

        client_session.assert_called_once_with(read_stream, write_stream)

        assert session_context.enter_calls == 1
        session.initialize.assert_awaited_once()

        assert client.session is session
        assert client._streams_context is streams_context

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, client):
        client.session = SimpleNamespace(
            call_tool=AsyncMock(
                return_value=SimpleNamespace(
                    content=[{"type": "text", "text": "tool result"}]
                )
            )
        )

        result = await client.execute_tool(
            tool_name="testmcp",
            tool_args={"query": "MCP"},
            tool_call_id="call-123",
        )

        client.session.call_tool.assert_awaited_once_with(
            "testmcp",
            {"query": "MCP"},
        )

        assert result == {
            "role": "tool",
            "tool_call_id": "call-123",
            "name": "testmcp",
            "content": [{"type": "text", "text": "tool result"}],
        }

    @pytest.mark.asyncio
    async def test_execute_tool_error(self, client):
        error = RuntimeError("tool failed")

        client.session = SimpleNamespace(
            call_tool=AsyncMock(side_effect=error)
        )

        result = await client.execute_tool(
            tool_name="testmcp",
            tool_args={"query": "MCP"},
            tool_call_id="call-456",
        )

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call-456"
        assert result["name"] == "testmcp"
        assert result["content"] == {error}

    @pytest.mark.asyncio
    async def test_list_tools(self, client):
        tool_one = SimpleNamespace(
            name="testmcp",
            description="Testing mcp",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        )

        tool_two = SimpleNamespace(
            name="get_test",
            description="Get test information",
            input_schema={
                "type": "object",
                "properties": {
                    "number": {"type": "string"},
                    "units": {"type": "string"},
                },
                "required": ["number"],
            },
        )

        client.session = SimpleNamespace(
            list_tools=AsyncMock(
                return_value=SimpleNamespace(tools=[tool_one, tool_two])
            )
        )

        result = await client.list_tools()

        client.session.list_tools.assert_awaited_once()

        assert result == [
            {
                "type": "function",
                "function": {
                    "name": "testmcp",
                    "description": "Testing mcp",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_test",
                    "description": "Get test information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "number": {"type": "string"},
                            "units": {"type": "string"},
                        },
                        "required": ["number"],
                    },
                },
            },
        ]

    def test_convert_tool_format_without_required_fields(self, client):
        tool = SimpleNamespace(
            name="testmcp",
            description=None,
            input_schema={
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                },
            },
        )

        result = client._McpClientStreamableHttp__convert_tool_format(tool)

        assert result == {
            "type": "function",
            "function": {
                "name": "testmcp",
                "description": None,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                    },
                    "required": [],
                },
            },
        }

    @pytest.mark.asyncio
    async def test_cleanup_closes_session_and_streams(self, client):
        session_context = AsyncContextManagerMock((None, None))
        streams_context = AsyncContextManagerMock((None, None))

        client._session_context = session_context
        client._streams_context = streams_context

        await client.cleanup()

        assert session_context.exit_calls == 1
        assert streams_context.exit_calls == 1

    @pytest.mark.asyncio
    async def test_cleanup_when_contexts_are_missing(self, client):
        client._session_context = None
        client._streams_context = None

        await client.cleanup()
