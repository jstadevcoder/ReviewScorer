from typing import Optional, List
from contextlib import AsyncExitStack
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from openai import OpenAI
import json
from mcp.shared._httpx_utils import (
    create_mcp_http_client,
)

class ReviewScorer:
    gitMcpUrl = "https://api.githubcopilot.com/mcp"
    def __init__(self, agentUrl: str, agentKey: str, agentModel: str, gitKey: str, prUrl: List[str]):
        self.agentUrl = agentUrl
        self.agentKey = agentKey
        self.agentModel = agentModel
        self.gitKey = gitKey
        self.prUrl = prUrl

    async def get_score(
        self, server_url: str, headers: Optional[dict] = None
    ) -> str:
        "Connect to an MCP server running with HTTP Streamable transport"
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