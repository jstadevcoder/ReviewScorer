import asyncio
import os
from dotenv import load_dotenv
from mcp_client import McpClientStreamableHttp
#from openai import OpenAI
#from agents import Agent, Runner, trace
#from agents.mcp import MCPServerStreamableHttp

"""
async def getGitMcpPr(question, params, model):
    async with MCPServerStreamableHttp(name="GithubMcpServer", params=params) as server:
        agent = Agent(
            name="GithubMcpServerAgent",
            instructions="Use Github MCP server to answer the question.",
            mcp_servers=[server],
            model=model
        )
        result = await Runner.run(agent, question)
        return result
"""

async def main() -> None:
    load_dotenv(override=True)
    agent_api_url = os.getenv('AGENT_API_URL')
    agent_api_key = os.getenv('AGENT_API_KEY')
    agent_model = os.getenv('AGENT_MODEL')
    github_mcp_url = os.getenv('GITHUB_MCP_URL')
    github_api_key = os.getenv('GITHUB_API_KEY')
    print(f"Agent Url exists {agent_api_url}")
    print(f"Agent API Key exists and begins {agent_api_key[:8]}")
    print(f"Agent model exists {agent_model}")
    print(f"Github MCP Url exists {github_mcp_url}")
    print(f"Github API Key exists and begins {github_api_key[:10]}")

    prUrl = "https://github.com/sqlitebrowser/sqlitebrowser/pull/4147"

    """
    githubMcpParams = params = {
        "url": github_mcp_url,
        "headers": {
            "Authorization": "Bearer " + github_api_key
        }
    }
    result = asyncio.run(getGitMcpPr("What tools are there in the github mcp server?", githubMcpParams, agent_model))
    print(result)
    """
    
    """
    agent = OpenAI(base_url=agent_api_url, api_key=agent_api_key)
    messages = [{"role": "user", "content": "Tell me a fun fact"}]
    response = agent.chat.completions.create(model=agent_model, messages=messages)
    print(response.choices[0].message.content)
    """

    client = McpClientStreamableHttp(agent_api_url, agent_api_key, agent_model)
    try:
        await client.connect_to_streamable_http_server(github_mcp_url, {"Authorization": "Bearer " + github_api_key})
        await client.chat_loop([])
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
