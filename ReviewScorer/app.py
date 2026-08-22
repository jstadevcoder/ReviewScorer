import asyncio
import os
from dotenv import load_dotenv
from clients.mcp_client import McpClientStreamableHttp
from review_scorer import ReviewScorer
from rag_agent import RAGAgent
import numpy
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
    embedding_model = os.getenv('EMBEDDING_MODEL')
    embedding_dimension_str = os.getenv('EMBEDDING_DIMENSION')
    github_mcp_url = os.getenv('GITHUB_MCP_URL')
    github_api_key = os.getenv('GITHUB_API_KEY')
    print(f"Agent Url exists {agent_api_url}")
    print(f"Agent API Key exists and begins {agent_api_key[:8]}")
    print(f"Agent model exists {agent_model}")
    print(f"Github MCP Url exists {github_mcp_url}")
    print(f"Github API Key exists and begins {github_api_key[:10]}")

    if embedding_dimension_str is not None:
        try:
            embedding_dimension = int(embedding_dimension_str)
        except ValueError:
            embedding_dimension = 0
    else:
        test_int = 0

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

    rag = RAGAgent(agent_api_url, agent_api_key, embedding_model, embedding_dimension)
    testingRag = ["This is a test",
                  "The quick brown fox jumped over the lazy dog",
                  "Does this belong here",
                  "Is this correct",
                  "Are you sure doing this here is not going to cause a conflict later on",
                  "LGTM",
                  "Some gibberish",
                  "Woof woof",
                  "This if statement needs to be inverted",
                  "This recursive function won't terminate if we throw an exception",
                  "Please add class comments here to explaining what this is for",
                  "Please follow coding standards"]
    endingEmbeddings:numpy.ndarray = await rag.get_embeddings(testingRag)
    rag.add_to_index(endingEmbeddings, testingRag)
    testingRag = ["The error handling here does not catch all exception types"]
    endingEmbeddings = await rag.get_embeddings(testingRag)
    testSimilar = rag.retrieve_similar(endingEmbeddings)
    await rag.cleanup()

    scorer = ReviewScorer(agent_api_url, agent_api_key, agent_model, embedding_model, embedding_dimension)
    try:
        await scorer.connect_to_mcp_client(github_mcp_url, github_api_key)
        await scorer.scorePrComment("https://github.com/jstadevcoder/ReviewScorer/pull/1")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        await scorer.cleanup()

    client = McpClientStreamableHttp()
    try:
        await client.connect_to_streamable_http_server(github_mcp_url, {"Authorization": "Bearer " + github_api_key})
        await client.chat_loop([])
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
