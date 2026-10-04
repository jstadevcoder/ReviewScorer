from dotenv import load_dotenv
from logging_helper.logging_config import LoggingConfig
from review_scorer import ReviewScorer

import asyncio
import os
import structlog

logging_config = LoggingConfig(
    os.getenv("LOGGING_CONFIG")
)
logging_config.configure()  # Configures structlog globally

logger = structlog.get_logger(__name__)

async def main() -> None:
    load_dotenv(override=True)
    agent_api_url = os.getenv('AGENT_API_URL')
    agent_api_key = os.getenv('AGENT_API_KEY')
    agent_model = os.getenv('AGENT_MODEL')
    embedding_model = os.getenv('EMBEDDING_MODEL')
    embedding_dimension_str = os.getenv('EMBEDDING_DIMENSION')
    github_mcp_url = os.getenv('GITHUB_MCP_URL')
    github_api_key = os.getenv('GITHUB_API_KEY')

    logger.info(f"Agent Url exists {agent_api_url}")
    logger.info(f"Agent API Key exists and begins {agent_api_key[:8]}")
    logger.info(f"Agent model exists {agent_model}")
    logger.info(f"Embedding model exists {embedding_model}")
    logger.info(f"Embedding dimension exists {embedding_dimension_str}")
    logger.info(f"Github MCP Url exists {github_mcp_url}")
    logger.info(f"Github API Key exists and begins {github_api_key[:10]}")

    if embedding_dimension_str is not None:
        try:
            embedding_dimension = int(embedding_dimension_str)
        except ValueError:
            embedding_dimension = 0
    
    scorer = ReviewScorer(agent_api_url, agent_api_key, agent_model, embedding_model, embedding_dimension)
    try:
        await scorer.connect_to_mcp_client(github_mcp_url, github_api_key)
        await scorer.score_pr_comment("https://github.com/jstadevcoder/ReviewScorer/pull/1")
    except Exception as e:
        logger.exception("ReviewScorer error")
    finally:
        await scorer.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
