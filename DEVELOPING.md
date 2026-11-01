
# ReviewScorer development

## Prerequisites

* Install `uv`: https://docs.astral.sh/uv/getting-started/installation/#installation-methods
* Procure relevant API keys. See the "API keys" section below. Currently we only integrate with Github and OpenRouter.

## Getting started

First check out the package, and then run `uv sync` to all dependencies.

Then update the `.env` file with the necesary variables. See the example below:

```
AGENT_API_KEY=fake-openrouter-api-key
GITHUB_API_KEY=fake-github-api-key
AGENT_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
AGENT_API_URL=https://openrouter.ai/api/v1
GITHUB_MCP_URL=https://api.githubcopilot.com/mcp
EMBEDDING_MODEL=nvidia/nemotron-3-embed-1b:free
EMBEDDING_DIMENSION=2048
#LOGGING_CONFIG can be left out if just logging to console
```

The project can now be run with

```bash
uv run ./ReviewScorer/app.py [-h] [--reviewUrl REVIEWURL]

options:
  -h, --help            Show this help message and exit
  --reviewUrl REVIEWURL The review Url to perform a review on.
```

### Tests

To run the tests, use one of the following

```bash
# all tests
uv run pytest
# single file
uv run pytest tests/llmclients/test_chat_client.py
# single test
uv run pytest uv run pytest tests/llmclients/test_chat_client.py::test_that_doesnt_exist
# single test within a class
uv run pytest uv run pytest tests/llmclients/test_chat_client.py::TestChatClient::test_process_simple_query
```

### Formatting

Ruff has been added to the project to help with formatting. Run it with:

```bash
# check for formatting issues
uv run ruff check
# check for formatting issues in specific files
uv run ruff check ./ReviewScorer/llmclients/
# auto-fix formatting issues
uv run ruff check --fix 
# auto-fix formatting issues in specific files
uv run ruff check --fix  ./tests/llmclients/
```

## API keys

### OpenRouter

Currently OpenRouter is the only back-end LLM integration that we support.

OpenRouter API key docs: https://openrouter.ai/docs/guides/overview/auth/management-api-keys

Configure this in your `.env` file with the `AGENT_API_KEY` variable.

### Github

Github API key docs: https://docs.github.com/en/rest/authentication/authenticating-to-the-rest-api

Configure this in your `.env` file with the `GITHUB_API_KEY` variable.

The token needs permissions to read PRs.

## General Project Structure
REVIEWSCORER/
├── ReviewScorer/
│   ├── agent_instructions/ #Contains instructions used to feed the LLM to score a PR
│   │   └── ...
│   ├── agent_outputs/ #Contains structured output schemas for LLMs to produce
│   │   └── ...
│   ├── clients/ #The chat implementation to connect to a LLM model, Github MCP and RAG
│   │   └── ...
│   ├── enums/ #The categories the LLM produces to categorise a review
│   │   └── ...
│   ├── logging_helper/ #Helper to output logs to console/elsewhere
│   │   └── ...
│   ├── RAG_Data/ #Historic review scores are saved here
│   │   └── ...
│   ├── __init__.py
│   ├── app.py
│   └── review_scorer.py
├── tests/ #Tests for the ReviewScorer code
│   ├── test_review_scorer.py
├── .env_template
├── dev-requirements.txt
├── pyproject.toml
├── README.md
└── ...