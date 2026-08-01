
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
```

The project can now be run with

```bash
uv run ./ReviewScorer/app.py
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

