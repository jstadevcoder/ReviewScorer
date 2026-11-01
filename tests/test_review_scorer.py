from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy
import pytest

import ReviewScorer.review_scorer as review_scorer

class TestReviewScorer:
    @pytest.fixture
    def mocked_dependencies(self):
        with (
            patch.object(
                review_scorer,
                "McpClientStreamableHttp",
            ) as mcp_cls,
            patch.object(
                review_scorer,
                "ChatClient",
            ) as chat_cls,
            patch.object(
                review_scorer,
                "RAGAgent",
            ) as rag_cls,
        ):
            mcp_client = mcp_cls.return_value
            chat_client = chat_cls.return_value
            rag_agent = rag_cls.return_value

            mcp_client.cleanup = AsyncMock()
            mcp_client.connect_to_streamable_http_server = AsyncMock()
            mcp_client.list_tools = AsyncMock()
            mcp_client.execute_tool = AsyncMock()

            chat_client.cleanup = AsyncMock()
            chat_client.process_query = AsyncMock()
            chat_client.clear_history = MagicMock()

            rag_agent.load_index_data = MagicMock()
            rag_agent.save_index_data = MagicMock()
            rag_agent.cleanup = AsyncMock()
            rag_agent.get_embeddings = AsyncMock()
            rag_agent.add_to_index = MagicMock()
            rag_agent.retrieve_similar = MagicMock()
            rag_agent.faiss_index = SimpleNamespace(ntotal=0)

            yield {
                "mcp": mcp_client,
                "chat": chat_client,
                "rag": rag_agent,
                "mcp_cls": mcp_cls,
                "chat_cls": chat_cls,
                "rag_cls": rag_cls,
            }

    @pytest.fixture
    def scorer(self, mocked_dependencies):
        return review_scorer.ReviewScorer(
            agentUrl="https://agent.example.com",
            agentKey="agent-key",
            agentModel="test-model",
            embeddingModel="embedding-model",
            embeddingDimension=1536,
            ragPath="/tmp/rag.index",
        )

    def test_init_creates_clients_and_loads_rag_data(
        self,
        mocked_dependencies,
    ):
        review_scorer.ReviewScorer(
            agentUrl="agent-url",
            agentKey="agent-key",
            agentModel="agent-model",
            embeddingModel="embedding-model",
            embeddingDimension=768,
            ragPath="rag-path",
        )

        mocked_dependencies["mcp_cls"].assert_called_once_with()

        mocked_dependencies["chat_cls"].assert_called_once()
        chat_args = mocked_dependencies["chat_cls"].call_args.args

        assert chat_args[:4] == (
            "agent-url",
            "agent-key",
            "agent-model",
            [],
        )
        assert callable(chat_args[4])

        mocked_dependencies["rag_cls"].assert_called_once()
        rag_args = mocked_dependencies["rag_cls"].call_args.args

        assert rag_args[:4] == (
            "agent-url",
            "agent-key",
            "embedding-model",
            768,
        )
        assert callable(rag_args[4])

        mocked_dependencies["rag"].load_index_data.assert_called_once_with(
            "rag-path",
            review_scorer.CommentOutput,
        )

    def test_init_continues_when_rag_loading_fails(
        self,
        mocked_dependencies,
    ):
        mocked_dependencies["rag"].load_index_data.side_effect = RuntimeError(
            "cannot load index"
        )

        review_scorer.ReviewScorer(
            agentUrl="agent-url",
            agentKey="agent-key",
            agentModel="agent-model",
            embeddingModel="embedding-model",
            embeddingDimension=768,
            ragPath="rag-path",
        )

        mocked_dependencies["rag"].load_index_data.assert_called_once_with(
            "rag-path",
            review_scorer.CommentOutput,
        )

    @pytest.mark.asyncio
    async def test_cleanup(
        self,
        scorer,
        mocked_dependencies,
    ):
        await scorer.cleanup()

        mocked_dependencies["rag"].save_index_data.assert_called_once_with(
            "/tmp/rag.index"
        )
        mocked_dependencies["rag"].cleanup.assert_awaited_once_with()
        mocked_dependencies["mcp"].cleanup.assert_awaited_once_with()
        mocked_dependencies["chat"].cleanup.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_connect_to_mcp_client(
        self,
        scorer,
        mocked_dependencies,
    ):
        tools = [
            {"function": {"name": "get_pull_request"}},
            {"function": {"name": "get_file"}},
        ]
        mocked_dependencies["mcp"].list_tools.return_value = tools

        await scorer.connect_to_mcp_client(
            gitUrl="https://git.example.com",
            gitKey="git-key",
        )

        mocked_dependencies[
            "mcp"
        ].connect_to_streamable_http_server.assert_awaited_once_with(
            "https://git.example.com",
            {"Authorization": "Bearer git-key"},
        )
        mocked_dependencies["mcp"].list_tools.assert_awaited_once_with()

        assert scorer._ReviewScorer__tools == tools

    @pytest.mark.asyncio
    async def test_connect_to_mcp_client_replaces_existing_tools(
        self,
        scorer,
        mocked_dependencies,
    ):
        scorer._ReviewScorer__tools.append(
            {"function": {"name": "old_tool"}}
        )

        new_tools = [{"function": {"name": "new_tool"}}]
        mocked_dependencies["mcp"].list_tools.return_value = new_tools

        await scorer.connect_to_mcp_client("git-url", "git-key")

        assert scorer._ReviewScorer__tools == new_tools

    @pytest.mark.asyncio
    async def test_tool_callback(
        self,
        scorer,
        mocked_dependencies,
    ):
        mocked_dependencies["mcp"].execute_tool.return_value = {
            "result": "ok"
        }

        result = await scorer._ReviewScorer__tool_callback(
            "get_file",
            {"path": "README.md"},
            "tool-call-id",
        )

        assert result == {"result": "ok"}
        mocked_dependencies["mcp"].execute_tool.assert_awaited_once_with(
            "get_file",
            {"path": "README.md"},
            "tool-call-id",
        )

    def test_rag_key_callback(self, scorer):
        data = SimpleNamespace(comment="comment text")

        result = scorer._ReviewScorer__rag_key_callback(data)

        assert result == "comment text"

    @pytest.mark.asyncio
    async def test_review_initial_pr(
        self,
        scorer,
        mocked_dependencies,
    ):
        instructions = [
            "instruction 1",
            "instruction 2",
        ]
        ending_instruction = (
            "ending instruction",
            {"response_format": "review"},
        )
        expected_review = [
            SimpleNamespace(instruction="review comment"),
        ]

        with (
            patch.object(
                review_scorer.ReviewAgentInstructions,
                "instructions_list",
                return_value=instructions,
            ) as instructions_list,
            patch.object(
                review_scorer.ReviewAgentInstructions,
                "ending_instruction",
                return_value=ending_instruction,
            ) as ending,
        ):
            mocked_dependencies["chat"].process_query.side_effect = [
                {"intermediate": 1},
                {"intermediate": 2},
                expected_review,
            ]

            result = await scorer._ReviewScorer__review_initial_pr(
                "https://git.example.com/pr/1"
            )

        assert result == expected_review

        instructions_list.assert_called_once_with(
            {"url": "https://git.example.com/pr/1"}
        )
        ending.assert_called_once_with(
            {"url": "https://git.example.com/pr/1"}
        )

        calls = mocked_dependencies["chat"].process_query.await_args_list

        assert calls[0].args == ("instruction 1", [])
        assert calls[1].args == ("instruction 2", [])
        assert calls[2].args == (
            "ending instruction",
            [],
            {"response_format": "review"},
        )

    @pytest.mark.asyncio
    async def test_get_rag_query_returns_empty_when_index_is_empty(
        self,
        scorer,
        mocked_dependencies,
    ):
        starting_instruction = (
            "find relevant comments",
            {"response_format": "pr_comments"},
        )
        pr_comments = [
            SimpleNamespace(comment="new comment"),
        ]

        with patch.object(
            review_scorer.ScorerAgentInstructions,
            "starting_instruction",
            return_value=starting_instruction,
        ) as starting:
            mocked_dependencies["chat"].process_query.return_value = (
                pr_comments
            )
            mocked_dependencies["rag"].faiss_index.ntotal = 0

            result = await scorer._ReviewScorer__get_rag_query(
                "pr-url",
                [SimpleNamespace(instruction="review instruction")],
            )

        assert result == ""

        starting.assert_called_once_with(
            {
                "url": "pr-url",
                "comments": "review instruction",
            }
        )
        mocked_dependencies["chat"].process_query.assert_awaited_once_with(
            "find relevant comments",
            [],
            {"response_format": "pr_comments"},
        )
        mocked_dependencies["rag"].get_embeddings.assert_not_awaited()
        mocked_dependencies["rag"].retrieve_similar.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_rag_query_returns_similar_comments(
        self,
        scorer,
        mocked_dependencies,
    ):
        starting_instruction = (
            "find relevant comments",
            {"response_format": "pr_comments"},
        )
        pr_comments = [
            SimpleNamespace(comment="new comment 1"),
            SimpleNamespace(comment="new comment 2"),
        ]
        similar_comments = [
            {"comment": "old comment", "score": 0.92},
        ]
        embeddings = numpy.array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
            ]
        )

        with patch.object(
            review_scorer.ScorerAgentInstructions,
            "starting_instruction",
            return_value=starting_instruction,
        ):
            mocked_dependencies["chat"].process_query.return_value = (
                pr_comments
            )
            mocked_dependencies["rag"].faiss_index.ntotal = 3
            mocked_dependencies["rag"].get_embeddings.return_value = embeddings
            mocked_dependencies[
                "rag"
            ].retrieve_similar.return_value = similar_comments

            result = await scorer._ReviewScorer__get_rag_query(
                "pr-url",
                [
                    SimpleNamespace(
                        instruction="first review instruction"
                    ),
                    SimpleNamespace(
                        instruction="second review instruction"
                    ),
                ],
            )

        assert result == 'Given these past PR comments with their metadata and scores:\n[{"comment":"old comment","score":0.92}]\n'
        
        mocked_dependencies["chat"].process_query.assert_awaited_once_with(
            "find relevant comments",
            [],
            {"response_format": "pr_comments"},
        )
        mocked_dependencies["rag"].get_embeddings.assert_awaited_once_with(
            ["new comment 1", "new comment 2"]
        )
        mocked_dependencies["rag"].retrieve_similar.assert_called_once_with(
            embeddings
        )

    @pytest.mark.asyncio
    async def test_score_pr_comment(
        self,
        scorer,
        mocked_dependencies,
    ):
        review_output = [
            SimpleNamespace(instruction="review instruction"),
        ]
        comment_output = [
            SimpleNamespace(comment="scored comment 1"),
            SimpleNamespace(comment="scored comment 2"),
        ]
        embeddings = numpy.array(
            [
                [0.5, 0.6],
                [0.7, 0.8],
            ]
        )

        scorer._ReviewScorer__review_initial_pr = AsyncMock(
            return_value=review_output
        )
        scorer._ReviewScorer__get_rag_query = AsyncMock(
            return_value="previous context\n"
        )

        ending_instruction = (
            "score the comments",
            {"response_format": "comment_outputs"},
        )

        with patch.object(
            review_scorer.ScorerAgentInstructions,
            "ending_instruction",
            return_value=ending_instruction,
        ) as ending:
            mocked_dependencies["chat"].process_query.return_value = (
                comment_output
            )
            mocked_dependencies["rag"].get_embeddings.return_value = (
                embeddings
            )

            result = await scorer.score_pr_comment("pr-url")

        assert result is None

        scorer._ReviewScorer__review_initial_pr.assert_awaited_once_with(
            "pr-url"
        )
        scorer._ReviewScorer__get_rag_query.assert_awaited_once_with(
            "pr-url",
            review_output,
        )

        mocked_dependencies["chat"].clear_history.assert_called_once_with()

        ending.assert_called_once_with({"url": "pr-url"})
        mocked_dependencies["chat"].process_query.assert_awaited_once_with(
            "previous context\nscore the comments",
            [],
            {"response_format": "comment_outputs"},
        )
        mocked_dependencies["rag"].get_embeddings.assert_awaited_once_with(
            ["scored comment 1", "scored comment 2"]
        )
        mocked_dependencies["rag"].add_to_index.assert_called_once_with(
            embeddings,
            comment_output,
        )

    @pytest.mark.asyncio
    async def test_score_pr_comment_with_empty_rag_context(
        self,
        scorer,
        mocked_dependencies,
    ):
        scorer._ReviewScorer__review_initial_pr = AsyncMock(
            return_value=[]
        )
        scorer._ReviewScorer__get_rag_query = AsyncMock(
            return_value=""
        )

        ending_instruction = ("score", {"format": "json"})
        scored_comments = [
            SimpleNamespace(comment="comment"),
        ]

        with patch.object(
            review_scorer.ScorerAgentInstructions,
            "ending_instruction",
            return_value=ending_instruction,
        ):
            mocked_dependencies["chat"].process_query.return_value = (
                scored_comments
            )
            mocked_dependencies["rag"].get_embeddings.return_value = (
                numpy.array([[1.0]])
            )

            await scorer.score_pr_comment("pr-url")

        mocked_dependencies["chat"].process_query.assert_awaited_once_with(
            "score",
            [],
            {"format": "json"},
        )