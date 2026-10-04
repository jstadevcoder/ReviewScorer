from pathlib import Path
from pydantic import BaseModel
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import faiss
import json
import numpy
import pytest
import shutil

from ReviewScorer.clients.rag_agent import RAGAgent

class Item(BaseModel):
    id: int
    text: str

class TestRagAgent:
    @pytest.fixture
    def agent(self):
        with patch("ReviewScorer.clients.rag_agent.AsyncOpenAI") as openai_class:
            openai_class.return_value = MagicMock()

            yield RAGAgent(
                agent_url="https://ragagent.test/v1",
                agent_key="test-key",
                embedding_model="test-model",
                embedding_dimension=3,
            )

    @pytest.fixture
    def index_filepath(self, tmp_path: Path):
        directory = tmp_path / "rag-index"
        directory.mkdir()
        filepath = directory / "index"

        try:
            yield str(filepath)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_initializes_openai_client_and_empty_index(self):
        with patch("ReviewScorer.clients.rag_agent.AsyncOpenAI") as openai_class:
            client = MagicMock()
            openai_class.return_value = client

            agent = RAGAgent(
                agent_url="https://ragagent.test/v1",
                agent_key="secret",
                embedding_model="test-model",
                embedding_dimension=3,
            )

            openai_class.assert_called_once_with(
                base_url="https://ragagent.test/v1",
                api_key="secret",
            )

            assert agent.embedding_model == "test-model"
            assert agent.embedding_dimension == 3
            assert agent.encoding_format == "base64"
            assert agent.faiss_index.ntotal == 0
            assert agent.index_data == []
            assert agent.data_keys == set()

    def test_uses_float_encoding_for_configured_model(self):
        with patch("ReviewScorer.clients.rag_agent.AsyncOpenAI"):
            agent = RAGAgent(
                agent_url="url",
                agent_key="key",
                embedding_model="nvidia/nemotron-3-embed-1b:free",
                embedding_dimension=3,
            )

        assert agent.encoding_format == "float"

    @pytest.mark.asyncio
    async def test_get_embeddings_returns_float32_numpy_array(self, agent):
        agent.openai.embeddings.create = AsyncMock(
            return_value=SimpleNamespace(
                data=[
                    SimpleNamespace(embedding=[1, 2, 3]),
                    SimpleNamespace(embedding=[4, 5, 6]),
                ]
            )
        )

        result = await agent.get_embeddings(["first", "second"])

        agent.openai.embeddings.create.assert_awaited_once_with(
            input=["first", "second"],
            model="test-model",
            encoding_format="base64",
        )

        assert isinstance(result, numpy.ndarray)
        assert result.dtype == numpy.float32

        numpy.testing.assert_array_equal(
            result,
            numpy.array([[1, 2, 3], [4, 5, 6]], dtype=numpy.float32),
        )

    @pytest.mark.asyncio
    async def test_cleanup_closes_openai_client(self, agent):
        agent.openai.close = AsyncMock()

        await agent.cleanup()

        agent.openai.close.assert_awaited_once()

    def test_retrieve_from_empty_index_returns_empty_list(self, agent):
        query = numpy.array([[1, 0, 0]], dtype=numpy.float32)

        assert agent.retrieve_similar(query) == []

    def test_add_to_index_and_retrieve_similar_items(self, agent):
        embeddings = numpy.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=numpy.float32,
        )

        data = [
            {"id": "x", "text": "x"},
            {"id": "y", "text": "y"},
            {"id": "z", "text": "z"},
        ]

        agent.add_to_index(embeddings, data)

        assert agent.faiss_index.ntotal == 3
        assert agent.index_data == data
        assert len(agent.data_keys) == 3

        results = agent.retrieve_similar(
            numpy.array([[0.9, 0.1, 0]], dtype=numpy.float32),
            top_k=2,
        )

        assert results == [
            {"id": "x", "text": "x"},
            {"id": "y", "text": "y"},
        ]

    def test_add_to_index_does_not_modify_input_embeddings(self, agent):
        embeddings = numpy.array(
            [
                [3, 0, 0],
                [0, 4, 0],
            ],
            dtype=numpy.float32,
        )
        original = embeddings.copy()

        agent.add_to_index(embeddings, ["x", "y"])

        numpy.testing.assert_array_equal(embeddings, original)

    def test_add_to_index_rejects_mismatched_data_length(self, agent):
        embeddings = numpy.ones((2, 3), dtype=numpy.float32)

        with pytest.raises(
            ValueError,
            match="Number of embeddings and data items must match",
        ):
            agent.add_to_index(embeddings, ["only-one-item"])

    def test_add_to_index_rejects_wrong_embedding_dimension(self, agent):
        embeddings = numpy.ones((2, 2), dtype=numpy.float32)

        with pytest.raises(
            ValueError,
            match=r"Embedding dimension mismatch\. Expected 3, got 2",
        ):
            agent.add_to_index(embeddings, ["x", "y"])

    def test_duplicate_items_are_not_added(self, agent):
        embeddings = numpy.array(
            [
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=numpy.float32,
        )

        agent.add_to_index(embeddings, [{"id": 1}, {"id": 2}])

        agent.add_to_index(
            numpy.array(
                [
                    [0, 0, 1],
                    [1, 1, 1],
                ],
                dtype=numpy.float32,
            ),
            [{"id": 1}, {"id": 3}],
        )

        assert agent.faiss_index.ntotal == 3
        assert agent.index_data == [
            {"id": 1},
            {"id": 2},
            {"id": 3},
        ]

    def test_custom_key_callback_controls_duplicate_detection(self):
        key_callback = lambda item: item["id"]

        with patch("ReviewScorer.clients.rag_agent.AsyncOpenAI"):
            agent = RAGAgent(
                agent_url="url",
                agent_key="key",
                embedding_model="test-model",
                embedding_dimension=3,
                get_key_callback=key_callback,
            )

        agent.add_to_index(
            numpy.array([[1, 0, 0]], dtype=numpy.float32),
            [{"id": "same", "value": "first"}],
        )

        agent.add_to_index(
            numpy.array([[0, 1, 0]], dtype=numpy.float32),
            [{"id": "same", "value": "second"}],
        )

        assert agent.faiss_index.ntotal == 1
        assert agent.index_data == [
            {"id": "same", "value": "first"},
        ]

    def test_retrieve_top_k_is_capped_to_index_size(self, agent):
        agent.add_to_index(
            numpy.array([[1, 0, 0]], dtype=numpy.float32),
            ["only-item"],
        )

        results = agent.retrieve_similar(
            numpy.array([[1, 0, 0]], dtype=numpy.float32),
            top_k=100,
        )

        assert results == ["only-item"]

    def test_save_and_load_index_data(self, agent, index_filepath):
        data = [
            {"id": 1, "text": "first"},
            {"id": 2, "text": "second"},
        ]

        embeddings = numpy.array(
            [
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=numpy.float32,
        )

        agent.add_to_index(embeddings, data)
        agent.save_index_data(index_filepath)

        assert Path(f"{index_filepath}_faiss.idx").exists()
        assert Path(f"{index_filepath}_data.json").exists()

        with patch("ReviewScorer.clients.rag_agent.AsyncOpenAI"):
            loaded_agent = RAGAgent(
                agent_url="url",
                agent_key="key",
                embedding_model="test-model",
                embedding_dimension=3,
            )

        loaded_agent.load_index_data(index_filepath, dict)

        assert loaded_agent.faiss_index.ntotal == 2
        assert loaded_agent.index_data == data

    def test_save_empty_index_does_nothing(self, agent, index_filepath):
        agent.save_index_data(index_filepath)

        assert not Path(f"{index_filepath}_faiss.idx").exists()
        assert not Path(f"{index_filepath}_data.json").exists()

    def test_load_missing_files_does_nothing(self, agent, index_filepath):
        agent.load_index_data(index_filepath, dict)

        assert agent.faiss_index.ntotal == 0
        assert agent.index_data == []

    def test_load_index_data_into_pydantic_models(self, agent, index_filepath):
        data = [
            {"id": 1, "text": "first"},
            {"id": 2, "text": "second"},
        ]

        agent.add_to_index(
            numpy.array(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                ],
                dtype=numpy.float32,
            ),
            data,
        )

        agent.save_index_data(index_filepath)

        with patch("ReviewScorer.clients.rag_agent.AsyncOpenAI"):
            loaded_agent = RAGAgent(
                agent_url="url",
                agent_key="key",
                embedding_model="test-model",
                embedding_dimension=3,
            )

        loaded_agent.load_index_data(index_filepath, Item)

        assert loaded_agent.index_data == [
            Item(id=1, text="first"),
            Item(id=2, text="second"),
        ]

    def test_load_rejects_wrong_index_dimension(self, agent, index_filepath):
        wrong_index = faiss.IndexFlatIP(2)

        faiss.write_index(
            wrong_index,
            f"{index_filepath}_faiss.idx",
        )

        with open(f"{index_filepath}_data.json", "w", encoding="utf-8") as file:
            json.dump([], file)

        with pytest.raises(
            ValueError,
            match=r"Loaded index dimension 2 does not match configured dimension 3\.",
        ):
            agent.load_index_data(index_filepath, dict)

    def test_load_rejects_vector_data_count_mismatch(
        self,
        agent,
        index_filepath,
    ):
        index = faiss.IndexFlatIP(3)
        index.add(numpy.array([[1, 0, 0]], dtype=numpy.float32))

        faiss.write_index(
            index,
            f"{index_filepath}_faiss.idx",
        )

        with open(f"{index_filepath}_data.json", "w", encoding="utf-8") as file:
            json.dump([], file)

        with pytest.raises(
            ValueError,
            match=(
                r"Number of vectors in FAISS index \(1\) does not match "
                r"number of data entries \(0\)\."
            ),
        ):
            agent.load_index_data(index_filepath, dict)

    def test_pydantic_items_are_serialized_when_saving(
        self,
        agent,
        index_filepath,
    ):
        items = [
            Item(id=1, text="first"),
            Item(id=2, text="second"),
        ]

        agent.add_to_index(
            numpy.array(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                ],
                dtype=numpy.float32,
            ),
            items,
        )

        agent.save_index_data(index_filepath)

        with open(
            f"{index_filepath}_data.json",
            encoding="utf-8",
        ) as file:
            saved_data = json.load(file)

        assert saved_data == [
            {"id": 1, "text": "first"},
            {"id": 2, "text": "second"},
        ]
