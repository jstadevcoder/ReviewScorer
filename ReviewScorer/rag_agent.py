from collections.abc import Callable
from contextlib import AsyncExitStack
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import Any
import faiss
import json
import numpy
import os

class RAGAgent:
    __model_encoding_map = {"nvidia/nemotron-3-embed-1b:free": "float"}
    def __init__(
        self,
        agent_url: str,
        agent_key: str,
        embedding_model: str,
        embedding_dimension: int,
        get_key_callback: Callable[[Any], str] | None = None
    ):
        self.exit_stack = AsyncExitStack()
        self.openai = AsyncOpenAI(
            base_url=agent_url,
            api_key=agent_key
        )
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.encoding_format = self.__model_encoding_map.get(embedding_model, "base64")
        self.__get_key_callback = get_key_callback
        
        # Initialize FAISS index (Flat L2 for exact search)
        # Use inner product for cosine similarity if vectors normalized
        self.__reset_index_data()

    async def cleanup(self):
        await self.openai.close()
    
    async def get_embeddings(self, texts: list[str]) -> numpy.ndarray:
        """
        Get embeddings for a List of texts using the configured embedding model.
        """
        response = await self.openai.embeddings.create(
            input=texts,
            model=self.embedding_model,
            encoding_format=self.encoding_format
        )
        embeddings = [item.embedding for item in response.data]
        return numpy.array(embeddings, dtype=numpy.float32)
    
    def retrieve_similar(self, query_embedding: numpy.ndarray, top_k: int = 5) -> list[Any]:
        """
        Retrieve top-k similar comments from the index for a query embedding.
        """
        
        if self.faiss_index.ntotal == 0 or len(self.index_data) == 0:
            # Index or data empty, return empty List immediately
            return []
        
        if top_k > self.faiss_index.ntotal:
            top_k = self.faiss_index.ntotal

        # Normalize query embedding
        normalized_embedding = query_embedding.astype(numpy.float32).copy()
        faiss.normalize_L2(normalized_embedding)

        # distances is not used, it can be used to rank or filter results further based on similarity score
        distances, indices = self.faiss_index.search(normalized_embedding, top_k)
        
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.index_data):
                results.append(self.index_data[idx])
        
        return results
    
    def add_to_index(self, embeddings: numpy.ndarray, data: list[Any]) -> None:
        if embeddings.shape[0] != len(data):
            raise ValueError("Number of embeddings and data items must match")
        
        if embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.embedding_dimension}, got {embeddings.shape[1]}")
        
        # Make a copy so the caller's array is not modified.
        embeddings = numpy.asarray(embeddings, dtype=numpy.float32).copy()
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        new_embeddings = []
        new_data = []

        key_callback = self.__get_key_callback
        for embedding, item in zip(embeddings, data):
            if key_callback is not None:
                key = key_callback(item)
            else:
                key = self.__get_data_key(item)
            
            if key in self.data_keys:
                continue

            self.data_keys.add(key)
            new_embeddings.append(embedding)
            new_data.append(item)

        if not new_embeddings:
            return

        new_embeddings_array = numpy.asarray(
            new_embeddings,
            dtype=numpy.float32,
        )
        
        self.faiss_index.add(new_embeddings_array)
        self.index_data.extend(new_data)
    
    def save_index_data(self, filepath: str = None) -> None:
        """
        Save the current index (embeddings and associated data) to disk.
        
        :param filepath: base path without extension. Will save two files:
        <filepath>_faiss.idx and <filepath>_data.json
        """
        if self.faiss_index.ntotal == 0 or len(self.index_data) == 0:
            return

        if not filepath:
            # Default to '[projectDirectory]]/RAG_Data/index'
            project_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(project_dir, "RAG_Data", "index")

        # Create directory if it doesn't exist
        dir_path = os.path.dirname(filepath)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if self.faiss_index.ntotal == 0:
            raise ValueError("Index is empty, nothing to save.")
        
        faiss.write_index(self.faiss_index, f"{filepath}_faiss.idx")

        # Save JSON-serializable data List
        with open(f"{filepath}_data.json", "w", encoding="utf-8") as f:
            json.dump(self.index_data, f, ensure_ascii=False, indent=2, default=self.__serialize_value)
    
    def load_index_data(self, filepath: str, load_type: type) -> None:
        """
        Load index from disk.
        
        :param filepath: base path without extension where files are stored:
        <filepath>_faiss.idx and <filepath>_data.json
        """
        if not filepath:
            # Default to '[projectDirectory]]/RAG_Data/index'
            project_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(project_dir, "RAG_Data", "index")
        
        faiss_idx_path = f"{filepath}_faiss.idx"
        data_path = f"{filepath}_data.json"
        
        if not os.path.exists(faiss_idx_path):
            return
        if not os.path.exists(data_path):
            return
        
        self.faiss_index = faiss.read_index(faiss_idx_path)

        if self.faiss_index.d != self.embedding_dimension:
            self.__reset_index_data()
            raise ValueError(f"Loaded index dimension {self.faiss_index.d} does not match configured dimension {self.embedding_dimension}.")
        
        with open(data_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        if isinstance(load_type, type) and issubclass(load_type, BaseModel):
            self.index_data = [load_type.model_validate(data) for data in data_list]
        else:
            self.index_data = [load_type(**data) if isinstance(data, dict) else load_type(data) for data in data_list]

        num_vectors = self.faiss_index.ntotal
        num_data_entries = len(self.index_data)
        if num_vectors != num_data_entries:
            self.__reset_index_data()
            raise ValueError(f"Number of vectors in FAISS index ({num_vectors}) does not match number of data entries ({num_data_entries}).")

        key_callback = self.__get_key_callback
        self.data_keys = {
            key_callback(item) if key_callback is not None
            else self.__get_data_key(item)
            for item in self.index_data
        }

    def __reset_index_data(self) -> None:
        self.data_keys: set[str] = set()
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
        self.index_data: list[Any] = []

    def __get_data_key(self, item: Any) -> str:
        return json.dumps(
            item,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=self.__serialize_value
        )
    
    def __serialize_value(self, value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")

        if hasattr(value, "__dict__"):
            return value.__dict__

        raise TypeError(
            f"Object of type {type(value).__name__} is not JSON serializable"
        )