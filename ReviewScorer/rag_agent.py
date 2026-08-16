from contextlib import AsyncExitStack
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import Any, List, Type
import faiss
import json
import numpy
import os

class RAGAgent:
    __modelEncodingMap = {"nvidia/nemotron-3-embed-1b:free": "float"}
    def __init__(
        self,
        agentUrl: str,
        agentKey: str,
        embeddingModel: str,
        embeddingDimension: int
    ):
        self.exit_stack = AsyncExitStack()
        self.openai = AsyncOpenAI(
            base_url=agentUrl,
            api_key=agentKey
        )
        self.embeddingModel = embeddingModel
        self.embeddingDimension = embeddingDimension
        self.encodingFormat = self.__modelEncodingMap.get(embeddingModel, "base64")
        
        # Initialize FAISS index (Flat L2 for exact search)
        # Use inner product for cosine similarity if vectors normalized
        self.__reset_index_data()

    def cleanup(self):
        self.openai.close()
    
    async def get_embeddings(self, texts: List[str]) -> numpy.ndarray:
        """
        Get embeddings for a List of texts using the configured embedding model.
        """
        response = await self.openai.embeddings.create(
            input=texts,
            model=self.embeddingModel,
            encoding_format=self.encodingFormat
        )
        embeddings = [item.embedding for item in response.data]
        return numpy.array(embeddings, dtype=numpy.float32)
    
    def retrieve_similar(self, query_embedding: numpy.ndarray, top_k: int = 5) -> List[Any]:
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
            if idx < len(self.index_data):
                results.append(self.index_data[idx])
        
        return results
    
    def add_to_index(self, embeddings: numpy.ndarray, data: List[Any]) -> None:
        if embeddings.shape[0] != len(data):
            raise ValueError("Number of embeddings and data items must match")
        
        if embeddings.shape[1] != self.embeddingDimension:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.embeddingDimension}, got {embeddings.shape[1]}")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        self.faiss_index.add(embeddings)
        self.index_data.extend(data)
    
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
            json.dump([item.dict() for item in self.index_data], f, ensure_ascii=False, indent=2)
    
    def load_index_data(self, filepath: str, load_type: Type[BaseModel]) -> None:
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

        if self.faiss_index.d != self.embeddingDimension:
            self.__reset_index_data()
            raise ValueError(f"Loaded index dimension {self.faiss_index.d} does not match configured dimension {self.embeddingDimension}.")
        
        with open(data_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)
        self.index_data = [load_type.model_validate(data) for data in data_list]

        num_vectors = self.faiss_index.ntotal
        num_data_entries = len(self.index_data)
        if num_vectors != num_data_entries:
            self.__reset_index_data()
            raise ValueError(f"Number of vectors in FAISS index ({num_vectors}) does not match number of data entries ({num_data_entries}).")

    def __reset_index_data(self) -> None:
        self.faiss_index = faiss.IndexFlatIP(self.embeddingDimension)
        self.index_data: List[Any] = []