import faiss
import numpy as np
import json
from pathlib import Path

class VectorStore:
    def __init__(self, dimension: int = 384):
        # IndexFlatIp = exact search using inner product (dot product).
        # Since our embeddings are normalised, inner product == cosine similarty
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = []
        self.dimension = dimension

    def add(self, chunks: list[str], embeddings: np.ndarray):
        embeddings = np.array(embeddings).astype("float32") # FAISS requires float32
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        query_embedding = np.array([query_embedding]).astype("float32")
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: # FAISS returns -1 if fewer than k results exist
                continue
            results.append((self.chunks[idx], float(score)))
        
        return results
    
    def save(self, dir_path: str = "index_data"):
        """persist the FAISS index and the chunk text to disk"""
        Path(dir_path).mkdir(exist_ok=True)
        faiss.write_index(self.index, f"{dir_path}/index.faiss")
        with open(f"{dir_path}/chunks.json", "w") as f:
            json.dump(self.chunks, f)

    @classmethod
    def load(cls, dir_path: str = "index_data", dimension: int = 384) -> "VectorStore":
        """Load a previously saved index and chunks from disk."""
        store = cls(dimension=dimension)
        store.index = faiss.read_index(f"{dir_path}/index.faiss")
        with open(f"{dir_path}/chunks.json", "r") as f:
            store.chunks = json.load(f)
        return store
    
    @staticmethod
    def exists(dir_path: str = "index_data") -> bool: 
        """check if a saved index already exists on disk"""
        return Path(f"{dir_path}/index.faiss").exists()