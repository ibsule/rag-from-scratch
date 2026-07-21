import faiss
import numpy as np

class VectorStore:
    def __init__(self, dimension: int = 384):
        # IndexFlatIp = exact search using inner product (dot product).
        # Since our embeddings are normalised, inner product == cosine similarty
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = []

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