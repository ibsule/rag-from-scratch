import numpy as np
from embedder import get_model
from similarity import cosine_similarity

def retrieve(query: str, chunks: list[str], embeddings: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
    """REturn the top-k most relevant chunks for a query, with their similarity scores"""
    model = get_model()
    query_embedding = model.encode(query) # embed the query the same way as chunks

    scores = []
    for i, chunk_embedding in enumerate(embeddings):
        score = cosine_similarity(query_embedding, chunk_embedding)
        scores.append((i, score))
    
    # sort by score, descending, take top k
    scores.sort(key=lambda x: x[1], reverse=True)
    tok_k = scores[:k]

    return [(chunks[i], score) for i, score in tok_k]
