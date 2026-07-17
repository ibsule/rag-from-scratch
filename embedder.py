from sentence_transformers import SentenceTransformer
import numpy as np

_model = None

def get_model() -> SentenceTransformer:
    """Load the model once and reuse it (loading is slow, embeding is fast)"""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embed_chunks(chunks: list[str]) -> np.ndarray:
    """Embed a list of text chunks into a 2D array of shape (num_chunks, 384)."""
    model = get_model()
    return model.encode(chunks, show_progress_bar=True)