import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Measures how similar two vectors are, from -1 (opposite) to 1 (identical direction)"""
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    return dot_product / (norm_a * norm_b)