import time
from main import load_corpus
from chunker import chunk_text
from embedder import embed_chunks
from embedder import get_model
from retriever import retrieve
from vector_store import VectorStore

def make_large_chunk_set(multiplier: int = 30) -> list[str]:
    """Duplicate real corpus's chunks to similate larger dataset."""
    text = load_corpus()
    base_chunks = chunk_text(text, chunk_size=500, overlap=50)
    
    return base_chunks * multiplier

chunks = make_large_chunk_set(multiplier=2)
print(f"Benchmarking with {len(chunks)} chunks")

print("Embedding...")
embeddings = embed_chunks(chunks)

store = VectorStore(dimension=384)
store.add(chunks, embeddings)

query = "who invented the airplane?"
model = get_model()
query_embedding = model.encode(query)

start = time.time()
results = store.search(query_embedding, k=3)
elapsed = time.time() - start

print(f"\nFAISS search took {elapsed:.4f} seconds for {len(chunks)} chunks")