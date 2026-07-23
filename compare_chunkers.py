from pathlib import Path
from chunker import chunk_text as naive_chunk
from recursive_chunker import recursive_chunk
from embedder import embed_chunks, get_model
from vector_store import VectorStore

def load_corpus(data_dir: str = "data") -> str:
    all_text = ""
    for filepath in Path(data_dir).glob("*.txt"):
        all_text += filepath.read_text(encoding="utf-8") + "\n\n"
    return all_text

def build_store(chunks: list[str], save_dir: str) -> VectorStore:
    embeddings = embed_chunks(chunks)
    store = VectorStore(dimension=384)
    store.add(chunks, embeddings)
    store.save(save_dir)
    return store

def compare_query(query: str, naive_store: VectorStore, recursive_store: VectorStore, k: int = 3):
    model = get_model()
    query_embedding = model.encode(query)

    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"\n{'='*60}")

    print("\n--- NAIVE chunker results ---")
    for chunk, score in naive_store.search(query_embedding, k=k):
        print(f"[{score:.3f}] {chunk[:150]}...")

    print("\n--- RECURSIVE chunk results")
    for chunk, score in recursive_store.search(query_embedding, k):
        print(f"[{score:.3f}] {chunk[:150]}...")

if __name__ == "__main__":
    text = load_corpus()

    naive_chunks = naive_chunk(text, chunk_size=500, overlap=50)
    recursive_chunks = recursive_chunk(text, chunk_size=500)

    print(f"Naive chunker: {len(naive_chunks)} chunks")
    print(f"Recursive chunker: {len(recursive_chunks)} chunks")

    print("\nBuilding naive index...")
    naive_store = build_store(naive_chunks, "index_data_naive")

    print("\nBuilding recursive index...")
    recursive_store = build_store(recursive_chunks, "index_data_recursive")

    print("\nBuilt both indexes successfully.. ")

    test_queries = [
    "who invented the airplane?",
    "what year was the airplane invented?",
    "when did nikola tesla migrated to the united states?",
    "what was einstein known for?",
    ]

    for q in test_queries:
        compare_query(q, naive_store, recursive_store)

