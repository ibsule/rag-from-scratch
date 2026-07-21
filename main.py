from pathlib import Path
from chunker import chunk_text
from embedder import embed_chunks, get_model
from retriever import retrieve
from prompt import build_prompt
from generator import generate_answer
from vector_store import VectorStore

def load_corpus(data_dir: str = "data") -> str:
    """Concatenate all .txt files in the data directory into one string"""
    all_text = ""
    for filepath in Path(data_dir).glob("*.txt"):
        all_text += filepath.read_text(encoding="utf-8") + "\n\n"
    return all_text

def build_or_load_store() -> VectorStore:
    if VectorStore.exists():
        print("found existing index, loading from disk...")
        return VectorStore.load()
    
    print("No existing index found. building from scratch...")
    text = load_corpus()
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    embeddings = embed_chunks(chunks)

    store = VectorStore(dimension=384)
    store.add(chunks, embeddings)
    store.save()
    print(f"Indexed {len(chunks)} chunks and saved to disk.")
    return store

def answer_question(query: str, store: VectorStore) -> str:
    model = get_model()
    query_embedding = model.encode(query)
    results = store.search(query_embedding, k=3)
    retrieved_texts = [chunk for chunk, store in results]
    prompt = build_prompt(query, retrieved_texts)
    return generate_answer(prompt)


def main():
    store = build_or_load_store()
    print(f"Ready. {len(store.chunks)} chunks indexed.\n")

    print("Ask as question (or type 'quit' to exit):\n")
    while True:
        query = input("q> ").strip()
        if query.lower() in ('quit', 'exit', 'q'):
            break
        if not query:
            continue
        answer = answer_question(query, store)
        print(f"\na< {answer}\n")

if __name__ == "__main__":
    main()